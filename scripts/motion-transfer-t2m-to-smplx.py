#!/usr/bin/env python
"""Motion transfer from Text2Motion (T2M/HumanML3D) skeleton to SMPL-X.

This script demonstrates the world-space delta retargeting strategy described in
`context/tasks/task-fix-humanml-to-smplx-error.md`:

1) Generate two random T2M skeleton poses (A=bind/calibration, B=motion).
2) Create a neutral SMPL-X model via the unified model wrapper.
3) Establish joint name mapping between T2M(22) and SMPL-X core body joints.
4) Fit SMPL-X (pose+global+shape) to T2M A joints in 3D, allowing shape to deform.
5) Transfer motion from T2M A→B using world-rotation deltas:
     R_tgt^W_j(t) = R_src^W_j(t) · (R_src^W_j(0))^{-1} · R_tgt^W_j(0)
   then recover target local rotations and update the SMPL-X NamedPose.
6) Compare the final SMPL-X joints to the T2M B joints (MPJPE on mapped joints).

Notes
- Both T2M and SMPL-X are treated here as Y-up to avoid additional up-axis logic.
- This script requires a local SMPL(-X) model directory with proper licenses.

Usage
-----
python scripts/motion-transfer-t2m-to-smplx.py \
  --body-model-dir data/body_models \
  --gender neutral \
  --iters 200 \
  --device auto

"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import smplx  # type: ignore
except Exception as e:  # pragma: no cover - runtime import check
    print("[error] smplx package is required. Install it per project docs.")
    raise

from smplx_toolbox.core import (
    NamedPose,
    UnifiedSmplInputs,
    UnifiedSmplModel,
)
from smplx_toolbox.core.constants import CoreBodyJoint, ModelType
from smplx_toolbox.fitting.helper import SmplKeypointFittingHelper
from smplx_toolbox.utils.humanml_mapping import (
    T2MSkeleton,
    create_neutral_t2m_skeleton,
    humanml_joint_mapping,
)
try:
    from scipy.spatial.transform import Rotation as R  # type: ignore
except Exception as e:  # pragma: no cover - runtime import check
    print("[error] scipy is required for 3D rotation math. Install via 'pip install scipy'.")
    raise


# ----------------------------
# Math helpers (NumPy / Torch)
# ----------------------------


def aa_to_mat_np(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (rotvec) to rotation matrix using SciPy."""
    a = np.asarray(aa, dtype=np.float32).reshape(3)
    return R.from_rotvec(a).as_matrix().astype(np.float32)


def mat_to_rotvec_torch(M: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Rotation matrix (...,3,3) to axis-angle rotvec (...,3) using SciPy."""
    if isinstance(M, torch.Tensor):
        m_np = M.detach().cpu().numpy().astype(np.float32)
    else:
        m_np = M.astype(np.float32)
    rv = R.from_matrix(m_np.reshape(-1, 3, 3)).as_rotvec().astype(np.float32)
    rv_t = torch.from_numpy(rv).view(*(m_np.shape[:-2] + (3,)))
    return rv_t


def mat_to_6d_np(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3,3) -> 6D rep (6,) by stacking first two columns."""
    return np.concatenate([R[:, 0], R[:, 1]], axis=0).astype(np.float32)


def parents_from_t2m() -> List[int]:
    # Reconstruct parents from humanml_mapping internals
    from smplx_toolbox.utils.humanml_mapping import T2M_KINEMATIC_CHAIN

    parents = [0] * 22
    parents[0] = -1
    for chain in T2M_KINEMATIC_CHAIN:
        for j in range(1, len(chain)):
            parents[chain[j]] = chain[j - 1]
    return parents


def t2m_edges() -> List[Tuple[int, int]]:
    """Return parent-child edges for the 22-joint T2M skeleton."""
    from smplx_toolbox.utils.humanml_mapping import T2M_KINEMATIC_CHAIN

    edges: List[Tuple[int, int]] = []
    for chain in T2M_KINEMATIC_CHAIN:
        for i in range(1, len(chain)):
            edges.append((chain[i - 1], chain[i]))
    return edges


def fk_world_rotations_from_locals(
    R_locals: np.ndarray, parents: Sequence[int], R_root: np.ndarray
) -> np.ndarray:
    """Compute world rotations given local rotations and parent indices.

    Parameters
    ----------
    R_locals : (J, 3, 3)
        Local rotation per joint (pelvis included; pelvis local usually identity).
    parents : list[int]
        Parent index per joint (pelvis parent = -1).
    R_root : (3,3)
        Global orientation at the pelvis.

    Returns
    -------
    np.ndarray
        World rotations per joint (J, 3, 3).
    """
    J = int(R_locals.shape[0])
    Rw = np.zeros((J, 3, 3), dtype=np.float32)
    for j in range(J):
        p = parents[j]
        if p == -1:
            Rw[j] = R_root @ R_locals[j]
        else:
            Rw[j] = Rw[p] @ R_locals[j]
    return Rw


# ---------------------------------------
# T2M random pose generator and evaluators
# ---------------------------------------


def random_t2m_skeleton(
    *,
    angle_deg: float = 15.0,
    root_angle_deg: float = 10.0,
    seed: Optional[int] = None,
) -> T2MSkeleton:
    """Create a T2M skeleton with small random local joint rotations.

    Parameters
    ----------
    angle_deg : float
        Max absolute per-joint angle in degrees for local randomization.
    root_angle_deg : float
        Max absolute angle in degrees for root orientation randomization.
    seed : int, optional
        Random seed for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)

    skel = create_neutral_t2m_skeleton()
    J = 22

    # Random local rotations per joint
    max_rad = math.radians(float(angle_deg))
    R_locals = []
    for j in range(J):
        # Smaller noise on distal joints for stability
        scale = 1.0 if j < 16 else 0.7
        aa = (np.random.uniform(-max_rad, max_rad, size=(3,)).astype(np.float32)) * scale
        R = aa_to_mat_np(aa)
        R_locals.append(R)
    R_locals = np.stack(R_locals, axis=0)

    # Encode to 6D
    pose6d = np.stack([mat_to_6d_np(R) for R in R_locals], axis=0)

    # Random root orientation 6D
    max_root = math.radians(float(root_angle_deg))
    aa_root = np.random.uniform(-max_root, max_root, size=(3,)).astype(np.float32)
    R_root = aa_to_mat_np(aa_root)
    root6d = mat_to_6d_np(R_root)

    skel.pose6d = pose6d.astype(np.float32)
    skel.root_orient6d = root6d.astype(np.float32)
    # Keep translation at origin for clarity
    skel.trans = np.zeros((3,), dtype=np.float32)
    return skel


def t2m_world_rotations_and_positions(skel: T2MSkeleton) -> Tuple[np.ndarray, np.ndarray]:
    """Get T2M world rotations (J,3,3) and positions (J,3)."""
    from smplx_toolbox.utils.humanml_mapping import _cont6d_to_matrix_np, _fk_t2m_cont6d_np

    J = 22
    parents = parents_from_t2m()
    R_locals = _cont6d_to_matrix_np(skel.pose6d).reshape(J, 3, 3)
    R_root = _cont6d_to_matrix_np(skel.root_orient6d.reshape(1, 6)).reshape(3, 3)
    R_world = fk_world_rotations_from_locals(R_locals, parents, R_root)

    joints = _fk_t2m_cont6d_np(
        skel.pose6d[np.newaxis, ...], skel.trans[np.newaxis, ...], use_root_rotation=True
    )[0]
    return R_world, joints


# --------------------------------
# SMPL-X helpers (model + kintree)
# --------------------------------


def create_unified_smplx(body_model_dir: str, gender: str, device: torch.device) -> UnifiedSmplModel:
    base = smplx.create(
        model_path=body_model_dir,
        model_type="smplx",
        gender=gender,
        batch_size=1,
        use_pca=False,
        flat_hand_mean=True,
    )
    base = base.to(device)
    model = UnifiedSmplModel.from_smpl_model(base)
    return model


@dataclass
class SmplxKinematics:
    parents: np.ndarray  # (J,)
    J: int


def get_smplx_kinematics(model: UnifiedSmplModel, n_core: int = 22) -> SmplxKinematics:
    base = model.m_deformable_model  # type: ignore[attr-defined]
    if base is None:  # pragma: no cover - defensive
        raise RuntimeError("UnifiedSmplModel has no base model bound")
    parents = None
    if hasattr(base, "parents"):
        pr = getattr(base, "parents")
        if torch.is_tensor(pr):
            parents = pr.detach().cpu().numpy().astype(np.int64)
        else:
            parents = np.asarray(pr, dtype=np.int64)
    elif hasattr(base, "kintree_table"):
        kt = getattr(base, "kintree_table")
        parents = np.asarray(kt[0], dtype=np.int64)
    else:  # pragma: no cover - defensive
        raise RuntimeError("Unable to find SMPL-X parents")
    return SmplxKinematics(parents=parents[:n_core], J=int(n_core))


def smplx_world_rotations_from_named_pose(
    model: UnifiedSmplModel, npz: NamedPose, *, global_orient: torch.Tensor
) -> np.ndarray:
    """Compute SMPL-X world rotations (22x3x3) from NamedPose and root.

    This uses only the 22 core body joints (pelvis..wrists).
    """
    kin = get_smplx_kinematics(model, n_core=22)

    # Collect local rotations in axis-angle: pelvis (root_pose), then 21 body joints
    # Build per-joint rotation matrices
    # Names must follow CoreBodyJoint order
    names = [e.value for e in CoreBodyJoint]
    # Pelvis AA -> rotation
    root_aa_t = global_orient
    if not torch.is_tensor(root_aa_t):  # pragma: no cover - defensive
        root_aa_t = torch.zeros((1, 3))
    root_aa_np = root_aa_t.view(3).detach().cpu().numpy().astype(np.float32)
    R_root_np = aa_to_mat_np(root_aa_np)

    # Local rotations for joints 0..21
    R_locals = np.zeros((kin.J, 3, 3), dtype=np.float32)
    R_locals[0] = np.eye(3, dtype=np.float32)
    # Fill from intrinsic_pose for 21 joints
    for j, name in enumerate(names[1:], start=1):
        idx = npz.get_joint_index(name)
        if idx is None or npz.intrinsic_pose is None:
            R_locals[j] = np.eye(3, dtype=np.float32)
        else:
            aa = npz.intrinsic_pose[0, idx].detach().cpu().numpy().astype(np.float32)
            R_locals[j] = aa_to_mat_np(aa)

    # FK world
    Rw = fk_world_rotations_from_locals(R_locals, kin.parents.tolist(), R_root_np)
    return Rw


# ------------------------------
# Mapping and evaluation helpers
# ------------------------------


def build_t2m_to_smplx_core_mapping() -> List[Tuple[str, int]]:
    """Return list of (smplx_core_name, t2m_index) including pelvis.

    Collars in T2M are missing; they will map to -1 (handled by skip/identity).
    """
    order, idxs = humanml_joint_mapping()  # excludes pelvis
    mapping: List[Tuple[str, int]] = [(CoreBodyJoint.PELVIS.value, 0)]
    for j, k in zip(order, idxs):
        mapping.append((j.value, -1 if k is None else int(k)))
    return mapping


def mpjpe(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Mean per-joint position error (Euclidean, meters)."""
    if mask is not None:
        a = a[mask]
        b = b[mask]
    d = np.linalg.norm(a - b, axis=-1)
    return float(d.mean())


# -------------------------
# Visualization (PyVista)
# -------------------------


def _pv_or_none():
    try:
        import pyvista as pv  # type: ignore

        return pv
    except Exception:
        return None


def _pairs_from_edges(points: np.ndarray, edges: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    a = np.array([points[i] for i, _ in edges], dtype=float)
    b = np.array([points[j] for _, j in edges], dtype=float)
    return a, b


def visualize_comparison(
    *,
    model: UnifiedSmplModel,
    smpl_joints: np.ndarray,  # (55,3) or (22,3) if preselected
    t2m_joints: np.ndarray,  # (22,3)
    mapping_mask: np.ndarray,  # (22,) True for mapped joints
    title: str,
    out_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Visualize T2M and SMPL-X skeletons and their joint correspondences.

    - T2M skeleton: blue
    - SMPL-X skeleton: green
    - Correspondence lines (mapped joints): red
    """
    pv = _pv_or_none()
    if pv is None:
        print("[warn] pyvista not available; skipping visualization")
        return

    from smplx_toolbox.visualization.utils import add_connection_lines
    from smplx_toolbox.visualization.utils import get_smplx_bone_connections

    # Prepare plotter (use off-screen screenshot if out_path provided)
    try:
        pl = pv.Plotter(off_screen=bool(out_path is not None))
    except TypeError:
        pl = pv.Plotter()

    # Edges
    t_edges = t2m_edges()
    smpl_edges = [(i, j) for (i, j) in get_smplx_bone_connections() if i < 22 and j < 22]

    # Ensure smpl_joints shape is (22,3) for overlay
    if smpl_joints.shape[0] > 22:
        smpl_core = smpl_joints[:22]
    else:
        smpl_core = smpl_joints

    # Draw skeletons as edge lines
    a, b = _pairs_from_edges(t2m_joints, t_edges)
    add_connection_lines(pl, a, b, color=(0.2, 0.6, 0.9), line_width=4)

    sa, sb = _pairs_from_edges(smpl_core, smpl_edges)
    add_connection_lines(pl, sa, sb, color=(0.2, 0.9, 0.2), line_width=4)

    # Add points for clarity
    pl.add_points(t2m_joints, color=(1.0, 0.9, 0.1), render_points_as_spheres=True, point_size=12)
    pl.add_points(smpl_core, color=(0.9, 0.2, 0.9), render_points_as_spheres=True, point_size=10)

    # Add correspondence lines for mapped joints
    idxs = np.arange(22)[mapping_mask]
    add_connection_lines(pl, t2m_joints[idxs], smpl_core[idxs], color=(1.0, 0.1, 0.1), line_width=2, opacity=0.6)

    pl.add_text(title, font_size=10)

    if out_path is not None:
        try:
            pl.show(screenshot=out_path, auto_close=True)
            print(f"[viz] saved screenshot: {out_path}")
        except Exception as e:  # pragma: no cover - runtime viz guard
            print(f"[warn] failed to save screenshot to {out_path}: {e}")
            if show:
                pl.show()
    elif show:
        pl.show()


# ---------------
# Main procedure
# ---------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Transfer motion from T2M to SMPL-X using world rotation deltas")
    ap.add_argument("--body-model-dir", type=str, required=True, help="Path to SMPL(-X) body models (data/body_models)")
    ap.add_argument("--gender", type=str, default="neutral", choices=["neutral", "male", "female"], help="SMPL-X gender")
    ap.add_argument("--iters", type=int, default=200, help="Total inner optimization iterations for initial fit")
    ap.add_argument("--lr", type=float, default=0.05, help="Learning rate for Adam optimizer")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Compute device")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--verbose", action="store_true", help="Print progress details")
    ap.add_argument("--viz", type=str, default="save", choices=["none", "save", "show"], help="Visualization mode: none|save|show")
    ap.add_argument("--out-dir", type=str, default="tmp", help="Output directory for logs/screenshots")
    args = ap.parse_args(argv)

    # Device
    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1) Generate two random T2M skeletons (A: calibration, B: motion)
    t2m_A = random_t2m_skeleton(angle_deg=5.0, root_angle_deg=0.0, seed=args.seed)
    t2m_B = random_t2m_skeleton(angle_deg=15.0, root_angle_deg=10.0, seed=args.seed + 1)

    # World rotations and positions in Y-up (shared)
    Rw_A, Pw_A = t2m_world_rotations_and_positions(t2m_A)
    Rw_B, Pw_B = t2m_world_rotations_and_positions(t2m_B)

    # 2) Create neutral SMPL-X unified model
    model = create_unified_smplx(args.body_model_dir, args.gender, dev)

    # 3) Establish joint mapping (SMPL-X core name -> T2M index)
    mapping = build_t2m_to_smplx_core_mapping()

    # Build T2M->name target dictionary for A (positions)
    tgt_A: Dict[str, torch.Tensor] = {}
    for name, idx in mapping:
        if idx < 0:
            continue  # skip missing collar joints
        pos = torch.from_numpy(Pw_A[idx : idx + 1]).to(device=model.device, dtype=model.dtype)
        tgt_A[name] = pos  # (1,3)

    # 4) Fit SMPL-X to T2M A (allow shape to deform)
    # Initial parameters
    B = 1
    npz = NamedPose(model_type=ModelType.SMPLX, batch_size=B)
    # Make trainable
    if npz.intrinsic_pose is None:
        raise RuntimeError("NamedPose failed to allocate intrinsic_pose")
    npz.intrinsic_pose = torch.nn.Parameter(npz.intrinsic_pose.detach().to(device=model.device, dtype=model.dtype), requires_grad=True)
    root_init = torch.zeros((B, 3), device=model.device, dtype=model.dtype)
    npz.root_pose = root_init  # keep as plain tensor; helper will gate by DOF

    # Trainable global orient/translation/shape
    global_orient = torch.nn.Parameter(root_init.clone(), requires_grad=True)
    trans = torch.nn.Parameter(torch.zeros((B, 3), device=model.device, dtype=model.dtype), requires_grad=True)
    betas = torch.nn.Parameter(torch.zeros((B, model.num_betas), device=model.device, dtype=model.dtype), requires_grad=True)

    inputs0 = UnifiedSmplInputs(
        named_pose=npz,
        global_orient=global_orient,
        trans=trans,
        betas=betas,
    )

    helper = SmplKeypointFittingHelper.from_model(model)
    helper.set_keypoint_targets(tgt_A, weights=1.0, robust="gmof", rho=50.0)
    helper.set_dof_global_orient(True)
    helper.set_dof_global_translation(True)
    helper.set_dof_shape_deform(True)
    helper.set_reg_pose_l2(1e-3)
    helper.set_reg_shape_l2(1e-2)

    # Run optimization in chunks
    per_step = 25
    steps = max(1, int(math.ceil(args.iters / per_step)))
    it = helper.init_fitting(inputs0, optimizer="adam", lr=args.lr, num_iter_per_step=per_step)
    last_status = None
    for s in range(steps):
        last_status = next(it)
        if args.verbose:
            print(f"[fit] step {s+1}/{steps} loss={last_status.loss_total:.6f}")

    assert last_status is not None
    fitted_inputs: UnifiedSmplInputs = last_status.params
    fitted_output = last_status.output

    # Cache SMPL-X bind world rotations for core 22
    Rw_smpl_bind = smplx_world_rotations_from_named_pose(
        model, fitted_inputs.named_pose, global_orient=fitted_inputs.global_orient  # type: ignore[arg-type]
    )  # (22,3,3)

    # 5) Transfer motion (world rotation delta)
    # Compute per-joint delta: δ^W_j = R_src^W_j(B) · (R_src^W_j(A))^{-1}
    delta_world = []
    for name, idx in mapping:
        if idx < 0:
            delta_world.append(np.eye(3, dtype=np.float32))
            continue
        Ra = Rw_A[idx]
        Rb = Rw_B[idx]
        delta_world.append(Rb @ Ra.T)
    delta_world = np.stack(delta_world, axis=0)  # (22,3,3)

    # Apply to SMPL-X bind world
    Rw_smpl_t = delta_world @ Rw_smpl_bind

    # Recover target local rotations in topo order; obtain parents
    kin = get_smplx_kinematics(model, n_core=22)
    # Parent world at t; compute by walking hierarchy
    Rw_parent_t = np.zeros_like(Rw_smpl_t)
    for j in range(kin.J):
        p = int(kin.parents[j])
        if p < 0:
            Rw_parent_t[j] = np.eye(3, dtype=np.float32)
        else:
            Rw_parent_t[j] = Rw_smpl_t[p]
    Rl_smpl_t = np.zeros_like(Rw_smpl_t)
    for j in range(kin.J):
        # R_local = R_parent^{-1} * R_world
        Rl_smpl_t[j] = Rw_parent_t[j].T @ Rw_smpl_t[j]

    # Build new NamedPose from local rotations (pelvis in global_orient)
    names = [e.value for e in CoreBodyJoint]
    npz_t = NamedPose(model_type=ModelType.SMPLX, batch_size=1)
    # Fill only the core body joints; keep hands/face at zeros
    for j, name in enumerate(names[1:], start=1):
        aa = mat_to_rotvec_torch(Rl_smpl_t[j]).view(1, 1, 3).to(device=model.device, dtype=model.dtype)
        npz_t.set_joint_pose_value(name, aa)
    root_aa_t = mat_to_rotvec_torch(Rw_smpl_t[0]).view(1, 3).to(device=model.device, dtype=model.dtype)
    npz_t.root_pose = root_aa_t

    # Compose final inputs for evaluation (reuse fitted shape and possibly translation)
    # For translation, align pelvis positions: set SMPL-X trans to T2M B pelvis
    trans_t = torch.from_numpy(Pw_B[0:1]).to(device=model.device, dtype=model.dtype)
    inputs_t = UnifiedSmplInputs(
        named_pose=npz_t,
        global_orient=root_aa_t,
        trans=trans_t,
        betas=fitted_inputs.betas.detach() if isinstance(fitted_inputs.betas, torch.Tensor) else None,
    )
    out_t = model(inputs_t)

    # 6) Compare joint locations (mapped subset) + Visualization
    # Extract predicted joints at mapped names
    pred_joints = model.select_joints(out_t.joints, names=[name for name, _ in mapping])
    pred_np = pred_joints[0].detach().cpu().numpy()

    # Build target array from T2M B positions in same order
    tgt_np_list = []
    mask = []
    for name, idx in mapping:
        if idx < 0:
            # Missing joints (collars) — exclude from MPJPE
            tgt_np_list.append(np.zeros((3,), dtype=np.float32))
            mask.append(False)
        else:
            tgt_np_list.append(Pw_B[idx])
            mask.append(True)
    tgt_np = np.stack(tgt_np_list, axis=0)
    mask_np = np.array(mask, dtype=bool)

    err = mpjpe(pred_np[mask_np], tgt_np[mask_np])

    print("=== Motion Transfer: T2M → SMPL-X (world-rotation delta) ===")
    print(f"Device: {dev.type}; Gender: {args.gender}; Fitting iters: {args.iters}")
    print(f"Mapped joints (incl. pelvis): {sum(mask)} used / 22 total")
    print(f"MPJPE (meters): {err:.6f}")

    # Build SMPL-X joints for the fitted stage (A)
    smpl_A = model.select_joints(fitted_output.joints, names=[name for name, _ in mapping])[0].detach().cpu().numpy()

    # Visualization
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    if args.viz in ("save", "show"):
        # Stage A: after fitting (alignment)
        visualize_comparison(
            model=model,
            smpl_joints=smpl_A,
            t2m_joints=Pw_A,
            mapping_mask=mask_np,
            title="Stage A: T2M A vs SMPL-X (after fitting)",
            out_path=(os.path.join(args.out_dir, "viz_stage_A.png") if args.viz == "save" else None),
            show=(args.viz == "show"),
        )
        # Stage B: after motion transfer
        visualize_comparison(
            model=model,
            smpl_joints=pred_np,
            t2m_joints=Pw_B,
            mapping_mask=mask_np,
            title=f"Stage B: T2M B vs SMPL-X (after transfer) MPJPE={err:.3f} m",
            out_path=(os.path.join(args.out_dir, "viz_stage_B.png") if args.viz == "save" else None),
            show=(args.viz == "show"),
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - script entry
    raise SystemExit(main())
