"""Convert FlowMDM HumanML3D extended outputs to a pickled list of UnifiedSmplInputs.

This utility reads FlowMDM's extended HumanML3D artifacts (results_ext.npy)
and constructs a list of `UnifiedSmplInputs` objects, one per frame, populated with:
 - named_pose: 21 body joints (pelvis excluded) in axis‑angle, ModelType.SMPLX by default
 - global_orient: pelvis axis‑angle (B, 3)
 - transl: global translation (B, 3)

Notes
-----
- The script expects the extended outputs produced by our modified FlowMDM generator
  (generate-ex.py) when `dataset=humanml`.
- Primary source is `results_ext.npy` (required). A minimal fallback from
  `results.npy` is provided but yields zero body pose and zero global_orient.

Examples
--------
Run from workspace root using the dev environment:

    pixi run -e dev python scripts/cvt_flowmdm_humanml_to_smpl_animation.py \
        --input_dir tmp/flowmdm-out/humanml3d \
        --output_path tmp/flowmdm-out/humanml3d/unified_smpl_animation.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
import torch

from smplx_toolbox.core.constants import CoreBodyJoint, ModelType  # type: ignore[import-untyped]
from smplx_toolbox.core.containers import NamedPose, UnifiedSmplInputs  # type: ignore[import-untyped]
from smplx_toolbox.vposer.model import _matrot_to_axis_angle  # type: ignore[import-untyped]
from smplx_toolbox.utils.humanml_mapping import (
    retarget_t2m_cont6d_to_named_pose,
)


@dataclass
class HumanMLSample:
    feats_denorm: np.ndarray       # (T, 263)
    r_quat: np.ndarray             # (T, 4) in (w, x, y, z)
    r_pos: np.ndarray              # (T, 3)
    cont6d: np.ndarray             # (T, 22, 6) T2M order including root
    joints_world: np.ndarray       # (T, 22, 3)
    length: int
    text: str
    fps: int


def _quat_wxyz_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) per frame to axis‑angle (T,3).

    Uses SciPy if available, otherwise a numerically stable fallback.
    """
    T = int(q.shape[0])
    try:
        from typing import cast
        from scipy.spatial.transform import Rotation as R  # type: ignore[import-untyped]

        # SciPy expects (x, y, z, w)
        xyzw = np.stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]], axis=1)
        aa = cast(np.ndarray, R.from_quat(xyzw).as_rotvec()).astype(np.float32)
        return aa.reshape(T, 3)
    except Exception:
        # Robust fallback
        aa = np.zeros((T, 3), dtype=np.float32)
        for i in range(T):
            w, x, y, z = q[i]
            # Normalize to avoid drift
            n = np.sqrt(max(1e-8, w * w + x * x + y * y + z * z))
            w, x, y, z = w / n, x / n, y / n, z / n
            angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
            s = np.sqrt(max(1e-8, 1.0 - w * w))
            axis = np.array([x / s, y / s, z / s], dtype=np.float32)
            aa[i] = (axis * angle).astype(np.float32)
        return aa


def _aa_to_matrix_batch(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (T,3) to rotation matrices (T,3,3)."""
    T = int(aa.shape[0])
    try:
        from scipy.spatial.transform import Rotation as R  # type: ignore[import-untyped]
        mats = R.from_rotvec(aa.reshape(-1, 3)).as_matrix().astype(np.float32)
        return mats.reshape(T, 3, 3)
    except Exception:
        mats = []
        for v in aa:
            theta = float(np.linalg.norm(v))
            if theta < 1e-8:
                mats.append(np.eye(3, dtype=np.float32))
                continue
            k = v / theta
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float32)
            Rm = np.eye(3, dtype=np.float32) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
            mats.append(Rm)
        return np.stack(mats, axis=0)


def _matrix_to_aa_batch(mats: np.ndarray) -> np.ndarray:
    """Rotation matrices (T,3,3) to axis-angle (T,3) using torch helper."""
    T = int(mats.shape[0])
    m = torch.from_numpy(mats.astype(np.float32)).contiguous().view(-1, 3, 3)
    aa = _matrot_to_axis_angle(m).view(T, 3)
    return aa.detach().cpu().numpy().astype(np.float32)


def _y_up_to_z_up_rmat() -> np.ndarray:
    """Return Rx(+90°) rotation matrix to convert Y-up data to Z-up."""
    ang = np.deg2rad(90.0)
    c, s = np.cos(ang), np.sin(ang)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)
    return Rx


def _build_named_pose_from_t2m_cont6d(cont6d: np.ndarray, model_type: ModelType = ModelType.SMPLX) -> List[NamedPose]:
    """Create NamedPose list from T2M 6D rotations using explicit mapping.

    - Input cont6d: (T, 22, 6) in T2M order (index 0 = Pelvis/root).
    - Output: NamedPose sequence in toolbox canonical order (CoreBodyJoint
      without Pelvis). Missing joints (e.g., collars) are set to identity.
    """
    q = torch.from_numpy(cont6d).contiguous().float()
    named = retarget_t2m_cont6d_to_named_pose(q, model_type=model_type)
    return named


def _to_unified_inputs(sample: HumanMLSample) -> List[UnifiedSmplInputs]:
    """Convert one extended HumanML3D sample into `UnifiedSmplInputs` list."""
    # Prefer the full sequence length present in cont6d
    T = int(sample.cont6d.shape[0])
    cont = sample.cont6d[:T]
    rpos = sample.r_pos[:T]
    rquat = sample.r_quat[:T]

    named_list = _build_named_pose_from_t2m_cont6d(cont, model_type=ModelType.SMPLX)
    if len(named_list) != T:
        raise RuntimeError(f"NamedPose length mismatch: {len(named_list)} vs {T}")

    global_orient = _quat_wxyz_to_axis_angle(rquat).reshape(T, 3)
    transl = rpos.reshape(T, 3).astype(np.float32)

    # Coordinate frame fix: HumanML3D recoveries are Y-up; SMPL-X expects Z-up.
    # Convert by applying Rx(+90°): R_zup = Rx * R_yup; t_zup = Rx * t_yup.
    Rx = _y_up_to_z_up_rmat()
    R_go = _aa_to_matrix_batch(global_orient)
    R_go_zup = (Rx @ R_go)  # broadcasts Rx (3,3) over (T,3,3)
    global_orient = _matrix_to_aa_batch(R_go_zup)
    transl = (Rx @ transl.T).T.astype(np.float32)

    unified: List[UnifiedSmplInputs] = []
    for t in range(T):
        u = UnifiedSmplInputs(
            named_pose=named_list[t],
            global_orient=torch.from_numpy(global_orient[t]).view(1, 3).to(torch.float32),
            trans=torch.from_numpy(transl[t]).view(1, 3).to(torch.float32),
        )
        unified.append(u)
    return unified


def _load_ext_bundle(input_dir: Path) -> Tuple[List[HumanMLSample], dict[str, Any]]:
    path = input_dir / "results_ext.npy"
    if not path.exists():
        raise FileNotFoundError(f"Extended bundle not found: {path}")
    arr = np.load(path, allow_pickle=True).item()
    samples_raw = arr.get("samples", [])
    meta = arr.get("meta", {})
    samples: List[HumanMLSample] = []
    for s in samples_raw:
        samp = HumanMLSample(
            feats_denorm=np.asarray(s["feats_denorm"], dtype=np.float32),
            r_quat=np.asarray(s["r_quat"], dtype=np.float32),
            r_pos=np.asarray(s["r_pos"], dtype=np.float32),
            cont6d=np.asarray(s["cont6d"], dtype=np.float32),
            joints_world=np.asarray(s["joints_world"], dtype=np.float32),
            length=int(s.get("length", s["cont6d"].shape[0])),
            text=str(s.get("text", "")),
            fps=int(s.get("fps", meta.get("fps", 20))),
        )
        samples.append(samp)
    return samples, meta


def _fallback_from_results_npy(input_dir: Path) -> List[UnifiedSmplInputs]:
    """Minimal fallback path: build empty poses and zero global_orient.

    This uses only `results.npy` joints to set translations from pelvis positions.
    """
    path = input_dir / "results.npy"
    if not path.exists():
        raise FileNotFoundError(f"results.npy not found at {path}")
    D = np.load(path, allow_pickle=True).item()
    motion = D["motion"]  # (B,22,3,T)
    jw = motion[0].transpose(2, 0, 1)  # (T,22,3)
    T = jw.shape[0]
    transl = jw[:, 0, :]  # pelvis
    named_list: List[NamedPose] = []
    for t in range(T):
        named_list.append(NamedPose(model_type=ModelType.SMPLX, batch_size=1))
    unified: List[UnifiedSmplInputs] = []
    for t in range(T):
        u = UnifiedSmplInputs(
            named_pose=named_list[t],
            global_orient=torch.zeros(1, 3, dtype=torch.float32),
            trans=torch.from_numpy(transl[t]).view(1, 3).to(torch.float32),
        )
        unified.append(u)
    return unified


def convert(input_dir: Path, output_path: Path, sample_index: int = 0) -> Path:
    input_dir = input_dir.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        samples, meta = _load_ext_bundle(input_dir)
        if not samples:
            raise RuntimeError("Empty extended bundle")
        idx = max(0, min(sample_index, len(samples) - 1))
        unified = _to_unified_inputs(samples[idx])
    except Exception as e:
        # Fallback
        print(f"[warn] Extended bundle missing or invalid ({e}); falling back to results.npy")
        unified = _fallback_from_results_npy(input_dir)

    with open(output_path, "wb") as f:
        pickle.dump(unified, f)

    return output_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert FlowMDM HumanML3D outputs to UnifiedSmplInputs animation.")
    ap.add_argument("--input_dir", type=str, default="tmp/flowmdm-out/humanml3d", help="Directory containing FlowMDM HumanML3D outputs")
    ap.add_argument("--output_path", type=str, default=None, help="Output pickle path (defaults to <input_dir>/unified_smpl_animation.pkl)")
    ap.add_argument("--sample_index", type=int, default=0, help="Index of sample in extended bundle to convert")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path) if args.output_path else (input_dir / "unified_smpl_animation.pkl")

    out = convert(input_dir, output_path, sample_index=int(args.sample_index))
    print(f"[ok] Wrote UnifiedSmplInputs list ({out})")


if __name__ == "__main__":
    main()
