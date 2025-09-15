"""Manual smoke script — 3D keypoint fitting (SMPL-X, SMPL-H, SMPL).

This script performs a simple keypoint-based fitting experiment:
- Generate target 3D keypoints by taking the neutral model's joints for
  a small subset (left wrist + right foot) and adding Gaussian noise.
- Optimize pose parameters to match those targets using the
  KeypointMatchLossBuilder (3D data term).
- Add simple L2 regularization on pose parameters to discourage overfitting.
- Visualize original (neutral) vs target vs fitted keypoints and wireframe meshes.

Notes
-----
- This is a manual smoke script, not collected by pytest.
- No `if __name__ == '__main__'` guard (per project guidance).
- Uses CPU by default; small optimization steps keep it quick.
- Skips gracefully if `data/body_models` is missing.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4
from typing import Any, Iterable

import numpy as np
import torch
import pyvista as pv
import smplx

# Do not force a Jupyter backend; default plotter is fine in shells/CI

from smplx_toolbox.core.constants import CoreBodyJoint, ModelType
from smplx_toolbox.core import NamedPose
from smplx_toolbox.core.unified_model import UnifiedSmplInputs, UnifiedSmplModel
from smplx_toolbox.optimization import KeypointMatchLossBuilder
from smplx_toolbox.visualization import SMPLVisualizer, add_connection_lines


class Params:
    """Adjustable parameters for the smoke test (easy to tweak).

    Modify these values to change optimization/visualization behavior.
    """

    # Data/optimization
    NOISE_SCALE: float = 0.3  # std of Gaussian noise for targets (meters)
    STEPS: int = 100  # optimizer iterations
    LR: float = 0.05  # Adam learning rate
    L2_WEIGHT: float = 1e-2  # global L2 regularization weight on pose params
    SHAPE_L2_WEIGHT: float = 1e-2  # L2 regularization weight on betas (shape)
    ROBUST_KIND: str = "gmof"  # "gmof" or "l2" (simple MSE); also supports "huber"
    ROBUST_RHO: float = 100.0  # scale parameter for robustifier (used for gmof/huber)

    # Models to test
    MODEL_TYPES: tuple[str, ...] = ("smplx", "smplh", "smpl")

    # Visualization
    NEUTRAL_COLOR: tuple[float, float, float] = (0.5, 0.5, 0.5)
    FITTED_COLOR: tuple[float, float, float] = (0.0, 0.3, 1.0)
    TARGET_COLOR: tuple[float, float, float] = (1.0, 0.0, 0.0)
    POINT_SIZE_NEUTRAL: float = 0.01
    POINT_SIZE_FITTED: float = 0.012
    TARGET_POINT_SIZE_PX: int = 12
    LINE_WIDTH: int = 2
    LINE_OPACITY: float = 0.9

    # Trainable DOFs
    ENABLE_GLOBAL_ORIENT: bool = True
    ENABLE_GLOBAL_TRANSLATION: bool = True
    ENABLE_SHAPE_DEFORMATION: bool = True


def _left_wrist_name() -> str:
    """Return the unified joint name for the left wrist keypoint."""
    return CoreBodyJoint.LEFT_WRIST.value


def _select_targets(
    model: UnifiedSmplModel,
    output_neutral: Any,
    names: Iterable[str],
    noise_scale: float = Params.NOISE_SCALE,
) -> dict[str, torch.Tensor]:
    """Build a name->(B,3) target dict from neutral joints + Gaussian noise.

    Parameters
    ----------
    model : UnifiedSmplModel
        The unified model used to resolve joint names.
    output_neutral : UnifiedSmplOutput
        Output from a neutral forward pass (zeros pose/trans/betas).
    names : Iterable[str]
        Joint names to include as targets.
    noise_scale : float, optional
        Standard deviation for Gaussian noise in model units (meters).

    Returns
    -------
    dict[str, torch.Tensor]
        Mapping from joint name to tensor of shape (B, 3) on the model's device/dtype.
    """
    name_list = list(names)
    # Select in unified space by names
    # Detach to avoid building a graph from the neutral forward pass
    with torch.no_grad():
        sel = model.select_joints(output_neutral.joints, names=name_list)  # (B,n,3)
        # Add Gaussian noise (same device/dtype)
        noise = torch.randn_like(sel) * float(noise_scale)
        tgt = sel + noise
    # Split into dict
    result: dict[str, torch.Tensor] = {}
    for i, nm in enumerate(name_list):
        result[nm] = tgt[:, i]
    return result


def _optimize_pose_to_targets(
    model: UnifiedSmplModel,
    names: list[str],
    targets: dict[str, torch.Tensor],
    steps: int = Params.STEPS,
    lr: float = Params.LR,
    l2_weight: float = Params.L2_WEIGHT,
) -> tuple[Any, dict[str, float]]:
    """Fit pose parameters to match target joints using a 3D data term.

    Returns the final UnifiedSmplOutput and simple diagnostics.
    """
    device = model.device
    dtype = model.dtype

    # Trainable parameters: intrinsic pose + optional global orient and translation
    npz = NamedPose(model_type=ModelType(str(model.model_type)), batch_size=1)
    npz.intrinsic_pose = torch.nn.Parameter(torch.zeros_like(npz.intrinsic_pose))
    global_orient = (
        torch.nn.Parameter(torch.zeros((1, 3), device=device, dtype=dtype))
        if Params.ENABLE_GLOBAL_ORIENT
        else None
    )
    translation = (
        torch.nn.Parameter(torch.zeros((1, 3), device=device, dtype=dtype))
        if Params.ENABLE_GLOBAL_TRANSLATION
        else None
    )

    params = [npz.intrinsic_pose]
    if global_orient is not None:
        params.append(global_orient)
    if translation is not None:
        params.append(translation)
    betas = None
    if Params.ENABLE_SHAPE_DEFORMATION:
        try:
            n_betas = int(model.num_betas)
        except Exception:
            n_betas = 10
        betas = torch.nn.Parameter(torch.zeros((1, n_betas), device=device, dtype=dtype))
        params.append(betas)
    opt = torch.optim.Adam(params, lr=lr)

    # Loss builder
    km = KeypointMatchLossBuilder.from_model(model)
    term = km.by_target_positions(
        targets,
        robust=Params.ROBUST_KIND,
        rho=Params.ROBUST_RHO,
        reduction="mean",
    )

    # Initial eval
    out0 = model.forward(
        UnifiedSmplInputs(
            named_pose=npz,
            global_orient=(global_orient.detach() if global_orient is not None else None),
            trans=(translation.detach() if translation is not None else None),
            betas=(betas.detach() if betas is not None else None),
        )
    )
    with torch.no_grad():
        loss0 = float(term(out0).item())

    # Optimize
    for i in range(steps):
        opt.zero_grad()
        out = model.forward(
            UnifiedSmplInputs(
                named_pose=npz,
                global_orient=global_orient if global_orient is not None else None,
                trans=translation if translation is not None else None,
                betas=betas if betas is not None else None,
            )
        )
        # Data term
        loss = term(out)
        # L2 regularization on intrinsic pose (not on global_orient)
        reg: torch.Tensor = (npz.intrinsic_pose**2).sum()
        loss = loss + float(l2_weight) * reg
        # Optional L2 regularization on shape parameters (betas)
        if betas is not None and float(Params.SHAPE_L2_WEIGHT) > 0:
            loss = loss + float(Params.SHAPE_L2_WEIGHT) * (betas**2).sum()
        if i % 10 == 0 or i == steps - 1:
            try:
                loss_val = float(loss.detach().item())
            except Exception:
                loss_val = float(loss.item())
            print(f"[smoke] iter {i:03d} | loss = {loss_val:.6f}")
        loss.backward()
        opt.step()

    out_final = model.forward(
        UnifiedSmplInputs(
            named_pose=npz,
            global_orient=(global_orient.detach() if global_orient is not None else None),
            trans=(translation.detach() if translation is not None else None),
            betas=(betas.detach() if betas is not None else None),
        )
    )
    with torch.no_grad():
        loss_final = float(term(out_final).item())

    diag = {"loss_init": loss0, "loss_final": loss_final}
    return out_final, diag


print("[smoke] 3D keypoint fitting: left hand + right foot (SMPL-X, SMPL-H)")
print("[smoke] Testing models: smplx, smplh, smpl (no hand)")
print(
    f"[smoke] Params: steps={Params.STEPS}, lr={Params.LR}, noise={Params.NOISE_SCALE}, l2_weight={Params.L2_WEIGHT}"
)

model_path = Path("data/body_models")
if not model_path.exists():
    print("[smoke] data/body_models not found — skipping.")
    raise SystemExit(0)

# Common subset of joint names: left wrist + right foot
subset_names = [_left_wrist_name(), CoreBodyJoint.RIGHT_FOOT.value]

for model_type in Params.MODEL_TYPES:
    print(f"[smoke] Building base model: {model_type}")
    # Pick gender/extension compatible with available resources
    if model_type == "smplx":
        gender = "neutral"
    elif model_type == "smplh":
        gender = "male"
    else:  # smpl
        gender = "neutral"
    ext = "pkl"
    # Prefer PKL if present; otherwise try NPZ fallback for SMPL-H
    expected_pkl = (
        model_path
        / ("smplx" if model_type == "smplx" else "smplh")
        / f"{model_type.upper()}_{gender.upper()}.pkl"
    )
    expected_npz = (
        model_path / "smplh" / gender / "model.npz" if model_type == "smplh" else None
    )
    if not expected_pkl.exists() and model_type == "smplh" and expected_npz is not None and expected_npz.exists():
        ext = "npz"
    try:
        base = smplx.create(
            str(model_path),
            model_type=model_type,
            gender=gender,
            use_pca=False,
            batch_size=1,
            ext=ext,
        )
    except AssertionError as e:
        print(f"[smoke] Missing resources for {model_type} ({e}); skipping this model.")
        continue
    uni: UnifiedSmplModel = UnifiedSmplModel.from_smpl_model(base)

    # Forward neutral
    neutral_out = uni.forward(UnifiedSmplInputs())

    # Generate targets from neutral + Gaussian noise
    # Unique random seed per run to vary target noise
    _seed = uuid4().int & 0xFFFFFFFF
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    targets = _select_targets(uni, neutral_out, subset_names, noise_scale=Params.NOISE_SCALE)

    # Fit
    print(f"[smoke] Optimizing pose for {model_type} ...")
    fitted_out, diag = _optimize_pose_to_targets(
        uni,
        subset_names,
        targets,
        steps=Params.STEPS,
        lr=Params.LR,
        l2_weight=Params.L2_WEIGHT,
    )
    print(f"[smoke] {model_type} loss: init={diag['loss_init']:.6f} -> final={diag['loss_final']:.6f}")

    # Visualize original vs target vs fitted
    viz: SMPLVisualizer = SMPLVisualizer.from_model(uni)
    plotter = viz.get_plotter()

    # Original neutral mesh and keypoints (same color)
    neutral_color = Params.NEUTRAL_COLOR
    viz.add_mesh(neutral_out, style="wireframe", color=neutral_color, opacity=1.0)
    viz.add_smpl_joints(
        neutral_out, joints=subset_names, labels=False, color=neutral_color, size=Params.POINT_SIZE_NEUTRAL
    )

    # Fitted mesh and keypoints (blue mesh and same-colored kpts)
    fitted_color = Params.FITTED_COLOR
    viz.add_mesh(fitted_out, style="wireframe", color=fitted_color, opacity=1.0)
    viz.add_smpl_joints(
        fitted_out, joints=subset_names, labels=False, color=fitted_color, size=Params.POINT_SIZE_FITTED
    )

    # Target keypoints (red) as raw points
    # Stack to (n,3) numpy for plotting
    with torch.no_grad():
        tgt_stack = torch.stack([targets[nm][0] for nm in subset_names], dim=0)
    tgt_np = tgt_stack.detach().cpu().numpy()
    plotter.add_points(
        tgt_np,
        color=Params.TARGET_COLOR,
        render_points_as_spheres=True,
        point_size=Params.TARGET_POINT_SIZE_PX,
    )

    # Draw connection lines between predicted keypoints and targets
    with torch.no_grad():
        neutral_sel = uni.select_joints(neutral_out.joints, names=subset_names)[0]
        fitted_sel = uni.select_joints(fitted_out.joints, names=subset_names)[0]
    neutral_np = neutral_sel.detach().cpu().numpy()
    fitted_np = fitted_sel.detach().cpu().numpy()
    add_connection_lines(
        plotter, neutral_np, tgt_np, color=neutral_color, line_width=Params.LINE_WIDTH, opacity=Params.LINE_OPACITY
    )
    add_connection_lines(
        plotter, fitted_np, tgt_np, color=fitted_color, line_width=Params.LINE_WIDTH, opacity=Params.LINE_OPACITY
    )

    plotter.add_text(f"{model_type.upper()} 3D keypoint fit", font_size=12)
    plotter.reset_camera_clipping_range()
    plotter.show()
    # Proactively release VTK resources to avoid __del__ warnings at interpreter exit
    try:
        plotter.close()
    except Exception:
        pass
    try:
        pv.close_all()
    except Exception:
        pass
    # Drop Python references and hint GC
    del plotter
    del viz
    try:
        import gc

        gc.collect()
    except Exception:
        pass
