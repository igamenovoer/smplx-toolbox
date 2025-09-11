"""Manual smoke script — 3D keypoint fitting with VPoser prior (SMPL-X).

This script mirrors the basic keypoint-fitting example but adds a VPoser
pose prior to regularize the 21-joint body pose. It:

- Uses only the SMPL-X model (with hands/face available but not optimized).
- Selects a set of symmetric movable keypoints: wrists, feet, elbows, hips.
- Generates 3D target keypoints by perturbing the neutral model output.
- Optimizes model pose to match targets + VPoser prior + L2 pose regularization.
- Visualizes original vs target vs fitted keypoints and meshes, with connecting lines.

Notes
-----
- This is a manual smoke script, not collected by pytest.
- No `if __name__ == '__main__'` guard (per project guidance).
- Uses GPU (CUDA or MPS) if available; otherwise falls back to CPU.
- For VPoser checkpoint, see `data/vposer/vposer-v2.ckpt`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import pyvista as pv
import smplx

from smplx_toolbox.core.constants import CoreBodyJoint
from smplx_toolbox.core.unified_model import UnifiedSmplInputs, UnifiedSmplModel
from smplx_toolbox.optimization import (
    KeypointMatchLossBuilder,
    VPoserPriorLossBuilder,
)
from smplx_toolbox.vposer import load_vposer
from smplx_toolbox.visualization import SMPLVisualizer, add_connection_lines
from smplx_toolbox.utils import select_device


class Params:
    """Adjustable parameters for the smoke test (easy to tweak)."""

    # Data/optimization
    NOISE_SCALE: float = 0.3  # std of Gaussian noise for targets (meters)
    STEPS: int = 250  # optimizer iterations
    LR: float = 0.05  # Adam learning rate
    L2_WEIGHT: float = 5e-3  # global L2 regularization weight on pose params
    ROBUST_KIND: str = "gmof"  # "gmof" or "l2" (simple MSE); also supports "huber"
    ROBUST_RHO: float = 100.0  # scale parameter for robustifier (used for gmof/huber)

    # VPoser prior weights
    VPOSER_POSE_FIT: float = 0.1  # weight for self-reconstruction MSE
    VPOSER_LATENT_L2: float = 0.05  # weight for latent magnitude

    # Paths
    MODEL_ROOT: Path = Path("data/body_models")
    VPOSER_CKPT: Path = Path("data/vposer/vposer-v2.ckpt")

    # Visualization
    NEUTRAL_COLOR: tuple[float, float, float] = (0.5, 0.5, 0.5)
    FITTED_COLOR: tuple[float, float, float] = (0.0, 0.3, 1.0)
    TARGET_COLOR: tuple[float, float, float] = (1.0, 0.0, 0.0)
    POINT_SIZE_NEUTRAL: float = 0.01
    POINT_SIZE_FITTED: float = 0.012
    TARGET_POINT_SIZE_PX: int = 12
    LINE_WIDTH: int = 2
    LINE_OPACITY: float = 0.9


def _select_device() -> torch.device:
    """Project-wide device selection (OS-aware)."""
    return select_device()


def _movable_joint_names() -> list[str]:
    """Wrists, feet, elbows, hips (left and right)."""
    return [
        CoreBodyJoint.LEFT_WRIST.value,
        CoreBodyJoint.RIGHT_WRIST.value,
        CoreBodyJoint.LEFT_FOOT.value,
        CoreBodyJoint.RIGHT_FOOT.value,
        CoreBodyJoint.LEFT_ELBOW.value,
        CoreBodyJoint.RIGHT_ELBOW.value,
        CoreBodyJoint.LEFT_HIP.value,
        CoreBodyJoint.RIGHT_HIP.value,
    ]


def _select_targets(
    model: UnifiedSmplModel,
    output_neutral: Any,
    names: Iterable[str],
    noise_scale: float = Params.NOISE_SCALE,
) -> dict[str, torch.Tensor]:
    """Build a name->(B,3) target dict from neutral joints + Gaussian noise."""
    name_list = list(names)
    # Select in unified space by names; detach from graph
    with torch.no_grad():
        sel = model.select_joints(output_neutral.joints, names=name_list)  # (B,n,3)
        noise = torch.randn_like(sel) * float(noise_scale)
        tgt = sel + noise
    return {nm: tgt[:, i] for i, nm in enumerate(name_list)}


def _optimize_pose_to_targets(
    model: UnifiedSmplModel,
    vposer_ckpt: Path,
    names: list[str],
    targets: dict[str, torch.Tensor],
    steps: int = Params.STEPS,
    lr: float = Params.LR,
    l2_weight: float = Params.L2_WEIGHT,
    w_pose_fit: float = Params.VPOSER_POSE_FIT,
    w_latent_l2: float = Params.VPOSER_LATENT_L2,
) -> tuple[Any, dict[str, float]]:
    """Fit pose with data term + VPoser prior + L2 pose regularization."""
    device = model.device
    dtype = model.dtype

    # Trainable pose parameters (start at identity/zeros)
    root_orient = torch.nn.Parameter(torch.zeros((1, 3), device=device, dtype=dtype))
    pose_body = torch.nn.Parameter(torch.zeros((1, 63), device=device, dtype=dtype))

    params = [root_orient, pose_body]
    opt = torch.optim.Adam(params, lr=lr)

    # Loss: data
    km = KeypointMatchLossBuilder.from_model(model)
    term_data = km.by_target_positions(
        targets,
        robust=Params.ROBUST_KIND,
        rho=Params.ROBUST_RHO,
        reduction="mean",
    )

    # VPoser prior (eval mode; loads on device)
    if not vposer_ckpt.exists():
        print(f"[smoke] VPoser ckpt not found at {vposer_ckpt} — skipping VPoser prior.")
        term_vposer = None
    else:
        vposer = load_vposer(str(vposer_ckpt), map_location=device)
        vposer.to(device=device)
        vposer.eval()
        vp_builder = VPoserPriorLossBuilder.from_vposer(model, vposer)
        term_vposer = vp_builder.by_pose(pose_body, w_pose_fit, w_latent_l2)

    # Initial eval
    out0 = model.forward(UnifiedSmplInputs(root_orient=root_orient.detach(), pose_body=pose_body.detach()))
    with torch.no_grad():
        loss0 = float(term_data(out0).item())

    # Optimize
    for i in range(steps):
        opt.zero_grad()
        out = model.forward(UnifiedSmplInputs(root_orient=root_orient, pose_body=pose_body))
        loss = term_data(out)
        # Add VPoser term if available
        if term_vposer is not None:
            loss = loss + term_vposer(out)
        # L2 reg on root_orient and pose_body
        reg = (root_orient**2).sum() + (pose_body**2).sum()
        loss = loss + float(l2_weight) * reg
        if i % 10 == 0 or i == steps - 1:
            try:
                loss_val = float(loss.detach().item())
            except Exception:
                loss_val = float(loss.item())
            print(f"[smoke] iter {i:03d} | loss = {loss_val:.6f}")
        loss.backward()
        opt.step()

    out_final = model.forward(
        UnifiedSmplInputs(root_orient=root_orient.detach(), pose_body=pose_body.detach())
    )
    with torch.no_grad():
        loss_final = float(term_data(out_final).item())

    diag = {"loss_init": loss0, "loss_final": loss_final}
    return out_final, diag


print("[smoke] 3D keypoint fitting with VPoser prior (SMPL-X)")

device = _select_device()
print(f"[smoke] Using device: {device}")

model_root = Params.MODEL_ROOT
if not model_root.exists():
    print("[smoke] data/body_models not found — skipping.")
    raise SystemExit(0)

# Build SMPL-X base model and move to device
gender = "neutral"
try:
    base = smplx.create(
        str(model_root), model_type="smplx", gender=gender, use_pca=False, batch_size=1, ext="pkl"
    )
except AssertionError as e:
    print(f"[smoke] Missing SMPL-X resources ({e}); skipping.")
    raise SystemExit(0)
base = base.to(device)

uni: UnifiedSmplModel = UnifiedSmplModel.from_smpl_model(base)

# Neutral forward
neutral_out = uni.forward(UnifiedSmplInputs())

# Target keypoints
subset_names = _movable_joint_names()
torch.manual_seed(42)
targets = _select_targets(uni, neutral_out, subset_names, noise_scale=Params.NOISE_SCALE)

# Optimize
print(
    f"[smoke] Optimizing SMPL-X with VPoser | steps={Params.STEPS} lr={Params.LR} noise={Params.NOISE_SCALE} l2={Params.L2_WEIGHT}"
)
print(f"[smoke] VPoser ckpt: {Params.VPOSER_CKPT}")
fitted_out, diag = _optimize_pose_to_targets(
    uni,
    Params.VPOSER_CKPT,
    subset_names,
    targets,
    steps=Params.STEPS,
    lr=Params.LR,
    l2_weight=Params.L2_WEIGHT,
    w_pose_fit=Params.VPOSER_POSE_FIT,
    w_latent_l2=Params.VPOSER_LATENT_L2,
)
print(f"[smoke] smplx loss: init={diag['loss_init']:.6f} -> final={diag['loss_final']:.6f}")

# Visualization
viz: SMPLVisualizer = SMPLVisualizer.from_model(uni)
plotter = viz.get_plotter()

# Original neutral mesh and keypoints
neutral_color = Params.NEUTRAL_COLOR
viz.add_mesh(neutral_out, style="wireframe", color=neutral_color, opacity=1.0)
viz.add_smpl_joints(
    neutral_out,
    joints=subset_names,
    labels=False,
    color=neutral_color,
    size=Params.POINT_SIZE_NEUTRAL,
)

# Fitted mesh and keypoints
fitted_color = Params.FITTED_COLOR
viz.add_mesh(fitted_out, style="wireframe", color=fitted_color, opacity=1.0)
viz.add_smpl_joints(
    fitted_out,
    joints=subset_names,
    labels=False,
    color=fitted_color,
    size=Params.POINT_SIZE_FITTED,
)

# Target keypoints (red) and connecting lines
with torch.no_grad():
    tgt_stack = torch.stack([targets[nm][0] for nm in subset_names], dim=0)
tgt_np = tgt_stack.detach().cpu().numpy()
plotter.add_points(
    tgt_np,
    color=Params.TARGET_COLOR,
    render_points_as_spheres=True,
    point_size=Params.TARGET_POINT_SIZE_PX,
)

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

plotter.add_text("SMPL-X 3D keypoint fit + VPoser", font_size=12)
plotter.reset_camera_clipping_range()
plotter.show()
try:
    plotter.close()
except Exception:
    pass
try:
    pv.close_all()
except Exception:
    pass
del plotter, viz
try:
    import gc

    gc.collect()
except Exception:
    pass
