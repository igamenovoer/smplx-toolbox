"""Manual smoke script — 3D keypoint fitting via SmplKeypointFittingHelper.

Mimics the baseline smoke test but routes optimization through the helper.
Not collected by pytest; intended for ad‑hoc interactive runs.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4
from typing import Any, Iterable

import numpy as np
import torch
import smplx

from smplx_toolbox.core.constants import CoreBodyJoint, ModelType
from smplx_toolbox.core import NamedPose
from smplx_toolbox.core.unified_model import UnifiedSmplInputs, UnifiedSmplModel
from smplx_toolbox.fitting import SmplKeypointFittingHelper
from smplx_toolbox.utils import select_device
from smplx_toolbox.visualization import SMPLVisualizer, add_connection_lines
import pyvista as pv


class Params:
    # Std‑dev of Gaussian noise (meters) added to neutral joints to form targets
    NOISE_SCALE: float = 0.3
    # Number of outer steps to run; effective updates = STEPS * INNER_ITERS
    STEPS: int = 100
    # Inner optimizer iterations per yielded step (keep =1 for parity with baseline)
    INNER_ITERS: int = 1
    # Adam learning rate
    LR: float = 0.05
    # L2 weight on intrinsic body pose (excludes global orient); set 0.0 to disable
    L2_WEIGHT: float = 1e-2
    # L2 weight on shape coefficients (betas); set 0.0 to disable
    SHAPE_L2_WEIGHT: float = 1e-2
    # Robustifier kind for data term: "l2", "huber", or "gmof"
    ROBUST_KIND: str = "gmof"
    # Robustifier scale (larger = less downweighting of outliers)
    ROBUST_RHO: float = 100.0

    # Trainable DOFs (toggle whether these tensors are optimized)
    ENABLE_GLOBAL_ORIENT: bool = True
    ENABLE_GLOBAL_TRANSLATION: bool = True
    ENABLE_SHAPE_DEFORMATION: bool = True

    # Visualization styling
    NEUTRAL_COLOR: tuple[float, float, float] = (0.5, 0.5, 0.5)
    FITTED_COLOR: tuple[float, float, float] = (0.0, 0.3, 1.0)
    TARGET_COLOR: tuple[float, float, float] = (1.0, 0.0, 0.0)
    POINT_SIZE_NEUTRAL: float = 0.01
    POINT_SIZE_FITTED: float = 0.012
    TARGET_POINT_SIZE_PX: int = 12
    LINE_WIDTH: int = 2
    LINE_OPACITY: float = 0.9


def _select_targets(
    model: UnifiedSmplModel,
    output_neutral: Any,
    names: Iterable[str],
    noise_scale: float = Params.NOISE_SCALE,
) -> dict[str, torch.Tensor]:
    name_list = list(names)
    with torch.no_grad():
        sel = model.select_joints(output_neutral.joints, names=name_list)
        noise = torch.randn_like(sel) * float(noise_scale)
        tgt = sel + noise
    return {name_list[i]: tgt[:, i] for i in range(len(name_list))}


print("[smoke-helper] 3D keypoint fitting via helper (SMPL-X)")
device = select_device()
print(f"[smoke-helper] Using device: {device}")

model_root = Path("data/body_models")
if not model_root.exists():
    print("[smoke-helper] data/body_models not found — skipping.")
    raise SystemExit(0)

try:
    base = smplx.create(str(model_root), model_type="smplx", gender="neutral", use_pca=False, batch_size=1, ext="pkl")
except AssertionError as e:
    print(f"[smoke-helper] Missing SMPL-X resources ({e}); skipping.")
    raise SystemExit(0)
base = base.to(device)
model = UnifiedSmplModel.from_smpl_model(base)

# Neutral output to create targets
neutral = model.forward(UnifiedSmplInputs())
targets = _select_targets(
    model,
    neutral,
    [CoreBodyJoint.LEFT_WRIST.value, CoreBodyJoint.RIGHT_FOOT.value],
    noise_scale=Params.NOISE_SCALE,
)

# Seed for reproducibility
seed = uuid4().int & 0xFFFFFFFF
torch.manual_seed(seed)
np.random.seed(seed)

# Trainable inputs
npz = NamedPose(model_type=ModelType(str(model.model_type)), batch_size=1)
npz.intrinsic_pose = torch.nn.Parameter(torch.zeros_like(npz.intrinsic_pose))
global_orient = (
    torch.nn.Parameter(torch.zeros((1, 3), device=model.device, dtype=model.dtype))
    if Params.ENABLE_GLOBAL_ORIENT
    else None
)
translation = (
    torch.nn.Parameter(torch.zeros((1, 3), device=model.device, dtype=model.dtype))
    if Params.ENABLE_GLOBAL_TRANSLATION
    else None
)
betas = None
if Params.ENABLE_SHAPE_DEFORMATION:
    try:
        n_betas = int(model.num_betas)
    except Exception:
        n_betas = 10
    betas = torch.nn.Parameter(torch.zeros((1, n_betas), device=model.device, dtype=model.dtype))

init = UnifiedSmplInputs(
    named_pose=npz,
    global_orient=global_orient,
    trans=translation,
    betas=betas,
)

# Helper
helper = SmplKeypointFittingHelper.from_model(model)
helper.set_keypoint_targets(targets, robust=Params.ROBUST_KIND, rho=Params.ROBUST_RHO)
helper.set_dof_global_orient(Params.ENABLE_GLOBAL_ORIENT)
helper.set_dof_global_translation(Params.ENABLE_GLOBAL_TRANSLATION)
helper.set_dof_shape_deform(Params.ENABLE_SHAPE_DEFORMATION)
helper.set_reg_pose_l2(Params.L2_WEIGHT)
helper.set_reg_shape_l2(Params.SHAPE_L2_WEIGHT)

it = helper.init_fitting(
    init, optimizer="adam", lr=Params.LR, num_iter_per_step=Params.INNER_ITERS
)
for i, status in zip(range(Params.STEPS), it):
    if i % 10 == 0 or i == Params.STEPS - 1:
        print(f"[smoke-helper] iter {status.step:03d} | loss = {status.loss_total:.6f}")

# Visualize neutral vs target vs fitted
viz: SMPLVisualizer = SMPLVisualizer.from_model(model)
plotter = viz.get_plotter()

neutral_color = Params.NEUTRAL_COLOR
viz.add_mesh(neutral, style="wireframe", color=neutral_color, opacity=1.0)
subset_names = [CoreBodyJoint.LEFT_WRIST.value, CoreBodyJoint.RIGHT_FOOT.value]
viz.add_smpl_joints(neutral, joints=subset_names, labels=False, color=neutral_color, size=Params.POINT_SIZE_NEUTRAL)

fitted_color = Params.FITTED_COLOR
viz.add_mesh(status.output, style="wireframe", color=fitted_color, opacity=1.0)
viz.add_smpl_joints(status.output, joints=subset_names, labels=False, color=fitted_color, size=Params.POINT_SIZE_FITTED)

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
    neutral_sel = model.select_joints(neutral.joints, names=subset_names)[0]
    fitted_sel = model.select_joints(status.output.joints, names=subset_names)[0]
neutral_np = neutral_sel.detach().cpu().numpy()
fitted_np = fitted_sel.detach().cpu().numpy()
add_connection_lines(plotter, neutral_np, tgt_np, color=neutral_color, line_width=Params.LINE_WIDTH, opacity=Params.LINE_OPACITY)
add_connection_lines(plotter, fitted_np, tgt_np, color=fitted_color, line_width=Params.LINE_WIDTH, opacity=Params.LINE_OPACITY)

plotter.add_text("SMPL-X 3D keypoint fit (helper)", font_size=12)
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
del plotter
del viz
try:
    import gc

    gc.collect()
except Exception:
    pass
