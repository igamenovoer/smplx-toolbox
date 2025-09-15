"""Manual smoke script — 3D keypoint fitting via helper + VPoser prior.

Mirrors the baseline VPoser smoke test but uses the helper; intended for
ad‑hoc interactive runs, not collected by pytest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import smplx

from smplx_toolbox.core import NamedPose
from smplx_toolbox.core.constants import CoreBodyJoint, ModelType
from smplx_toolbox.core.unified_model import UnifiedSmplInputs, UnifiedSmplModel
from smplx_toolbox.fitting import SmplKeypointFittingHelper
from smplx_toolbox.utils import select_device
from smplx_toolbox.vposer.model import VPoserModel


class Params:
    # Std‑dev of Gaussian noise (meters) added to neutral joints to form targets
    NOISE_SCALE: float = 0.15
    # Number of outer steps to run; effective updates = STEPS * INNER_ITERS
    STEPS: int = 150
    # Inner optimizer iterations per yielded step (keep =1 to match baseline)
    INNER_ITERS: int = 1
    # Adam learning rate
    LR: float = 1e-2
    # L2 weight on intrinsic body pose (excludes global orient); set 0.0 to disable
    L2_WEIGHT: float = 1e-4
    # L2 weight on shape coefficients (betas); set 0.0 to disable
    SHAPE_L2_WEIGHT: float = 1e-3
    # Robustifier kind for data term: "l2", "huber", or "gmof"
    ROBUST_KIND: str = "gmof"
    # Robustifier scale (larger = less downweighting of outliers)
    ROBUST_RHO: float = 100.0

    # VPoser prior weights
    VPOSER_POSE_FIT: float = 0.5  # pose reconstruction weight on pose_body
    VPOSER_LATENT_L2: float = 0.2  # latent magnitude regularizer

    # Trainable DOFs (toggle whether these tensors are optimized)
    ENABLE_GLOBAL_ORIENT: bool = True
    ENABLE_GLOBAL_TRANSLATION: bool = True
    ENABLE_SHAPE_DEFORMATION: bool = True


def _select_targets(model: UnifiedSmplModel, names: list[str], noise_scale: float) -> dict[str, torch.Tensor]:
    neutral = model.forward(UnifiedSmplInputs())
    with torch.no_grad():
        sel = model.select_joints(neutral.joints, names=names)
        tgt = sel + float(noise_scale) * torch.randn_like(sel)
    return {names[i]: tgt[:, i] for i in range(len(names))}


def _movable_joint_names() -> list[str]:
    """Wrists, feet, elbows, hips (left and right)."""
    return [
        CoreBodyJoint.LEFT_WRIST.value,
        # CoreBodyJoint.RIGHT_WRIST.value,
        # CoreBodyJoint.LEFT_FOOT.value,
        CoreBodyJoint.RIGHT_FOOT.value,
        # CoreBodyJoint.LEFT_ELBOW.value,
        CoreBodyJoint.RIGHT_ELBOW.value,
        CoreBodyJoint.LEFT_KNEE.value,
        # CoreBodyJoint.RIGHT_KNEE.value,
        # CoreBodyJoint.LEFT_HIP.value,
        CoreBodyJoint.RIGHT_HIP.value,
    ]


print("[smoke-helper] 3D keypoint fitting + VPoser (SMPL-X)")
device = select_device()
print(f"[smoke-helper] Using device: {device}")

model_root = Path("data/body_models")
vposer_ckpt = Path("data/vposer/vposer-v2.ckpt")
if not model_root.exists():
    print("[smoke-helper] data/body_models not found — skipping.")
    raise SystemExit(0)
if not vposer_ckpt.exists():
    print("[smoke-helper] VPoser ckpt not found — skipping.")
    raise SystemExit(0)

try:
    base = smplx.create(str(model_root), model_type="smplx", gender="neutral", use_pca=False, batch_size=1, ext="pkl")
except AssertionError as e:
    print(f"[smoke-helper] Missing SMPL-X resources ({e}); skipping.")
    raise SystemExit(0)
base = base.to(device)
model = UnifiedSmplModel.from_smpl_model(base)

names = _movable_joint_names()
targets = _select_targets(model, names, Params.NOISE_SCALE)

# Seed
seed = uuid4().int & 0xFFFFFFFF
torch.manual_seed(seed)
np.random.seed(seed)

# Initial trainables
npz = NamedPose(model_type=ModelType(str(model.model_type)), batch_size=1)
npz.intrinsic_pose = torch.nn.Parameter(torch.zeros_like(npz.intrinsic_pose))
root_orient = (
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
    global_orient=root_orient,
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

vposer = VPoserModel.from_checkpoint(vposer_ckpt, map_location=device)
helper.vposer_init(vposer_model=vposer)
helper.vposer_set_reg(w_pose_fit=Params.VPOSER_POSE_FIT, w_latent_l2=Params.VPOSER_LATENT_L2)

it = helper.init_fitting(
    init, optimizer="adam", lr=Params.LR, num_iter_per_step=Params.INNER_ITERS
)
for i, status in zip(range(Params.STEPS), it):
    if i % 10 == 0 or i == Params.STEPS - 1:
        print(f"[smoke-helper] iter {status.step:03d} | loss = {status.loss_total:.6f}")

# Visualize original vs target vs fitted (helper)
from smplx_toolbox.visualization import SMPLVisualizer, add_connection_lines
import pyvista as pv

viz: SMPLVisualizer = SMPLVisualizer.from_model(model)
plotter = viz.get_plotter()

neutral = model.forward(UnifiedSmplInputs())
neutral_color = (0.5, 0.5, 0.5)
viz.add_mesh(neutral, style="wireframe", color=neutral_color, opacity=1.0)
names = _movable_joint_names()
viz.add_smpl_joints(neutral, joints=names, labels=False, color=neutral_color, size=0.01)

fitted_color = (0.0, 0.3, 1.0)
viz.add_mesh(status.output, style="wireframe", color=fitted_color, opacity=1.0)
viz.add_smpl_joints(status.output, joints=names, labels=False, color=fitted_color, size=0.012)

with torch.no_grad():
    tgt_stack = torch.stack([targets[nm][0] for nm in names], dim=0)
tgt_np = tgt_stack.detach().cpu().numpy()
plotter.add_points(tgt_np, color=(1.0, 0.0, 0.0), render_points_as_spheres=True, point_size=12)

with torch.no_grad():
    neutral_sel = model.select_joints(neutral.joints, names=names)[0]
    fitted_sel = model.select_joints(status.output.joints, names=names)[0]
neutral_np = neutral_sel.detach().cpu().numpy()
fitted_np = fitted_sel.detach().cpu().numpy()
add_connection_lines(plotter, neutral_np, tgt_np, color=neutral_color, line_width=2, opacity=0.9)
add_connection_lines(plotter, fitted_np, tgt_np, color=fitted_color, line_width=2, opacity=0.9)

plotter.add_text("SMPL-X 3D keypoint fit + VPoser (helper)", font_size=12)
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
