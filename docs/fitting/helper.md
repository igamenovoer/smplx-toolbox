Title: Keypoint Fitting Helper

Overview
- `SmplKeypointFittingHelper` orchestrates end‑to‑end keypoint‑based fitting for SMPL/SMPL‑H/SMPL‑X.
- It manages 3D keypoint targets, robust data terms, priors (VPoser and L2), DOF toggles, and a simple iterator for optimization.

Key Concepts
- Targets by name: Provide a dict `name -> (B, 3)`; builder resolves names to unified joints.
- Robust data term: Choose `l2`, `huber`, or `gmof` with scale `rho`.
- Priors
  - Pose L2 (intrinsic) and shape L2 (betas).
  - VPoser prior via `VPoserModel.from_checkpoint` and `VPoserPriorLossBuilder`.
- DOF toggles: enable/disable optimization of global orient, translation, and shape.
- In‑place updates: The helper optimizes the same `nn.Parameter` objects supplied in `UnifiedSmplInputs`.

Quick Start
```python
import torch, smplx
from pathlib import Path
from smplx_toolbox.core import NamedPose
from smplx_toolbox.core.constants import ModelType
from smplx_toolbox.core.unified_model import UnifiedSmplModel, UnifiedSmplInputs
from smplx_toolbox.fitting import SmplKeypointFittingHelper

# 1) Load and wrap model
base = smplx.create("data/body_models", model_type="smplx", gender="neutral", use_pca=False, batch_size=1, ext="pkl")
model = UnifiedSmplModel.from_smpl_model(base)

# 2) Build synthetic 3D targets from a neutral forward + noise
neutral = model.forward(UnifiedSmplInputs())
sel = model.select_joints(neutral.joints, names=["left_wrist", "right_foot"])  # (1,2,3)
targets = {"left_wrist": sel[:, 0] + 0.3 * torch.randn_like(sel[:, 0]),
           "right_foot": sel[:, 1] + 0.3 * torch.randn_like(sel[:, 1])}

# 3) Define trainable inputs
npz = NamedPose(model_type=ModelType(str(model.model_type)), batch_size=1)
npz.intrinsic_pose = torch.nn.Parameter(torch.zeros_like(npz.intrinsic_pose))
go = torch.nn.Parameter(torch.zeros((1, 3), device=model.device, dtype=model.dtype))
betas = torch.nn.Parameter(torch.zeros((1, int(model.num_betas)), device=model.device, dtype=model.dtype))
init = UnifiedSmplInputs(named_pose=npz, global_orient=go, betas=betas)

# 4) Configure helper
helper = SmplKeypointFittingHelper.from_model(model)
helper.set_keypoint_targets(targets, robust="gmof", rho=100.0)
helper.set_dof_global_orient(True)
helper.set_dof_global_translation(False)
helper.set_dof_shape_deform(True)
helper.set_reg_pose_l2(1e-2)
helper.set_reg_shape_l2(1e-2)

# 5) Run optimization
it = helper.init_fitting(init, optimizer="adam", lr=0.05, num_iter_per_step=1)
for i, status in zip(range(100), it):
    if i % 10 == 0 or i == 99:
        print(f"step={status.step:03d} loss={status.loss_total:.6f}")
```

VPoser Integration
- Load via `VPoserModel.from_checkpoint(path, map_location=device)`.
- Convert between NamedPose and `(B,63)` body pose using the static helpers on `VPoserModel`.

```python
from smplx_toolbox.vposer.model import VPoserModel

vposer = VPoserModel.from_checkpoint(Path("data/vposer/vposer-v2.ckpt"), map_location=model.device)
helper.vposer_init(vposer_model=vposer)
helper.vposer_set_reg(w_pose_fit=0.5, w_latent_l2=0.1)
```

API Summary
```python
helper = SmplKeypointFittingHelper.from_model(model)
helper.set_keypoint_targets(targets, weights=None, robust="gmof", rho=100.0, confidence=None)
helper.set_dof_global_orient(True)
helper.set_dof_global_translation(True)
helper.set_dof_shape_deform(True)
helper.set_reg_pose_l2(1e-2)
helper.set_reg_shape_l2(1e-2)
helper.vposer_init(vposer_model)
helper.vposer_set_reg(w_pose_fit=0.3, w_latent_l2=0.1)
enabled = helper.vposer_is_enabled()
it = helper.init_fitting(initial, optimizer="adam", lr=0.05, num_iter_per_step=1)
```

Notes
- The helper updates `nn.Parameter` objects in place; to restart from scratch, clone and restore values or rebuild parameters.
- Disabled DOFs are excluded from the optimizer and detached during forward.
- For longer runs, increase `num_iter_per_step` to reduce Python/yield overhead.

