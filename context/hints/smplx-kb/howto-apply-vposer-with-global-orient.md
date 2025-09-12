# How to apply VPoser without affecting global orientation

VPoser is a learned prior over the body pose space (21 joints), not the global
orientation. When fitting SMPL/SMPL‑H/SMPL‑X with a VPoser prior, ensure that the
global orientation (aka `global_orient`, sometimes called `root_orient`) is kept
separate from the body pose. This matches the official SMPL‑X and SMPLify‑X code.

## Key Principles

- The model’s global orientation is separate from its body pose.
  - In SMPL: full pose is `cat([global_orient, body_pose])`.
  - In SMPL‑X: full pose is `cat([global_orient, body_pose, jaw_pose, eye_poses, hand_poses])` in rotation‑matrix form.
- VPoser is trained on body pose only (21 joints = 63 DoF AA). Do not feed the 3‑DoF global orient into VPoser.
- Optimize or regularize `global_orient` separately from the VPoser prior.

## References (source code)

- SMPL (axis‑angle):
  - `context/refcode/smplx/smplx/body_models.py` → `full_pose = torch.cat([global_orient, body_pose], dim=1)`
- SMPL‑X (rot‑mat):
  - `context/refcode/smplx/smplx/body_models.py` → concatenate `global_orient`, `body_pose`, face and hands into `full_pose`.
- SMPLify‑X fitting:
  - `context/refcode/smplify-x/smplifyx/fit_single_frame.py`
    - Uses a VPoser latent (`pose_embedding`) to decode body pose only.
    - Optimizes `body_model.global_orient` as a separate parameter.

## Recommended pattern in this toolbox (NamedPose)

We centralize pose in a `NamedPose` instance (`npz`) and derive model kwargs from it.

```python
from smplx_toolbox.core import NamedPose, UnifiedSmplInputs
from smplx_toolbox.core.constants import ModelType
from smplx_toolbox.vposer.model import VPoserModel

# Build trainable NamedPose
npz = NamedPose(model_type=ModelType.SMPLX, batch_size=B)
npz.packed_pose = torch.nn.Parameter(torch.zeros_like(npz.packed_pose))

# Forward via UnifiedSmplInputs; global_orient is a view of npz.root_orient
inputs = UnifiedSmplInputs(named_pose=npz, betas=betas)
out = model.forward(inputs)

# Body pose for VPoser (63 DoF AA, pelvis excluded)
pose_body = VPoserModel.convert_named_pose_to_pose_body(npz)  # (B, 63)

# VPoser prior only on body pose
vp_builder = VPoserPriorLossBuilder.from_vposer(model, vposer)
term_vposer = vp_builder.by_pose(pose_body, w_pose_fit, w_latent_l2)

# Regularize global orientation separately (e.g., simple L2)
reg_global = (npz.root_orient**2).sum()
loss = data_term(out) + term_vposer(out) + lambda_reg * reg_global
```

If you add a convenience wrapper, keep global orientation excluded:

```python
# Optional ergonomic helper (not required):
# term_vposer = vp_builder.by_named_pose(npz, w_pose_fit, w_latent_l2)
# Internally calls convert_named_pose_to_pose_body(npz)
```

## Why this matters

- Global orientation controls how the body sits in world coordinates. It is not
  part of the intrinsic body configuration that VPoser models.
- Feeding `global_orient` to VPoser would wrongly bias the orientation and hurt
  pose fidelity.

## Step‑by‑step checklist

1) Build or optimize a single `NamedPose` (`npz.packed_pose`), not separate tensors.
2) Pass `UnifiedSmplInputs(named_pose=npz, ...)` to the unified model.
3) Derive `pose_body` from `npz` (exclude pelvis) and apply the VPoser prior to it only.
4) Regularize `npz.root_orient` separately (L2, or other priors if desired).
5) Keep hands/face priors independent as needed; do not mix into VPoser.

## Extra tips

- The pelvis AA (`npz.root_orient`) is a view into `npz.packed_pose`, so gradients propagate correctly.
- If you need a detached copy for logging or specific constraints, call `.detach()` or `.clone()` appropriately.
- For SMPL (no face/hands), body pose is still 63 DoF; VPoser expects that.

