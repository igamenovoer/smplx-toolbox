# How to Use VPoser with SMPL / SMPL-H / SMPL-X

This guide explains what VPoser expects and how to convert pose vectors across SMPL-family models to be VPoser-compatible. It also shows how to round-trip between VPoser and model-specific pose representations.

References
- Local source: `context/refcode/human_body_prior/src/human_body_prior/models/vposer_model.py`
  - VPoser is trained on body pose only (no root, no hands/face), using 21 body joints.
  - Encoder input: 21 × 3 axis-angle (AA) = 63 DoF.
  - Decoder output: 21 × 3 axis-angle per joint, plus a 9D matrot view.
- SMPLify-X usage: `context/refcode/smplify-x/smplifyx/fitting.py`
  - When using VPoser with SMPL, they append 6 zeros for wrists because SMPL body pose = 23 × 3, while VPoser produces 21 × 3.
- Upstream projects: Human Body Prior (VPoser)
  - GitHub: https://github.com/nghorbani/human_body_prior

What VPoser Expects
- Body-only pose, 21 joints, axis-angle format.
- Shape: `(B, 21, 3)` for AA or flattened `(B, 63)`.
- No global/root orientation (handled separately) and no hands/face.

SMPL Family Body Pose Dimensions (Axis-Angle)
- SMPL: body_pose is 23 × 3 (includes wrists), global_orient is separate 1 × 3.
- SMPL-H: body_pose is 21 × 3 (no wrists in body block; hands are separate). Global orient separate.
- SMPL-X: body_pose is 21 × 3 (no wrists in body block; hands/face separate). Global orient separate.

Key Takeaways
- VPoser operates on the 21-joint body pose shared by SMPL-H and SMPL-X.
- For SMPL specifically, the body_pose has 23 joints; to use VPoser you:
  - Encode: drop the two wrist joints (keep the first 21 joints) → feed 21×3 to VPoser.
  - Decode: get 21×3 from VPoser and append 6 zeros to fill the two wrist joints (to reach 23×3) before passing to SMPL.
  - This mirrors `smplify-x` behavior.

Joint Order Assumptions
- VPoser’s 21-body joints follow the standard SMPL/SMPL-X body ordering (excluding hands and face). In SMPL, the first 21 joints correspond to this set; the last two are wrists.
- For SMPL-H/X, the body block is already 21 joints in the expected order.

Code Snippets

Encode existing body pose to VPoser latent
```python
# Given: body_pose_aa
#   SMPL-X/H: (B, 21, 3)
#   SMPL: (B, 23, 3) → we will slice first 21

from torch import nn, Tensor

vposer: nn.Module  # loaded VPoser model (see human_body_prior)

if model_type in ("smplx", "smplh"):
    pose21 = body_pose_aa  # already (B, 21, 3)
elif model_type == "smpl":
    pose21 = body_pose_aa[:, :21]  # drop wrists (2 × 3)
else:
    raise ValueError("Unknown model type")

# VPoser encodes 21x3 AA to a Normal distribution in latent space
q_z = vposer.encode(pose21.reshape(pose21.shape[0], -1))  # (B, 63) → Normal
z_sample = q_z.rsample()  # (B, latentD)
```

Decode VPoser latent to body pose AA and adapt to model
```python
# Decode latent to 21×3 AA
decoded = vposer.decode(z_sample)
pb_aa: Tensor = decoded["pose_body"]  # (B, 21, 3)

if model_type in ("smplx", "smplh"):
    body_pose_aa = pb_aa  # directly compatible (21×3)
elif model_type == "smpl":
    B = pb_aa.shape[0]
    wrist_zeros = pb_aa.new_zeros((B, 2, 3))  # fill wrists with zeros
    body_pose_aa = torch.cat([pb_aa, wrist_zeros], dim=1)  # (B, 23, 3)
else:
    raise ValueError("Unknown model type")

# global_orient (root) is handled separately and not part of VPoser
```

Using VPoser with UnifiedSmplModel
```python
# Pseudocode using our unified wrapper
from smplx_toolbox.core.unified_model import UnifiedSmplModel

model = UnifiedSmplModel.from_smpl_model(smplx_like_model)

# 1) Optimize or set vposer latent z
z = torch.nn.Parameter(torch.zeros(B, latentD))

# 2) Decode to body pose
pb_aa = vposer.decode(z)["pose_body"]  # (B, 21, 3)

# 3) Adapt to the wrapped model
if model.model_type in ("smplx", "smplh"):
    body_pose_aa = pb_aa
else:  # smpl
    body_pose_aa = torch.cat([pb_aa, torch.zeros(B, 2, 3, device=pb_aa.device, dtype=pb_aa.dtype)], dim=1)

# 4) Feed to the model (global_orient handled separately)
out = model(UnifiedSmplInputs(body_pose=body_pose_aa.reshape(B, -1), global_orient=global_orient_aa))
```

Notes and Tips
- Axis-angle vs matrot: VPoser decodes to both AA and 3×3 rotation matrices; prefer AA to match SMPL inputs.
- Train/test consistency: Ensure the body joint order of your model matches the expected 21-joint ordering. For standard SMPL implementations, the first 21 entries of `body_pose` correspond to the VPoser set.
- Hands/face: VPoser does not cover hands or face. For SMPL-X/H, keep hand/face pose separate.
- Root/global orient: Not part of VPoser. Optimize or set independently.
- SMPL wrists: When using VPoser with SMPL, appending zeros for wrists is common; you can also learn these 2 × 3 AA terms separately if desired (initialize at zero).

Further Reading
- VPoser (Human Body Prior): https://github.com/nghorbani/human_body_prior
- SMPLify-X fitting code and VPoser integration: `context/refcode/smplify-x/smplifyx/fitting.py`
- SMPL / SMPL-H / SMPL-X skeleton comparison: `context/hints/smplx-kb/compare-smpl-skeleton.md`

