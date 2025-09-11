Title: VPoser Model and Body Pose Input Mapping

Overview
- VPoser is a learned prior over 21 body joints (axis‑angle), used to regularize body pose in SMPL/SMPL‑X fitting.
- This toolbox includes a minimal runtime (`smplx_toolbox.vposer.model.VPoserModel`) and convenience helpers to map between user‑friendly structures and the tensors VPoser expects.

Body Pose Mapping (21 joints)
- VPoser expects body axis‑angle in this exact order (pelvis/root excluded):
  - `left_hip, right_hip, spine1, left_knee, right_knee, spine2,
    left_ankle, right_ankle, spine3, left_foot, right_foot,
    neck, left_collar, right_collar, head,
    left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist`
- Each joint is a 3‑vector (axis‑angle). Stacked they form `(B, 63)`.

Interop Helpers
- In `smplx_toolbox.vposer.model.VPoserModel`:
  - `convert_named_pose_to_pose_body(npz: NamedPose) -> Tensor`
    - Builds `(B, 63)` body AA from a `NamedPose` using the mapping above. Missing joints are zero‑filled.
  - `convert_pose_body_to_named_pose(pose_body: Tensor) -> NamedPose`
    - Converts `(B, 63)` or `(B, 21, 3)` body AA back to a `NamedPose` with body joints populated (SMPL namespace).

Encoding/Decoding with VPoser
```python
from smplx_toolbox.vposer.model import VPoserModel
from smplx_toolbox.optimization import VPoserPriorLossBuilder

# Build body AA from a NamedPose
pose_body = VPoserModel.convert_named_pose_to_pose_body(npz)  # (B, 63)

# Encode to latent (uses mean of posterior)
vp_builder = VPoserPriorLossBuilder.from_vposer(model, vposer)
z = vp_builder.encode_pose_to_latent(pose_body)

# Decode back to pose AA (B, 21, 3)
pose_out = vp_builder.decode_latent_to_pose(z)

# Convert to NamedPose for inspection (SMPL namespace)
npz_out = VPoserModel.convert_pose_body_to_named_pose(pose_out)
```

Tips
- Keep VPoser in `eval()` and optimize only your latent or pose variables.
- For SMPL (not X/H), append 6 zeros for wrists when driving the model from decoded VPoser output.

References
- Human Body Prior (VPoser): https://github.com/nghorbani/human_body_prior
- SMPLify‑X usage example: `context/refcode/smplify-x/smplifyx/fit_single_frame.py`
