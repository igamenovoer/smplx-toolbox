Title: VPoser Pose Prior – Latent and Self-Reconstruction Terms

Context
- Scope: VPoser-based pose priors under `src/smplx_toolbox/optimization/`.
- Model: The VPoser model from `context/refcode/human_body_prior` provides `encode` (q(z|pose)) and `decode` (pose|z).
- Upstream reference: SMPLify‑X (`context/refcode/smplify-x/smplifyx`) uses VPoser by optimizing a low‑dim latent and penalizing its L2 norm.

Goals
- Provide two builder functions for common usages:
  1) Latent-only prior: given a latent `z`, return a weighted L2 loss on `z`.
  2) Pose self-reconstruction + latent prior: given a body pose `pose_in` (AA), encode to `latent_in`, decode back to `pose_out`, and penalize both the reconstruction error and the latent magnitude.

Interfaces (single source of truth)
- `VPoserPriorLossBuilder.from_vposer(model, vposer)` binds the VPoser model.
- Priors:
  - `by_pose_latent(latent: Tensor, weight: float | Tensor) -> SmplLossTerm`
    - Weighted L2 on the latent: `weight * mean(latent**2)`.
  - `by_pose(pose: Tensor, w_pose_fit: float | Tensor, w_latent_l2: float | Tensor) -> SmplLossTerm`
    - Self-reconstruction + latent prior: `MSE(pose_in, pose_out) * w_pose_fit + mean(latent_in**2) * w_latent_l2`, where `latent_in = encode(pose_in).mean` and `pose_out = decode(latent_in)['pose_body']`.
- Convenience encode/decode (for quick workflows; for finer control use the VPoser model directly):
  - `encode_pose_to_latent(pose: Tensor) -> Tensor`
    - Accepts `(B, 63)` or `(B, 21, 3)` AA; returns `(B, Z)` using the encoder mean.
  - `decode_latent_to_pose(latent: Tensor) -> Tensor`
    - Returns body pose `(B, 21, 3)` AA from a latent.
  - PoseByKeypoints interop (in `smplx_toolbox.vposer.model.VPoserModel`):
    - `convert_struct_to_pose(kpts: PoseByKeypoints) -> Tensor`
      - Produces a `(B, 63)` body AA in the exact 21‑joint order VPoser expects.
    - `convert_pose_to_struct(pose_body: Tensor) -> PoseByKeypoints`
      - Converts decoded `(B, 63)`/`(B, 21, 3)` body AA back to a `PoseByKeypoints` with body joints populated.
- Accessor:
  - `builder.vposer` (read-only) exposes the underlying VPoser module for advanced use cases.

Shapes and Conventions
- `pose` is body-only axis‑angle (no hands/face):
  - Accept `(B, 63)` flattened AA or `(B, 21, 3)` AA.
  - Encoder expects `(B, 63)`; the builder reshapes `(B, 21, 3)` to `(B, 63)` internally.
- `latent` is `(B, Z)` (e.g., `Z=32`).
- Reductions use mean to produce well-scaled scalars.

Behavior Details
- Encoding: `latent_in = VPoser.encode(pose_in)`; use the distribution mean as the latent for stability (`latent_in = q.mean`).
- Decoding: `pose_out = VPoser.decode(latent_in)['pose_body']` which returns `(B, 21, 3)`; the builder flattens to `(B, 63)` before MSE.
- The returned loss terms ignore the `UnifiedSmplOutput` and depend only on the provided latent/pose tensors; gradients flow to those tensors for optimization.

Example Usage
- Latent-only prior (SMPLify‑X style):
```python
vp = VPoserPriorLossBuilder.from_vposer(model, vposer)
loss_pose = vp.by_pose_latent(pose_embedding, weight=400.0)
total = data_term(out) + loss_pose(out) + shape_prior(out) + angle_prior(out)
```

- Pose self-reconstruction + latent L2:
```python
pose_body = torch.zeros(B, 63, requires_grad=True, device=device)
vp = VPoserPriorLossBuilder.from_vposer(model, vposer)
loss_pose = vp.by_pose(pose_body, w_pose_fit=1.0, w_latent_l2=0.1)
total = data_term(out) + loss_pose(out)
```

Notes
- For SMPL: VPoser covers 21 body joints; wrists (6 AA) are not decoded. When driving the SMPL model from a decoded VPoser pose, append 6 zeros for wrists.
- Keep VPoser in `eval()` and optimize only the latent or user-provided `pose` tensors.
- Users can combine these terms with 2D/3D data terms, shape priors, and angle priors.

References
- VPoser implementation: `context/refcode/human_body_prior/src/human_body_prior/models/vposer_model.py`
- SMPLify‑X usage: `context/refcode/smplify-x/smplifyx/fitting.py`, `context/refcode/smplify-x/smplifyx/fit_single_frame.py`
## Note: PoseByKeypoints Status (Outdated)

- PoseByKeypoints has been refactored out of the codebase and is no longer supported in new APIs.
- Use UnifiedSmplInputs for providing segmented pose tensors and NamedPose for optional inspection/editing of packed `(B, N, 3)` poses.
- VPoser interop now uses:
  - `VPoserModel.convert_named_pose_to_pose_body(npz: NamedPose) -> Tensor`
  - `VPoserModel.convert_pose_body_to_named_pose(pose_body: Tensor) -> NamedPose`
- This document is kept for archival purposes; design details reflect an earlier iteration.
