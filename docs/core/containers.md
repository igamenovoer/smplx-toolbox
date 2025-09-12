# Data Containers

The unified pipeline uses typed containers to make inputs/outputs explicit and safe.

## UnifiedSmplInputs

Holds a single `NamedPose` as the preferred source of pose truth and derives
SMPL/SMPL‑H/SMPL‑X kwargs from it. You can inspect/edit joint AAs via
`named_pose.packed_pose` and convenience getters/setters on `NamedPose`.

Legacy segmented fields (`root_orient`, `pose_body`, `left/right_hand_pose`,
`pose_jaw`, `left/right_eye_pose`) are still accepted for backward
compatibility but are deprecated. When `named_pose` is provided, these fields
are ignored. Missing segments are zero‑filled as needed for the target model.

Orientation
- Use `inputs.global_orient` to access the `(B, 3)` global orientation; when
  `named_pose` is set, this is a view of the pelvis joint axis‑angle from the
  packed pose. Prefer editing via `npz.root_orient` or
  `npz.set_joint_pose_value('pelvis', ...)` for clarity and gradient safety.
  The legacy `root_orient` field is deprecated.

::: smplx_toolbox.core.containers.UnifiedSmplInputs

## NamedPose

A lightweight utility for inspecting and editing packed axis‑angle poses `(B, N, 3)` by joint name, using the model type’s joint namespace.

Key conversion helper
- `to_model_type(smpl_type, copy=False) -> NamedPose`
  - Converts the instance to another model type by joint name.
  - When `copy=False`, the returned `packed_pose` is built from views/concats of views so gradients through it flow back to the source where joints overlap (superset targets are composed via `torch.cat`).
  - When `copy=True`, values are cloned so the result is independent.

::: smplx_toolbox.core.containers.NamedPose

Also provides convenient views:
- `root_orient` – `(B, 3)` view of the pelvis joint AA (global orientation).

## UnifiedSmplOutput

Unified outputs (vertices, faces, joints in SMPL‑X 55‑joint layout, flattened full pose used in LBS) plus an `extras` dict with raw joints and mappings.

::: smplx_toolbox.core.containers.UnifiedSmplOutput
