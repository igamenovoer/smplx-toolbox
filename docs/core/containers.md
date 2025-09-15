# Data Containers

The unified pipeline uses typed containers to make inputs/outputs explicit and safe.

## UnifiedSmplInputs

Holds a single `NamedPose` as the preferred source of intrinsic pose (pelvis excluded)
and derives SMPL/SMPL‑H/SMPL‑X kwargs from it. Inspect/edit joint AAs via
`named_pose.intrinsic_pose` and convenience getters/setters on `NamedPose`. The
pelvis rotation can be carried separately in `named_pose.root_pose`.

Segmented pose fields have been removed. Provide global orientation separately via
`global_orient: (B, 3)`. Missing segments are zero‑filled as needed for the target model.

Orientation
- Use `inputs.global_orient` for `(B, 3)` global orientation (pelvis). If unset and
  `named_pose.root_pose` is present, it is used automatically.

::: smplx_toolbox.core.containers.UnifiedSmplInputs

## NamedPose

A lightweight utility for inspecting and editing intrinsic axis‑angle poses `(B, N, 3)`
by joint name (pelvis excluded), using the model type’s joint namespace. Stores
the pelvis rotation separately in `root_pose: (B, 3)`.

Key conversion helper
- `to_model_type(smpl_type, copy=False) -> NamedPose`
  - Converts to another model type by joint name. Intrinsic pose uses views when
    `copy=False` to preserve gradient flow; clones when `copy=True`.
- `to_full_pose(pelvis_pose: Tensor | None = None) -> Tensor`
  - Returns full pose `(B, N+1, 3)` by prepending pelvis (from `pelvis_pose` or
    `root_pose` if present, zeros otherwise).

::: smplx_toolbox.core.containers.NamedPose

Additional notes and helpers
- `to_dict(pelvis_pose: Tensor | None = None)` includes pelvis by default (uses
  `pelvis_pose` if given, else `root_pose`/zeros).
- `pelvis` property returns `(B, 3)` from `root_pose` (or zeros if unset).
- Getters now accept `'pelvis'`; setters on `'pelvis'` write `root_pose`.

## UnifiedSmplOutput

Unified outputs (vertices, faces, joints in SMPL‑X 55‑joint layout, flattened full pose used in LBS) plus an `extras` dict with raw joints and mappings.

::: smplx_toolbox.core.containers.UnifiedSmplOutput
