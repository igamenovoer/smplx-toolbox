# Data Containers

The unified pipeline uses typed containers to make inputs/outputs explicit and safe.

## UnifiedSmplInputs

Holds a single `NamedPose` as the preferred source of intrinsic pose (pelvis excluded)
and derives SMPL/SMPL‑H/SMPL‑X kwargs from it. You can inspect/edit joint AAs via
`named_pose.packed_pose` and convenience getters/setters on `NamedPose`.

Segmented pose fields have been removed. Provide global orientation separately via
`global_orient: (B, 3)`. Missing segments are zero‑filled as needed for the target model.

Orientation
- Use `inputs.global_orient` for the `(B, 3)` global orientation (pelvis). `NamedPose`
  stores intrinsic joints only and excludes pelvis by design.

::: smplx_toolbox.core.containers.UnifiedSmplInputs

## NamedPose

A lightweight utility for inspecting and editing intrinsic axis‑angle poses `(B, N, 3)`
by joint name (pelvis excluded), using the model type’s joint namespace.

Key conversion helper
- `to_model_type(smpl_type, copy=False) -> NamedPose`
  - Converts the instance to another model type by joint name.
  - When `copy=False`, the returned `packed_pose` is built from views/concats of views so gradients through it flow back to the source where joints overlap (superset targets are composed via `torch.cat`).
  - When `copy=True`, values are cloned so the result is independent.

::: smplx_toolbox.core.containers.NamedPose

Additional notes and helpers
- `to_dict(with_pelvis: bool = False)` excludes pelvis by default; when set, includes a zero‑AA pelvis entry.
- `pelvis` property returns zero AA `(B, 3)` for convenience when constructing full poses for LBS.
- Getters/setters for `'pelvis'` raise `KeyError` to emphasize it is not part of the intrinsic pose.

## UnifiedSmplOutput

Unified outputs (vertices, faces, joints in SMPL‑X 55‑joint layout, flattened full pose used in LBS) plus an `extras` dict with raw joints and mappings.

::: smplx_toolbox.core.containers.UnifiedSmplOutput
