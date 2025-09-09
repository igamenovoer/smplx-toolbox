# Data Containers

The unified pipeline uses typed containers to make inputs/outputs explicit and safe.

## UnifiedSmplInputs

Segmented axis‑angle inputs (root/body/hands/face) plus shape/expr/translation. Missing segments are zero‑filled as needed for the target model.

::: smplx_toolbox.core.containers.UnifiedSmplInputs

## PoseByKeypoints

Per‑joint axis‑angle by name (e.g., `left_shoulder`, `jaw`). Convenient for specifying partial poses. Automatically converted to segments via `UnifiedSmplInputs.from_keypoint_pose`.

::: smplx_toolbox.core.containers.PoseByKeypoints

## UnifiedSmplOutput

Unified outputs (vertices, faces, joints in SMPL‑X 55‑joint layout, flattened full pose used in LBS) plus an `extras` dict with raw joints and mappings.

::: smplx_toolbox.core.containers.UnifiedSmplOutput

