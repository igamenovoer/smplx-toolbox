# Data Containers

The unified pipeline uses typed containers to make inputs/outputs explicit and safe.

## UnifiedSmplInputs

Segmented axis‑angle inputs (root/body/hands/face) plus shape/expr/translation. Missing segments are zero‑filled as needed for the target model.

::: smplx_toolbox.core.containers.UnifiedSmplInputs

## NamedPose

A lightweight utility for inspecting and editing packed axis‑angle poses `(B, N, 3)` by joint name, using the model type’s joint namespace.

::: smplx_toolbox.core.containers.NamedPose

## UnifiedSmplOutput

Unified outputs (vertices, faces, joints in SMPL‑X 55‑joint layout, flattened full pose used in LBS) plus an `extras` dict with raw joints and mappings.

::: smplx_toolbox.core.containers.UnifiedSmplOutput
