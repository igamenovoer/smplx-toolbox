# Refactor Plan: Simplify UnifiedSmplInputs to hold a single NamedPose

## What to Refactor

- Centralize all pose interpretation/conversion in `NamedPose` and make
  `UnifiedSmplInputs` a thin adapter that slices from it.
- Replace segmented pose fields in `UnifiedSmplInputs`
  (`root_orient`, `pose_body`, `left/right_hand_pose`, `pose_jaw`,
  `left/right_eye_pose`) with a single `named_pose: NamedPose` as the preferred
  source of pose truth (legacy fields remain temporarily for compatibility).
- Provide ergonomic, gradient-safe editing via
  `NamedPose.set_joint_pose_value(...)`, and cross-model conversion via
  `NamedPose.to_model_type(...)`.
- Move aggregate helpers (`hand_pose()`, `eyes_pose()`) from
  `UnifiedSmplInputs` to `NamedPose`.

## Why Refactor

- Single source of truth: avoid duplicated pose segments drifting out of sync.
- Simpler API: set/get by joint names through `NamedPose` rather than manual
  concatenation/splitting of (B, 63/45/3) slices.
- Safer edits: `NamedPose.set_joint_pose_value(...)` is gradient-safe and clear.
- Clearer conversions: `UnifiedSmplInputs` becomes a thin adapter that slices
  from `named_pose` to build SMPL/SMPL-H/SMPL-X kwargs.

## How to Refactor

Phased plan to minimize breakage and allow smooth migration. Items marked [DONE]
have been implemented in this branch.

1) Introduce `named_pose` and centralize pose logic [DONE]
- Add `named_pose: NamedPose | None = None` to `UnifiedSmplInputs`. [DONE]
- Prefer `named_pose` when present; keep segmented fields for compatibility. [DONE]
- Add internal slicers in `UnifiedSmplInputs` to derive per-family kwargs from `named_pose`. [DONE]
- Move aggregate helpers to `NamedPose`: `hand_pose()` and `eyes_pose()`. [DONE]
- Deprecate `UnifiedSmplInputs.hand_pose` / `eyes_pose` to delegate to `NamedPose`. [DONE]

2) Update conversion methods to read from `named_pose` [DONE]
- `to_smpl_inputs` builds root/body from `named_pose`. [DONE]
- `to_smplh_inputs` adds hand poses from `named_pose`. [DONE]
- `to_smplx_inputs` adds jaw/eye poses from `named_pose`. [DONE]
- Shapes and key names remain unchanged. [DONE]

3) Improve NamedPose ergonomics [DONE]
- Add `set_joint_pose_value(name, pose: Tensor | np.ndarray)`; no-grad copy. [DONE]
- Add `to_model_type(smpl_type, copy=False)`:
  - Name-based mapping to rebuild the target packed pose. [DONE]
  - When `copy=False`, concatenates views so gradients flow back to the source. [DONE]
  - When `copy=True`, clones and ensures `.contiguous()` layout. [DONE]
- Remove obsolete `.repeat()` from `NamedPose`. [DONE]

4) Migrate internal call sites and documentation [PARTIAL]
- docs: Updated `docs/core/containers.md` to describe `named_pose` and `to_model_type`. [DONE]
- Remaining docs: `docs/core/unified_model.md`, `docs/usage.md` examples should prefer `named_pose`. [TODO]
- Code: No downstream API changes; `UnifiedSmplModel` and builders remain compatible. [DONE]

5) Tests [PARTIAL]
- Keep compatibility with segmented fields while deprecated. [DONE]
- Add targeted tests: [TODO]
  - `NamedPose.to_model_type(copy=False/True)` gradient linkage and contiguity.
  - `NamedPose.hand_pose()` / `eyes_pose()` shapes and presence across SMPL/SMPLH/SMPLX.
  - `UnifiedSmplInputs(named_pose=npz)` produces expected kwargs.

6) Removal phase (follow-up PR) [PENDING]
- Remove segmented pose fields from `UnifiedSmplInputs` signature and all references.
- Drop deprecated `UnifiedSmplInputs.hand_pose` / `eyes_pose`.

## Before/After Examples

Before (segmented fields):

```python
inputs = UnifiedSmplInputs(
    root_orient=torch.zeros(1, 3),
    pose_body=torch.zeros(1, 63),
    left_hand_pose=torch.zeros(1, 45),
    right_hand_pose=torch.zeros(1, 45),
)
out = model.forward(inputs)
```

After (single NamedPose):

```python
from smplx_toolbox.core import NamedPose
from smplx_toolbox.core.constants import ModelType

# Create a NamedPose (SMPL-X namespace recommended for full coverage)
npz = NamedPose(model_type=ModelType.SMPLX, batch_size=1)
# Set joint AA values as needed (gradient-safe):
npz.set_joint_pose_value("left_wrist", torch.zeros(1, 3))

# Use as the sole pose input
inputs = UnifiedSmplInputs(named_pose=npz, betas=torch.zeros(1, 10))
out = model.forward(inputs)
```

Additional examples

1) Aggregate helpers from NamedPose

```python
npz = NamedPose(model_type=ModelType.SMPLX, batch_size=2)
hands_aa = npz.hand_pose()   # (B, 90) or None for SMPL
eyes_aa = npz.eyes_pose()    # (B, 6)  or None for SMPL/SMPL-H
```

2) Cross-model conversion with gradient control

```python
npz_x = NamedPose(model_type=ModelType.SMPLX, batch_size=1)
# Fill some joints on SMPL-X...

# View-based mapping (gradients flow back to npz_x)
npz_h_view = npz_x.to_model_type(ModelType.SMPLH, copy=False)

# Independent clone with contiguous memory
npz_s_clone = npz_x.to_model_type(ModelType.SMPL, copy=True)
assert npz_s_clone.packed_pose.is_contiguous()
```

## Expected Outcome

- `UnifiedSmplInputs` becomes simpler and less error-prone; poses are managed via a single `NamedPose`.
- Conversion to SMPL/SMPL-H/SMPL-X kwargs is derived consistently from `NamedPose`.
- Codebase-wide consistency: docs and examples show a single, coherent way to set/get joint poses.
- Backward compatibility during the migration window; minimal churn for downstream components.
- Clear gradient semantics for cross-model conversions (`copy` flag) and consolidated aggregate helpers.

## References

- Source files touched/remaining:
  - [DONE] `src/smplx_toolbox/core/containers.py` (UnifiedSmplInputs + NamedPose)
  - [DONE] `docs/core/containers.md` (pattern + conversion helper)
  - [TODO] `docs/core/unified_model.md`, `docs/usage.md` (examples -> named_pose)
  - [COMPAT] `src/smplx_toolbox/core/unified_model.py` (no API change)
  - [COMPAT] `src/smplx_toolbox/visualization/plotter.py` (neutral defaults)
  - [COMPAT] `src/smplx_toolbox/optimization/pose_prior_vposer_builder.py`
- Related enums and joint ordering:
  - `src/smplx_toolbox/core/constants.py` (`CoreBodyJoint`, `FaceJoint`, hand joint ordering)
- Skeleton index comparison (for name-aligned mapping):
  - `context/hints/smplx-kb/compare-smpl-skeleton.md`
- Tests to revisit:
  - `tests/test_unified_model.py`
  - `unittests/smplx_toolbox/core/test_unified_model.py`
  - `tests/fitting/*` and `unittests/fitting/*` (exercise conversion correctness)
- Third-party docs:
  - VPoser interop utilities in `src/smplx_toolbox/vposer/model.py`
