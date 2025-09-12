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

1) Introduce `named_pose` (intrinsic pose only) and centralize pose logic [DONE]
- Add `named_pose: NamedPose | None = None` to `UnifiedSmplInputs`. [DONE]
- NamedPose now stores INTRINSIC joints only (no pelvis/global-orient). [DONE]
- Add `global_orient: Tensor | None` as a separate member on `UnifiedSmplInputs` to pass to SMPL/SMPL‑X. [DONE]
- Prefer `named_pose` when present; keep segmented fields for compatibility. [DONE]
- Add internal slicers in `UnifiedSmplInputs` to derive per-family kwargs from `named_pose`. [DONE]
- Move aggregate helpers to `NamedPose`: `hand_pose()` and `eyes_pose()`. [DONE]

2) Update conversion methods to read from `named_pose` [DONE]
- `to_smpl_inputs` builds body from `named_pose` and uses `global_orient` separately. [DONE]
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
 - New: `pelvis` property returns zero AA `(B,3)`; getters/setters for `'pelvis'` raise `KeyError`. `to_dict(with_pelvis=False)` excludes pelvis by default. [DONE]

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

## Fitting Migration (Optimization module)

### What to Change
- Move fitting parameterization from separate tensors (`root_orient`, `pose_body`, `left/right_hand_pose`) to a single `NamedPose` parameter (`npz.packed_pose`: `(B, N, 3)`).
- Build `UnifiedSmplInputs` with `named_pose=npz` and keep betas/translation as needed.
- For priors and regularizers, derive slices from `NamedPose`:
  - `global_orient` via `npz.root_orient` (B, 3) — view of pelvis.
  - Body `(B, 63)` via `VPoserModel.convert_named_pose_to_pose_body(npz)`.
  - Hands via `npz.hand_pose()` or collecting explicit finger joint names.

### Why Change
- Single source of truth prevents drift between multiple pose tensors.
- Gradients flow cleanly through a unified `(B, N, 3)` parameter.
- Aligns with `UnifiedSmplInputs` conversion rules and upcoming removal of segmented fields.

### How to Change
1) Parameterize with `NamedPose`
```python
npz = NamedPose(model_type=model.model_type, batch_size=B)
npz.packed_pose = torch.nn.Parameter(torch.zeros_like(npz.packed_pose))
inputs = UnifiedSmplInputs(named_pose=npz, betas=betas)
out = model.forward(inputs)
```

2) Data terms: unchanged
- `KeypointMatchLossBuilder` and `ProjectedKeypointMatchLossBuilder` operate on `UnifiedSmplOutput` — no change required.

3) VPoser prior
- Option A (minimal):
```python
pose_body = VPoserModel.convert_named_pose_to_pose_body(npz)
term_vposer = VPoserPriorLossBuilder.from_vposer(model, vposer).by_pose(pose_body, w_pose_fit, w_latent_l2)
```
- Option B (ergonomic): add `by_named_pose(npz, w_pose_fit, w_latent_l2)` wrapper in `VPoserPriorLossBuilder` to perform the conversion internally. [TODO]

4) Regularization
- Use views/slices from `npz`:
```python
reg = (npz.root_orient**2).sum()
reg += (VPoserModel.convert_named_pose_to_pose_body(npz)**2).sum()
# Optional: include hands by collecting joint names via HandFingerJoint
```

### Tests and Smoke Scripts to Update
- `tests/fitting/smoke_test_keypoint_match.py`:
  - Replace separate `root_orient`/`pose_body`/`left_hand_pose` params with a single `npz.packed_pose` param.
  - Forward with `UnifiedSmplInputs(named_pose=npz, ...)`.
  - L2 reg computed from `npz.root_orient` and body slice.
- `tests/fitting/smoke_test_keypoint_match_vposer.py`:
  - Same parameterization via `NamedPose`.
  - Use `VPoserModel.convert_named_pose_to_pose_body(npz)` when calling `by_pose(...)`.
- If present, update `unittests/fitting/*` similarly.

### Migration Strategy
- Phase 1 (compat window): 
  - Keep builders unchanged; move fitting scripts/tests to `NamedPose`.
  - Optionally add `VPoserPriorLossBuilder.by_named_pose` for convenience.
- Phase 2 (removal):
  - Remove segmented fields from `UnifiedSmplInputs` signature and deprecated properties.
  - Ensure all fitting code relies solely on `NamedPose`.

### References (Fitting)
- Code: `src/smplx_toolbox/optimization/*`
- Smoke scripts: `tests/fitting/smoke_test_keypoint_match.py`, `tests/fitting/smoke_test_keypoint_match_vposer.py`
- Loss builders operate on outputs only; no changes required beyond call sites.
