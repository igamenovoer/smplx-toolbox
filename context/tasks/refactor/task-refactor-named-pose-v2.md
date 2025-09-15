Title: Refactor Plan — NamedPose v2 (root-aware, SMPL indexing)

What to Refactor
- Core class: `NamedPose` in `src/smplx_toolbox/core/containers.py`.
- Related APIs and call sites that interact with `NamedPose`:
  - `UnifiedSmplInputs` helpers that slice body/hand/face pose from `NamedPose`.
  - VPoser interop: `src/smplx_toolbox/vposer/model.py` convenience functions using `NamedPose`.
  - Any code/docs/tests referencing `NamedPose.packed_pose` or assuming pelvis is excluded and index-shifted.
  - Design/docs that mention `NamedPose.packed_pose` ordering (e.g., fitting helper design).

Why Refactor
- Usability: Store and access the pelvis (root) rotation with the rest of the pose to reduce special-casing and error-prone plumbing.
- Correct indexing: Align `NamedPose` getters-by-index with SMPL indexing (pelvis at index 0), eliminating the off-by-one mental model.
- Clarity: Rename `packed_pose` to `intrinsic_pose` to reflect that it excludes pelvis and encodes only intrinsic joint rotations; provide a new helper to produce a full pose including pelvis when needed.
- Interop: Simplify using `NamedPose` downstream in optimization and conversion flows that occasionally need the root rotation.

How to Refactor
1) Introduce NamedPose v2 members and semantics
   - Rename member `packed_pose -> intrinsic_pose` (still `(B, N, 3)`, pelvis excluded).
   - Add optional `root_pose: Tensor | None` of shape `(B, 3)` for pelvis/global orientation.
   - Update `pelvis` property to return the actual pelvis rotation (use `root_pose` if set; otherwise zeros `(B, 3)`).
   - Allow getters to return pelvis by name/index.
     - Name-based: `get_joint_pose('pelvis')` should return a `(B, 1, 3)` tensor (or zeros if `root_pose is None`).
     - Index-based: SMPL indexing where index 0 is pelvis. For `k > 0`, map to `intrinsic_pose[:, k-1, :]`.
   - Backward compatibility: Temporarily keep a deprecated `@property packed_pose` proxy to `intrinsic_pose` with a `DeprecationWarning` (and a symmetrical setter), so downstream code continues to run during migration.

2) Add a new convenience API on NamedPose
   - `to_full_pose(pelvis_pose: Tensor | None = None) -> Tensor`
     - Returns `(B, N+1, 3)` where the first joint is pelvis, followed by `intrinsic_pose` joints in model order.
     - If `pelvis_pose` is provided, use it; else use `self.pelvis`.
   - `to_dict(pelvis_pose: Tensor | None = None) -> dict`
     - Always include pelvis. If `pelvis_pose` is provided, prefer it; else use `self.pelvis`.

3) Update UnifiedSmplInputs helpers to use v2 API
   - Where the code currently reads `npz.packed_pose` for batch/device/dtype, switch to `npz.intrinsic_pose`.
   - Body/hand/face slicers remain based on intrinsic joints (pelvis excluded). No change to which joints are included in body/hand/face groups, but adjust attribute names.
   - Optional small enhancement: when `named_pose.root_pose` is present and `global_orient` is not provided, populate `global_orient` from `named_pose.root_pose` to preserve current forward-pass behavior; this keeps calling code unchanged while supporting root-aware `NamedPose`.

4) Update VPoser interop helpers
   - `convert_named_pose_to_pose_body(npz: NamedPose)`
     - Switch from `npz.packed_pose` -> `npz.intrinsic_pose` for B/device/dtype probing.
     - No change to the body joint list; output remains `(B, 63)` (pelvis excluded).
   - `convert_pose_body_to_named_pose(pose_body: Tensor)`
     - Create the `NamedPose` with `intrinsic_pose` filled; do not set `root_pose` (remains `None`) to preserve previous “zero pelvis” behavior unless explicitly provided by the caller later.

5) Sweep and update call sites and docs
   - Code: Find usages of `NamedPose.packed_pose` and migrate to `intrinsic_pose`.
     - Known hotspots (from ripgrep):
       - `src/smplx_toolbox/core/containers.py` (self-references, validation, slicers)
       - `src/smplx_toolbox/vposer/model.py` (lines where `npz.packed_pose` is referenced)
       - `src/smplx_toolbox/core/unified_model.py` (imports and examples)
     - Adjust any index-based getters to respect pelvis at index 0.
   - Design/docs:
     - Update fitting helper design to reference `NamedPose.intrinsic_pose` as the canonical packed order for intrinsic joints.
     - Update docs (`docs/vposer.md`, usage snippets) to mention `intrinsic_pose` and new `to_full_pose()`.
   - Tests:
     - Update or add unit tests covering:
       - `get_joint_pose('pelvis')` returns correct shape/value with and without `root_pose`.
       - `to_full_pose()` prepends pelvis and preserves device/dtype.
       - Back-compat alias `.packed_pose` works but emits a deprecation warning.

6) Migration/compat considerations
   - Keep `.packed_pose` alias for 1–2 iterations to avoid breaking downstream code; mark as deprecated in docstrings and warn at runtime.
   - Keep `UnifiedSmplInputs.global_orient` as-is to avoid changing forward interfaces; only auto-fill it from `NamedPose.root_pose` when present (non-breaking opt-in).
   - Do not change the semantics of body/hand/face slicers beyond renaming.

Impact Analysis
- Functional impact
  - `NamedPose` now carries optional pelvis rotation; getters by name/index retain the same shapes but now support pelvis directly.
  - Full pose assembly is simpler via `to_full_pose` instead of manual concatenation.
- Risk areas
  - Any code that relied on `.packed_pose` name will need updating or must rely on the deprecation alias.
  - Any code that assumed `get_joint_pose('pelvis')` raises must adapt; tests expecting an exception should be updated.
  - Index-based access must be validated to ensure “index 0 = pelvis” is respected consistently across the toolbox.
- Mitigations
  - Provide a deprecation alias for `.packed_pose` and a clear `DeprecationWarning`.
  - Add targeted tests for pelvis access and index mapping.
  - Keep `UnifiedSmplInputs.global_orient` behavior identical unless `NamedPose.root_pose` is present, in which case auto-fill is a convenience, not a requirement.

Expected Outcome
- Cleaner and less error-prone pose handling with optional root rotation stored in `NamedPose`.
- SMPL-consistent indexing for name- and index-based access.
- Minimal churn for downstream code thanks to a deprecation alias and backward-compatible `UnifiedSmplInputs` behavior.
- Improved interop for optimization and export paths that need full `(B, N+1, 3)` pose tensors.

Example Snippets (Before → After)

1) Accessing pelvis and packed pose
```python
# BEFORE
npz = NamedPose(model_type=ModelType.SMPLX, batch_size=B)
pelvis = npz.pelvis              # (B, 3) zeros only
npz.get_joint_pose('pelvis')     # KeyError
packed = npz.packed_pose         # (B, N, 3) intrinsic only

# AFTER (v2)
npz = NamedPose(model_type=ModelType.SMPLX, batch_size=B)
npz.root_pose = torch.zeros(B, 3)  # optional, stores actual pelvis
pelvis = npz.pelvis                 # (B, 3) from root_pose or zeros
pelvis_pose = npz.get_joint_pose('pelvis')  # (B, 1, 3), valid
intrinsic = npz.intrinsic_pose      # (B, N, 3) intrinsic only
full_pose = npz.to_full_pose()      # (B, N+1, 3) pelvis + intrinsic
```

2) VPoser interop helper probing batch/device/dtype
```python
# BEFORE
B = npz.packed_pose.shape[0]
device = npz.packed_pose.device
dtype = npz.packed_pose.dtype

# AFTER
B = npz.intrinsic_pose.shape[0]
device = npz.intrinsic_pose.device
dtype = npz.intrinsic_pose.dtype
```

3) Index-based access (SMPL indexing)
```python
# BEFORE (no pelvis support)
idx = npz.get_joint_index('left_knee')
left_knee = npz.packed_pose[:, idx:idx+1, :]  # implicit off-by-one logic elsewhere

# AFTER (pelvis at index 0)
idx = 0  # pelvis
pelvis = npz.get_joint_pose_by_index(idx)  # (B, 1, 3)
idx = npz.get_joint_index('left_knee')     # returns SMPL-consistent index
left_knee = npz.get_joint_pose_by_index(idx)
```

References
- Spec for this refactor: `context/tasks/task-anything.md`
- SMPL skeleton indexing: `context/hints/smplx-kb/compare-smpl-skeleton.md`
- VPoser + global orientation guidance: `context/hints/smplx-kb/howto-apply-vposer-with-global-orient.md`
- Core code paths to change:
  - `src/smplx_toolbox/core/containers.py` (NamedPose, UnifiedSmplInputs helpers)
  - `src/smplx_toolbox/vposer/model.py` (NamedPose interop)
  - `src/smplx_toolbox/core/unified_model.py` (examples/imports)
  - Design/docs mentioning `NamedPose.packed_pose` (e.g., fitting helper)
- Reference code (context/refcode): `context/tasks/task-prefix.md` lists:
  - human_body_prior (VPoser v2), smplify-x, and official smplx implementation
- Third-party library IDs (for context7):
  - PyTorch: `/pytorch/pytorch`
  - SMPL-X (official): `/vchoutas/smplx`
