# Refactor Plan: Replace `PoseByKeypoints` with `NamedPose`

## Objective

- Repurpose the current joint-wise pose container into a lightweight utility for interpreting packed poses.
- Introduce `NamedPose` as a small helper around a packed axis–angle pose tensor `(B, N, 3)` with knowledge of the SMPL variant and joint indices.
- Remove `PoseByKeypoints` from the function input surface; callers should pass packed poses directly. `NamedPose` is for optional inspection/editing only.

## Current State

- `PoseByKeypoints` (in `src/smplx_toolbox/core/containers.py`) stores many optional tensors (one per joint) and provides converters to packed pose vectors for SMPL/SMPL-H/SMPL-X.
- Several call sites, including `UnifiedSmplInputs.from_keypoint_pose`, accept `PoseByKeypoints` and convert it internally to packed segments.
- This leads to duplication (per-joint assembly) and encourages passing a heavy, optional-attribute container through APIs.

## Proposed Design

### `NamedPose` (utility only)

- Class style:
  - Implemented as an `attrs` class.
  - Required init parameter: `smpl_type: str` (one of `{smpl, smplh, smplx}`).

- Data members:
  - `smpl_type: str` — required.
  - `packed_pose: torch.Tensor` — shape `(B, N, 3)` (axis–angle per joint), contiguous.
  - Optional convenience: `batch_size: int = 1` — used to allocate `packed_pose` when not supplied.

- Core behavior:
  - Interpret `packed_pose` with a fixed joint namespace and index mapping based on `smpl_type`.
  - Provide safe getters/setters by joint name that do not create autograd edges or modify gradients.
  - Unsupported joints for the given `smpl_type` return `None` (get) / no-op with `False` (set).

- Public API (minimal):
  - `get_joint_pose(name: str) -> torch.Tensor | None`
    - Returns a detached tensor of shape `(B, 1, 3)` (copy) if supported; otherwise `None`.
  - `set_joint_pose(name: str, pose: torch.Tensor) -> bool`
    - Accepts `(B, 1, 3)` (or `(B, 3)` broadcasted) and assigns under `torch.no_grad()` into `packed_pose` at the appropriate index; returns `True` if set, `False` if unsupported.
  - `to_dict() -> dict[str, torch.Tensor]`
    - Returns a mapping from joint names to tensor views of shape `(B, 1, 3)` that slice `packed_pose` in-place. Modifying these tensors will modify `NamedPose.packed_pose` directly.
  - `get_joint_index(name: str) -> int`
    - Returns the integer index (0-based within the second dimension of `packed_pose`) for the joint `name`; raises `KeyError` if the name is not valid for the current `smpl_type` (aliases like `left_eyeball` map to the corresponding eye index).
  - `get_joint_indices(names: list[str]) -> list[int]`
    - Vectorized variant returning indices for each name; raises `KeyError` if any name is invalid for the current `smpl_type`.
  - `get_joint_name(index: int) -> str`
    - Inverse mapping: returns the canonical joint name for a given index; raises `IndexError` if out of range for the current `smpl_type`.
  - `get_joint_names(indices: list[int]) -> list[str]`
    - Vectorized variant returning names for each index; raises `IndexError` on any invalid index.

- Joint namespace and counts (N):
  - SMPL: `N=22` — `root` + 21 body joints.
  - SMPL-H: `N=52` — SMPL body + 15 left hand + 15 right hand.
  - SMPL-X: `N=55` — SMPL body + `jaw`, `left_eye`, `right_eye` + both hands.
  - Eye aliases: `left_eyeball -> left_eye`, `right_eyeball -> right_eye`.

- Validation:
  - Init behavior:
    - If `packed_pose` is provided, validate `packed_pose.ndim == 3`, last dim is 3, and second dim equals the expected `N` for `smpl_type`.
    - If `packed_pose` is not provided, allocate a zeros tensor of shape `(batch_size, N, 3)` on CPU with `float32`.
  - Type-check inputs in setters; raise `ValueError` for wrong shapes, return `False` for unsupported joints.

- Gradient-safety:
  - `get_joint_pose` returns a `.detach().clone()` to avoid creating edges.
  - `set_joint_pose` performs in-place copy under `torch.no_grad()` to avoid affecting the graph.
  - `to_dict` returns views; in-place edits will mutate `packed_pose` and thus the NamedPose instance. Prefer `set_joint_pose` when gradient-safety is required.

### API Surface Changes Elsewhere

- `UnifiedSmplInputs`:
  - Deprecate and remove `from_keypoint_pose(PoseByKeypoints, ...)`.
  - Accept only segmented tensors or a packed pose (if we decide to add a `from_packed_pose` helper). Prefer keeping `UnifiedSmplInputs` independent of any named wrapper.

- Remove `PoseByKeypoints` usage from all public forward paths. Keep it temporarily with a deprecation warning for internal tests that still reference it, then delete.

## Migration Plan (Step-by-step)

1. Introduce `NamedPose` alongside existing code
   - Add class to `src/smplx_toolbox/core/containers.py` with the API above.
   - Init: require `smpl_type`, optionally accept `packed_pose`; if missing, create zeros `(batch_size, N, 3)`.
   - Implement joint-index maps for all `smpl_type` variants from existing ordered lists used in conversions.
   - Add alias handling for eye joints.

2. Mark `PoseByKeypoints` as deprecated
   - Add `warnings.warn("PoseByKeypoints is deprecated; use packed poses + NamedPose for inspection", DeprecationWarning, stacklevel=2)` in its constructor or converters.
   - Update docstrings to point users to `NamedPose`.

3. Refactor call sites to stop accepting `PoseByKeypoints`
   - `UnifiedSmplInputs.from_keypoint_pose`: deprecate then remove.
   - Update any tests, examples, and internal helpers to pass packed poses directly.

4. Add unit tests for `NamedPose`
   - For each `smpl_type`, construct `packed_pose` with identifiable per-joint values, then:
     - `get_joint_pose(name)` returns expected `(B, 1, 3)` values and `None` for unsupported joints.
     - `set_joint_pose(name, new_value)` updates only that joint; gradients remain intact for unrelated parts.
     - `to_dict()` returns views: modifying `d[name][...]` changes `named_pose.packed_pose` accordingly.
     - Name/index mapping round-trip: `get_joint_indices(get_joint_names(range(N)))` equals `list(range(N))`; and for several specific names, `get_joint_name(get_joint_index(name)) == canonical_name`.
   - Validate shape errors and unsupported-joint behavior.

5. Update documentation and examples
   - Replace references to `PoseByKeypoints` with packed pose usage patterns and optional `NamedPose` inspection.

6. Remove `PoseByKeypoints`
   - After tests and call sites are migrated, delete the class and its references.

## Joint Index Mapping (source of truth)

- Reuse the ordered joint lists currently hard-coded in `PoseByKeypoints.to_smpl_pose/to_smplh_pose/to_smplx_pose` to build name→index dicts:
  - SMPL order: `root` + [left_hip, right_hip, spine1, left_knee, right_knee, spine2, left_ankle, right_ankle, spine3, left_foot, right_foot, neck, left_collar, right_collar, head, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist]
  - SMPL-H adds 15 left-hand joints then 15 right-hand joints (same names as current code).
  - SMPL-X inserts face before hands: `jaw`, `left_eye`, `right_eye`, then the hands.

## Coding Standards

- Strong typing with precise shapes in docstrings; prefer `torch.Tensor` with `(B, N, 3)` spelled out.
- Absolute imports within `smplx_toolbox`.
- No `__main__` guards; no side effects on import.
- Keep implementation small and focused — this is a utility, not an input container.

## Risks and Mitigations

- Risk: Hidden dependencies on `PoseByKeypoints` in tests/examples.
  - Mitigation: Grep and update; provide temporary deprecation path.
- Risk: Shape mismatches for different SMPL variants.
  - Mitigation: Strict validation in `NamedPose.__init__`; comprehensive unit tests per variant.
- Risk: Accidental autograd side effects when setting values.
  - Mitigation: Use `torch.no_grad()` + in-place `.copy_`; return detached clones on get.

## Acceptance Criteria

- `NamedPose` implemented as an attrs class with correct joint mapping per SMPL variant; getters/setters behave as specified.
- All public APIs accept packed poses directly; no usage of `PoseByKeypoints` in forward paths.
- Unit tests cover `NamedPose` get/set, `to_dict` view semantics, and unsupported-joint behavior across variants.
- Documentation/examples no longer reference `PoseByKeypoints` for inputs.

## Timeline (indicative)

- Day 1: Implement `NamedPose`, add tests, add deprecation warnings to `PoseByKeypoints`.
- Day 2: Update call sites and tests to packed poses; remove `from_keypoint_pose`.
- Day 3: Delete `PoseByKeypoints`, finalize docs.

## Open Questions

- Do we want an optional `from_flattened(packed_flat: (B, N*3))` helper? (Not strictly necessary.)
- Should `set_joint_pose` accept `(B, 3)` and reshape internally? (Plan: yes, with clear error messages.)
- Should `batch_size` be explicit when `packed_pose` is omitted, or infer from another field? (Plan: add `batch_size: int = 1`.)
- Any additional joint aliases beyond eye names required for downstream compatibility?
