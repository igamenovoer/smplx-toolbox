# Refactor Plan: Replace `PoseByKeypoints` with `NamedPose`

## Objective

- Repurpose the current joint-wise pose container into a lightweight utility for interpreting packed poses.
- Introduce `NamedPose` as a small helper around a packed axis–angle pose tensor `(B, N, 3)` with knowledge of the SMPL variant and joint indices.
- Remove `PoseByKeypoints` from the function input surface; callers should pass packed poses directly. `NamedPose` is for optional inspection/editing only.

## Current State

- `PoseByKeypoints` (in `src/smplx_toolbox/core/containers.py`) stores many optional tensors (one per joint) and provides converters to packed pose vectors for SMPL/SMPL-H/SMPL-X.
- Several call sites, including `UnifiedSmplInputs.from_keypoint_pose`, accept `PoseByKeypoints` and convert it internally to packed segments.
- This leads to duplication (per-joint assembly) and encourages passing a heavy, optional-attribute container through APIs.

## Proposed Design (Updated)

### `NamedPose` (utility only)

- Class style:
  - Implemented as an `attrs` class.
  - Required init parameter: `model_type: ModelType` (see `src/smplx_toolbox/core/constants.py`).

- Data members:
  - `model_type: ModelType` — required model kind (`SMPL`, `SMPLH`, `SMPLX`).
- `packed_pose: torch.Tensor` — shape `(B, N, 3)` (axis–angle per joint), contiguous. Intrinsic joints ONLY — pelvis (global orientation) is excluded.
  - Optional convenience: `batch_size: int = 1` — used to allocate `packed_pose` when not supplied.

- Core behavior:
  - Interpret `packed_pose` with a fixed joint namespace and index mapping based on `model_type`, EXCLUDING the pelvis.
  - Provide safe getters/setters by joint name that do not create autograd edges or modify gradients.
  - Unsupported joints for the given `model_type` return `None` on getters; setters raise a `KeyError`.

- Public API (minimal):
  - `get_joint_pose(name: str) -> torch.Tensor | None`
    - Returns a detached tensor of shape `(B, 1, 3)` (copy) if supported; otherwise `None`.
  - `set_joint_pose(name: str, pose: torch.Tensor) -> bool`
    - Accepts `(B, 1, 3)` or `(B, 3)` and reshapes `(B, 3) -> (B, 1, 3)` automatically. Assigns under `torch.no_grad()` into `packed_pose` at the appropriate index; returns `True` if set; raises `KeyError` if the joint name is not valid for the current `model_type`; raises `ValueError` for wrong shapes.
  - `to_dict(with_pelvis: bool = False) -> dict[str, torch.Tensor]`
    - Returns a mapping from intrinsic joint names to tensor views of shape `(B, 1, 3)`. If `with_pelvis=True`, includes `'pelvis'` mapped to zeros `(B, 1, 3)`.
  - `get_joint_index(name: str) -> int | None`
    - Returns the integer index (0-based within the second dimension of `packed_pose`) for the joint `name`; returns `None` if the name is not valid for the current `model_type`.
  - `get_joint_indices(names: list[str]) -> list[int | None]`
    - Vectorized variant returning indices for each name; returns `None` for any name that is invalid for the current `model_type`.
  - `get_joint_name(index: int) -> str`
    - Inverse mapping: returns the canonical joint name for a given index; raises `IndexError` if out of range for the current `model_type`.
  - `get_joint_names(indices: list[int]) -> list[str]`
    - Vectorized variant returning names for each index; raises `IndexError` on any invalid index.
  - `pelvis` property
    - Returns zero AA `(B, 3)` for convenience when building full poses for LBS. Getters/setters for `'pelvis'` raise `KeyError` to signal it's not part of intrinsic pose.

- Joint namespace and counts (N) — intrinsic only (no pelvis):
  - SMPL: `N=21` — 21 body joints (pelvis excluded).
  - SMPL-H: `N=51` — SMPL (21) + 15 left hand + 15 right hand.
  - SMPL-X: `N=54` — SMPL (21) + `jaw`, `left_eye_smplhf`, `right_eye_smplhf` + both hands.

- Validation:
  - Init behavior:
    - If `packed_pose` is provided, validate `packed_pose.ndim == 3`, last dim is 3, and second dim equals the expected `N` for `model_type`.
    - If `packed_pose` is not provided, allocate a zeros tensor of shape `(batch_size, N, 3)` on CPU with `float32`.
  - Type-check inputs in setters; raise `ValueError` for wrong shapes; raise `KeyError` for unsupported joints.

- Gradient-safety:
  - `get_joint_pose` returns a `.detach().clone()` to avoid creating edges.
  - `set_joint_pose` performs in-place copy under `torch.no_grad()` to avoid affecting the graph.
  - `to_dict` returns views; in-place edits will mutate `packed_pose` and thus the NamedPose instance. Prefer `set_joint_pose` when gradient-safety is required.

### API Surface Changes Elsewhere

- `UnifiedSmplInputs`:
  - Add `global_orient: (B, 3)` as a standalone field to pass to SMPL/SMPL‑X.
  - `named_pose` contains intrinsic pose only; conversion helpers use `global_orient` + `named_pose` to form model kwargs.
  - Deprecate and remove `from_keypoint_pose(PoseByKeypoints, ...)`.

- Remove `PoseByKeypoints` usage from all public forward paths. Keep it temporarily with a deprecation warning for internal tests that still reference it, then delete.

## Migration Plan (Step-by-step)

1. Introduce `NamedPose` alongside existing code
   - Add class to `src/smplx_toolbox/core/containers.py` with the API above.
   - Init: require `model_type: ModelType`, optionally accept `packed_pose`; if missing, create zeros `(batch_size, N, 3)`.
   - Use `ModelType` and mappings from `core.constants` to resolve joint indices; do not rely on legacy lists.
   - Add alias handling for eye joints.

2. Mark `PoseByKeypoints` as deprecated
   - Add `warnings.warn("PoseByKeypoints is deprecated; use packed poses + NamedPose for inspection", DeprecationWarning, stacklevel=2)` in its constructor or converters.
   - Update docstrings to point users to `NamedPose`.

3. Refactor call sites to stop accepting `PoseByKeypoints`
   - `UnifiedSmplInputs.from_keypoint_pose`: deprecate then remove.
   - Update any tests, examples, and internal helpers to pass packed poses directly.

4. Add unit tests for `NamedPose`
   - For each `model_type`, construct `packed_pose` with identifiable per-joint values, then:
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

- Use the explicit enums and mappings in `src/smplx_toolbox/core/constants.py`:
  - `ModelType` — model kind selector (`SMPL`, `SMPLH`, `SMPLX`).
  - `get_joint_index(joint_name: str, model_type: ModelType | str) -> int` — authoritative name→index.
  - `SMPL_JOINT_NAME_TO_INDEX`, `SMPLH_JOINT_NAME_TO_INDEX`, `SMPLX_JOINT_NAME_TO_INDEX` — explicit ordered mappings.
  - `ModelType.get_joint_names()` — returns ordered names per model type.
  - This removes reliance on any legacy hard-coded lists in `PoseByKeypoints`.

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
- `NamedPose` uses `ModelType` (from `core.constants`) for model selection and constants’ mappings for joint indices.

## Timeline (indicative)

- Day 1: Implement `NamedPose`, add tests, add deprecation warnings to `PoseByKeypoints`.
- Day 2: Update call sites and tests to packed poses; remove `from_keypoint_pose`.
- Day 3: Delete `PoseByKeypoints`, finalize docs.

## Decisions

- No `from_flattened(...)` helper. Users set `packed_pose` explicitly and are responsible for correct shape `(B, N, 3)` and dtype/device.
- `set_joint_pose` accepts `(B, 3)` and reshapes to `(B, 1, 3)` automatically. This behavior must be noted in the docstring.
- `batch_size` default of `1` is kept. Provide `repeat(n: int)` to expand the batch to `B * n` in-place for convenience.
- Getters that take a joint name return `None` if the name is not recognized for the current `model_type`. Setters raise a `KeyError` for unknown names.
