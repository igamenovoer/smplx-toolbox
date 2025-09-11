## Unified SMPL Family Model — Design Guide (Spec Only)

This guide specifies a single Python class that provides a unified API to load and use SMPL-H and SMPL-X models (and leaves space for SMPL). Do NOT implement yet—this is the design and acceptance criteria for the upcoming change.

References
- Env: see `context/tasks/task-prefix.md` (pixi usage, no ad-hoc installs)
- Baseline: `context/refcode/human_body_prior/src/human_body_prior/body_model/body_model.py` ("BodyModel")

### Goals
- Offer one class, e.g., `UnifiedSmplModel`, that abstracts model differences (SMPL-H vs SMPL-X) while exposing a consistent interface for posing and retrieving outputs.
- Accept a preloaded deformable model instance created by the caller (e.g., `deformable_model = smplx.create(...)`). No file loading in this class.
- Auto-detect model type from the provided instance when possible; allow explicit override.
- Normalize inputs (pose, betas, expressions) and outputs (verts, joints, faces) across models.
- Be drop-in friendly for current tests and utilities in this repo.

### Non-Goals
- Re-implement LBS or low-level math. Reuse existing `lbs` and registered buffers where possible.
- Change public data formats of the underlying models on disk.

---

## High-level API

Class: `UnifiedSmplModel`

Constructor signature (spec):
- `__init__(self) -> None` — no-arg constructor per Python Coding Guide. All configuration is done via the factory `from_smpl_model(...)`.

Key methods:
- `forward(inputs: UnifiedSmplInputs | PoseByKeypoints) -> UnifiedSmplOutput` — calls the wrapped `deformable_model` with normalized inputs and returns unified outputs. When given `PoseByKeypoints`, it is first converted to `UnifiedSmplInputs` (see “Pose by explicit keypoints”).
- `to(device) -> UnifiedSmplModel` — move only auxiliary buffers maintained by this adapter; users must handle `deformable_model.to(device)` separately (documented)
- `eval() / train()` — standard nn.Module behavior; proxies to `deformable_model`
- `as_smplx_like()` — returns a view/adapter making SMPL-H outputs shaped like SMPL-X where feasible (e.g., joint sets) — see Joint Unification below

Factory construction:
- `from_smpl_model(deformable_model: nn.Module, *, missing_joint_fill: Literal["nan", "zero"] = "nan", warn_fn: Optional[Callable[[str], None]] = None) -> UnifiedSmplModel`
	- Classmethod factory. Creates an empty instance via `cls()`, sets internal state (member variables prefixed with `m_`), and returns it.
	- All model properties (model_type, num_betas, num_expressions, dtype, device) are dynamically read from `deformable_model` - no overrides allowed
	- Returns a configured adapter instance

Dynamic properties (computed on-the-fly from deformable_model; not stored):
- `@property model_type -> str`
- `@property num_betas -> int`
- `@property num_expressions -> int | 0`
- `@property dtype -> torch.dtype` (adapter’s dtype; the wrapped model can have its own)
- `@property device -> torch.device` (best-effort device of key parameters of the wrapped model)

Implementation hints for dynamic properties (from `context/refcode/smplx`):
- Model type detection
	- Prefer `type(m).__name__.lower()` in {"smpl", "smplh", "smplx"}
	- Heuristics: `hasattr(m, "jaw_pose")` (or `leye_pose/reye_pose`) → "smplx"; `hasattr(m, "left_hand_pose") and hasattr(m, "right_hand_pose")` → "smplh"; else "smpl".
- num_betas
	- Try `m.num_betas` (SMPL classes expose this), else `m.shapedirs.shape[-1]`, else `m.betas.shape[-1]`.
- num_expressions
	- If SMPL-X and `hasattr(m, "expression")`, use `m.expression.shape[-1]`; otherwise 0.
- dtype
	- Use a float buffer like `m.v_template.dtype` (present in body_models) or fallback to `m.shapedirs.dtype`.
- device
	- From first available param or buffer: `next(m.parameters(), None)` or `next(m.buffers(), None)`, then read `.device`.
- faces
	- `m.faces_tensor` (registered long buffer in body_models).

References:
- `context/refcode/smplx/smplx/body_models.py` — buffers/params names: `shapedirs`, `faces_tensor`, `betas`, `global_orient`, `body_pose`, `left_hand_pose`, `right_hand_pose`, `jaw_pose`, `transl`, `v_template`.
- `context/refcode/smplx/smplx/joint_names.py` — `JOINT_NAMES` (SMPL-X), `SMPLH_JOINT_NAMES`, `SMPL_JOINT_NAMES`.

Single Source of Truth
- Do not cache or store any state that exists on `deformable_model` (e.g., model_type, faces, num_betas, expression size, dtype/device). Always compute via dynamic properties.
- No override parameters in factory method - all model properties must be read directly from the wrapped model instance to maintain consistency
- Caching is allowed only for adapter-owned artifacts that are not present on `deformable_model` (e.g., unified joint index mapping). If used, keep it lazy-initialized and easy to invalidate.

Member variables (minimal; initialized to None in `__init__`):
- `m_deformable_model: Optional[nn.Module]` — the wrapped instance (single source of truth)
- `m_missing_joint_fill: Optional[str]`
- Optional: small auxiliary mapping tensors for joint unification (not mirrored model state)

Dataclass/typed container: `UnifiedSmplOutput` (using `@define(kw_only=True)` from attrs)
- `vertices: Tensor` — (B, V, 3) mesh vertices
- `faces: Tensor` — (F, 3) face connectivity
- `joints: Tensor` — (B, J, 3) unified joint set; see below
- `full_pose: Tensor` — (B, P) flattened axis-angle pose actually used for LBS
- `extras: Dict[str, Any] = field(factory=dict)` — model-specific extras (e.g., raw joint set, masks)

Properties (computed from tensors):
- `@property num_vertices -> int`
- `@property num_joints -> int`
- `@property num_faces -> int`
- `@property batch_size -> int`
- `@property body_joints -> Tensor` — first 22 joints
- `@property hand_joints -> Tensor` — 30 hand joints (15 per hand)
- `@property face_joints -> Tensor` — 3 face joints (jaw + 2 eyes)

### Inputs to forward (attrs structure)

Use an attrs-based container for inputs: `UnifiedSmplInputs`.

Spec (using `@define(kw_only=True)` from attrs for keyword-only initialization):
- `root_orient: Tensor | None` — (B, 3)
- `pose_body: Tensor | None` — (B, 63)
- `left_hand_pose: Tensor | None` — (B, 45)
- `right_hand_pose: Tensor | None` — (B, 45)
- `pose_jaw: Tensor | None` — (B, 3)
- `left_eye_pose: Tensor | None` — (B, 3)
- `right_eye_pose: Tensor | None` — (B, 3)
- `betas: Tensor | None` — (B, nb)
- `expression: Tensor | None` — (B, ne)
- `trans: Tensor | None` — (B, 3)
- `v_template: Tensor | None` — (B, V, 3) when supported; else ignored
- `joints_override: Tensor | None` — (B, J*, 3) if supported; else ignored
- `v_shaped: Tensor | None` — (B, V, 3) if supported; else ignored

On-the-fly computed properties (read-only):
- `hand_pose: Tensor | None` — concatenation of left/right hand poses → (B, 90) when both present; else None
- `eyes_pose: Tensor | None` — concatenation of left/right eye poses → (B, 6) when both present; else None

Helpers on the container:
- `@classmethod from_kwargs(**kwargs) -> UnifiedSmplInputs` — convenience constructor
- `batch_size()` — infers batch size from first non-None field

Validation:
- `check_valid(model_type: Literal["smpl", "smplh", "smplx"]) -> None`
	- Verifies tensor presence and shapes consistent with the model type; raises `ValueError` on failure.
	- Rules:
		- Common: if provided, `root_orient`=(B,3), `pose_body`=(B,63), `betas`=(B,nb), `trans`=(B,3)
		- SMPL: disallow hand/eye/jaw/expression inputs (must be None); hands optional but if provided must be None (error if not None)
		- SMPL-H: require both `left_hand_pose` and `right_hand_pose` present with shape (B,45); `pose_jaw`/eyes/expression must be None
		- SMPL-X: require both hands (B,45) and both eyes (B,3) if any of them is provided; `pose_jaw` if provided must be (B,3); `expression` if provided must match model’s `num_expressions` when known
		- If `v_template`, `joints_override`, or `v_shaped` are provided, check shapes and ignore if unsupported by the wrapped model

---

## Pose by explicit keypoints (attrs)

Add an attrs container that specifies per-joint axis-angle pose by name. This is a user-friendly alternative to opaque flattened pose vectors.

Container: `PoseByKeypoints` (using `@define(kw_only=True)` from attrs)
- Shapes: every joint field, when present, is `(B, 3)` axis-angle in radians in the model’s kinematic frame; `None` means “not specified → use zeros”.
- This container targets the SMPL-X jointed pose DoFs (root/body/jaw/eyes/hands). It can be down-converted to SMPL-H (drop face) and to SMPL (drop hands+face).

Fields (SMPL-X superset):
- Root and body trunk
	- `root` (aka pelvis/global_orient)
	- `left_hip`, `right_hip`, `spine1`, `left_knee`, `right_knee`, `spine2`,
	- `left_ankle`, `right_ankle`, `spine3`, `left_foot`, `right_foot`,
	- `neck`, `left_collar`, `right_collar`, `head`,
	- `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`
- Face/eyes (SMPL-X only DoFs)
	- `jaw`
	- `left_eye` (eyeball), `right_eye` (eyeball)
- Left hand (15 finger joints × 3 DoF)
	- `left_thumb1`, `left_thumb2`, `left_thumb3`
	- `left_index1`, `left_index2`, `left_index3`
	- `left_middle1`, `left_middle2`, `left_middle3`
	- `left_ring1`, `left_ring2`, `left_ring3`
	- `left_pinky1`, `left_pinky2`, `left_pinky3`
- Right hand (mirrors left)
	- `right_thumb1`, `right_thumb2`, `right_thumb3`
	- `right_index1`, `right_index2`, `right_index3`
	- `right_middle1`, `right_middle2`, `right_middle3`
	- `right_ring1`, `right_ring2`, `right_ring3`
	- `right_pinky1`, `right_pinky2`, `right_pinky3`

Helpers and behaviors:
- `@classmethod from_kwargs(**kwargs) -> PoseByKeypoints` — convenience ctor.
- `batch_size()` — infer B from the first non-None tensor.
- Use `UnifiedSmplInputs.from_keypoint_pose(kpts: PoseByKeypoints, *, model_type: Literal["smpl","smplh","smplx"]) -> UnifiedSmplInputs` to convert the per-joint fields to the segmented inputs:
	- `root_orient` ← `root`
	- `pose_body` ← concatenation of the 21 body joints in SMPL/SMPL-H/SMPL-X order:
		- `[left_hip, right_hip, spine1, left_knee, right_knee, spine2, left_ankle, right_ankle, spine3, left_foot, right_foot, neck, left_collar, right_collar, head, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist]` → shape `(B, 63)`
	- `pose_jaw` ← `jaw` (SMPL-X only)
	- `left_eye_pose` ← `left_eye`; `right_eye_pose` ← `right_eye` (SMPL-X only)
	- `left_hand_pose` ← concat of left hand fingers in order `[thumb1, thumb2, thumb3, index1, index2, index3, middle1, middle2, middle3, ring1, ring2, ring3, pinky1, pinky2, pinky3]` → `(B, 45)`
	- `right_hand_pose` ← same order for right hand → `(B, 45)`
	- Missing/None joints are filled with zeros in their segment. Extra fields not used by the model type are ignored; see strict rules below.
	 - Aliases: accept `left_eyeball` ≡ `left_eye` and `right_eyeball` ≡ `right_eye` for user clarity; both map to SMPL-X `leye_pose`/`reye_pose`.

Validation on `PoseByKeypoints`:
- `check_valid_by_keypoints(model_type: Literal["smpl","smplh","smplx"], strict: bool = False) -> None`
	- Common: if a field is provided it must be `(B,3)`.
	- SMPL: must NOT provide face or hand joint fields when `strict=True`; if `strict=False`, they are ignored with a warning hook.
	- SMPL-H: must NOT provide face/eye joints when `strict=True`; if hands are partially provided, missing finger joints are zero-filled but a warning is emitted.
	- SMPL-X: accepts any subset; unspecified joints default to zero. If eyes are partially specified (one eye only), the missing eye defaults to zero and a warning is emitted.

Notes and sources:
- SMPL-X segmented pose layout: `3 (root) + 63 (21 body) + 3 (jaw) + 3 (leye) + 3 (reye) + 45 (left hand) + 45 (right hand) = 165`.
- The 21 body joint order above follows the official models (see `context/refcode/smplx/smplx/joint_names.py`).
- For hands, the per-finger order matches the SMPL/SMPL-H/MANO convention used by the official implementations.

---

## Auto-detection and model metadata

Detection sources (from the instance):
- Inspect `type(deformable_model).__name__` and known attributes of official `smplx` models to infer type (e.g., presence of face/jaw/eye components)
- Fallback: try probing expected parameter names or buffers (e.g., hand pose dimensionality) to determine `model_type`

Stored metadata (properties cached by adapter):
- `model_type: str` — "smplh" | "smplx" | "smpl"
- `n_verts: int`, `n_faces: int`, `n_betas: int`, `n_expr: int | 0` (if derivable)
- `joint_names_raw: List[str] | None` (if the model exposes them)
- `faces: Tensor[F, 3]` (from `deformable_model.faces_tensor` or equivalent)
- Any additional info required for joint unification

---

## Joint set unification

Unified joint set target: SMPL-X 55-joint scheme (body + hands + face) where possible.

- For SMPL-X (official smplx models): use the model’s standard joint outputs; expose as-is and map to unified names.
- For SMPL-H: produce a best-effort mapping to the SMPL-X joint set:
	- Body: direct map
	- Hands: direct map (SMPL-H has hands)
	- Face: not present — fill with NaNs or zeros (configurable via `missing_joint_fill`), and mark in `extras["missing_joints"]`.

Provide utilities:
- `get_joint_names(unified: bool = True) -> List[str]`
- `select_joints(indices | names)` — returns a view of joints

Edge cases:
- If the source regressor lacks exact joints, document the approximation and ensure index alignment is deterministic.

---

## Pose vector normalization

Compose `full_pose` internally based on `model_type`:
- SMPL-H: `[root(3), body(63), hands(90)]` → P=156
- SMPL-X: `[root(3), body(63), jaw(3), eyes(6), hands(90)]` → P=165

Normalization rules:
- Missing segments are auto-filled with zeros of the expected size for that model type.
- Extra segments supplied for a model that doesn’t use them are ignored with a warning (optionally raise if `strict=True`).
- Hand pose input accepted as:
	- concatenated `B x 90`
	- tuple `(left: B x 45, right: B x 45)`
	- dict `{"left": Bx45, "right": Bx45}`

From `PoseByKeypoints`:
- The adapter composes the segments by concatenating the corresponding `(B,3)` joint tensors in the specified orders. Any missing joint field is zero-filled.
- Conversion is performed before calling model-specific forward; the resulting `UnifiedSmplInputs` are then validated with `check_valid()`.

---

## Output contract

Forward returns `UnifiedSmplOutput` with:
- `vertices (B, V, 3)` and `faces (F, 3)`
- `joints (B, J_u, 3)` — unified set
- `full_pose (B, P)` — pose used by LBS
- `extras` may include:
	- `joints_raw (B, J_raw, 3)`
	- `joint_names_raw`
	- `joint_mapping (Dict[raw_idx -> unified_idx])`
	- `v_shaped (B, V, 3)` when requested
	- `warnings: List[str]`

Error modes:
- Shape mismatches surface as `ValueError` with clear messages including expected shapes by model type.
- Unsupported file formats raise `ValueError`.

---

## Configuration knobs

- `missing_joint_fill: Literal["nan", "zero"] = "nan"`
- `return_v_shaped: bool = False`
- `warn_fn: Callable[[str], None] | None` — hook for warning logging

---

## Implementation sketch (no code yet)

1) Accept the preloaded `deformable_model`
- Store reference and infer `model_type` from attributes; validate available capabilities (hands, face, expressions)

2) Minimal auxiliary buffers
- Keep only what the adapter needs for normalization and joint unification (e.g., unified joint names, mapping indices, missing masks)

3) Normalization helpers
- `_normalize_inputs(...)` prepares `full_pose` and all optional tensors per model type.
- `_compose_full_pose_*` functions for each model type to concatenate pose parts.
 - These helpers operate on `UnifiedSmplInputs` and return a normalized dict of arguments expected by the underlying `deformable_model` (e.g., `global_orient`, `body_pose`, `left_hand_pose`, `right_hand_pose`, `jaw_pose`, `leye_pose`, `reye_pose`, `betas`, `expression`, `transl`). Forward calls `inputs.check_valid(model_type)` before normalization.
 - When `inputs` is a `PoseByKeypoints`, first call `inputs.check_valid_by_keypoints(model_type, strict=False)` and then convert with `UnifiedSmplInputs.from_keypoint_pose(inputs, model_type=model_type)`.

4) Forward call
- Normalize/compose inputs into the parameter structure the `deformable_model` expects (e.g., SMPL-X expects separate jaw/eyes/hands; SMPL-H expects hands only)
- Call `deformable_model(**normalized_inputs)` to obtain vertices and joints (and faces from the instance)
- Apply translation if not already handled by the model

5) Joint unification
- Convert raw joints to unified SMPL-X-like set; inject placeholders for missing joints on SMPL-H.
- Package `extras` accordingly.

6) Device handling
- Document that users must move `deformable_model` themselves (`deformable_model.to(device)`)
- The adapter’s `.to()` moves only its own auxiliary tensors

---

## Acceptance criteria

- API surface matches this spec; names and shapes as listed above.
- Auto-detection works for provided `deformable_model` instances created via `smplx.create(...)` for SMPL-H and SMPL-X.
- For SMPL-H input with hand poses only, forward succeeds and returns unified joints with face joints filled as configured.
- For SMPL-X input, expressions are supported when `num_expressions` set, and jaw/eye poses are properly included.
 - Input validation errors are raised when shapes/presence are inconsistent with model type.
 - Unit tests cover: shape checks, detection, hand pose/eye variants via left/right fields, missing joint fill, input validation, and that dynamic properties reflect changes on `deformable_model` (no stale cached state).
 - New: `PoseByKeypoints` works end-to-end. For SMPL-X, providing only a subset (e.g., only elbow and jaw) produces valid outputs with other joints zeroed. For SMPL-H/SMPL, face/eye keypoints are ignored (warning) or rejected in strict mode.
 - New: The joint ordering used by `UnifiedSmplInputs.from_keypoint_pose()` matches the official body/hand ordering; the assembled segments have exact shapes `(B,63)`, `(B,3)`, `(B,3)`, `(B,3)`, `(B,45)`.

---

## Testing plan (phase 1)

New tests under `tests/`:
- `test_unified_model_detection.py`
	- creates models via `smplx.create(model_type=...)`; asserts detected `model_type` from instances
- `test_unified_model_forward_shapes.py`
	- checks output shapes for both models (verts, faces, joints, full_pose)
- `test_unified_model_hand_inputs.py`
	- provides left/right hands; asserts `hand_pose` computed property shape and equivalence to concatenation
- `test_unified_model_eye_inputs.py`
	- provides left/right eyes; asserts `eyes_pose` computed property shape and equivalence to concatenation
- `test_unified_model_input_container.py`
	- constructs `UnifiedSmplInputs`; verifies `check_valid()` rules for smpl/smplh/smplx
- `test_unified_model_missing_face_joints.py`
	- SMPL-H forward with `missing_joint_fill = "nan"` and `"zero"`
- `test_unified_model_strict_mode.py`
	- ensures irrelevant inputs raise

Additional tests for keypoint-based inputs:
- `test_pose_by_keypoints_smplex_minimal.py`
	- Build `PoseByKeypoints` with only `root`, `left_elbow`, `right_elbow`, `jaw`; convert to inputs for SMPL-X; assert segment shapes and that only provided joints are non-zero.
- `test_pose_by_keypoints_eye_alias.py`
	- Provide `left_eyeball`/`right_eyeball` only; assert they map to `left_eye_pose`/`right_eye_pose` and compose eyes block `(B,6)` correctly.
- `test_pose_by_keypoints_smplh_subset.py`
	- Provide several hand finger joints and body joints; convert to SMPL-H; assert hands/body compose correctly; face joints are ignored or rejected in strict mode.
- `test_pose_by_keypoints_smpl_strict.py`
	- Provide any hand/face joints; with `strict=True` validation raises; with `strict=False` they are ignored and zeros used.
- `test_pose_by_keypoints_ordering.py`
	- Verify that the 21-body joint order and 15-finger joint order compose the expected `(B,63)` and `(B,45)` blocks by comparing against manual concatenation.

Smoke scripts under `tmp/` (optional):
- `tmp/check_unified_forward.py` — tiny runner using pixi to validate model load and forward once.

---

## Migration notes for repo code

- Add `src/smplx_toolbox/core/unified_model.py` with `UnifiedSmplModel` (implementation phase)
- Create lightweight adapter/wrapper to mimic `BodyModel` outputs if needed by existing callers
- Update visualization/util scripts to call unified class where appropriate

---

## Run instructions (for tests once implemented)

Use pixi environment and python inline (from `task-prefix.md`):

```powershell
pixi run -e dev python -c 'print("Unified model smoke test pending implementation")'
```

Once tests are added:
```powershell
pixi run -e dev python -m pytest -q
```

---

## Open questions / assumptions

- Assumption: `.npz` body model files exist for both SMPL-H and SMPL-X in `data/body_models/**` and follow the same structure as `BodyModel` expects.
- Assumption: Unifying to SMPL-X joint set is acceptable; face joints missing on SMPL-H are filled with configured placeholder.
- Question: Should we expose an option to emit the raw/native joint set alongside unified by default? Current spec puts it in `extras`.
- Question: Should we support `.pkl` models from the official `smplx` repo directly in v1, or defer to a later phase?
## Note: PoseByKeypoints Status (Outdated)

- PoseByKeypoints has been removed from the implementation and replaced by a simpler flow.
- Current flow: pass segmented AA tensors via `UnifiedSmplInputs`; use `NamedPose` only for inspecting/editing packed `(B, N, 3)` poses by joint name.
- Joint namespaces and mappings are defined by `ModelType` and constants in `src/smplx_toolbox/core/constants.py`.
- This document remains as an archived design reference; code paths referencing PoseByKeypoints no longer exist.

