# Code Review: Unified SMPL Family Model implementation

Time: 2025-09-08 00:00:00
Target: `src/smplx_toolbox/core/unified_model.py`
Related Spec: `context/tasks/features/unified-smpl-model/task-unified-smpl.md`
Method: Knowledge-based review with spec cross-check (no code edits)

## Summary
The file defines skeletons for UnifiedSmplModel, UnifiedSmplInputs, PoseByKeypoints, and UnifiedSmplOutput consistent in naming with the spec. However, most methods are stubs (incomplete `pass`-like regions) and many behaviors required by the spec are not implemented yet. The current state reads like a scaffold awaiting implementation, not a working adapter.

Overall consistency with the spec is partial: type/class names align and signatures mostly match the latest spec (notably `UnifiedSmplInputs.from_keypoint_pose`). But large gaps remain in validation, normalization, pose composition, joint unification, properties, and detection. Several spec changes (factory restrictions, properties on UnifiedSmplOutput, warnings) are not yet reflected.

## Findings by section

### UnifiedSmplInputs
- PROS:
  - Fields match spec (root_orient, pose_body, hands, jaw, eyes, betas, expression, trans, v_template, joints_override, v_shaped).
  - `from_keypoint_pose(kpts, model_type)` exists, matching the latest spec direction.
  - `hand_pose` and `eyes_pose` properties declared.
- GAPS/ISSUES:
  - `hand_pose`, `eyes_pose`, `from_kwargs`, `batch_size`, and `check_valid` bodies are empty.
  - `check_valid` must enforce per-model rules (SMPL/SMPL-H/SMPL-X) and shape checks.
  - `from_keypoint_pose` must implement the exact ordering defined in spec and handle alias mapping (left_eyeball/right_eyeball).

### PoseByKeypoints
- PROS:
  - Contains comprehensive joint fields, aligned with spec including aliases `left_eyeball/right_eyeball`.
  - `from_kwargs`, `batch_size`, and `check_valid_by_keypoints` placeholders present.
- GAPS/ISSUES:
  - Body trunk list includes spine/hip/ankle etc, which matches earlier spec; ensure final spec’s simplified section in the summarized attachment is reconciled (the full spec lists all 21 body joints; keep that order in conversion logic).
  - Methods are unimplemented; must perform shape checks and strict/non-strict behavior per model type.

### UnifiedSmplOutput
- PROS:
  - Matches spec attributes and types.
- GAPS/ISSUES:
  - Property methods (`num_vertices`, `num_joints`, `num_faces`, `batch_size`, and convenience slices for body/hand/face joints) are declared but empty.

### UnifiedSmplModel
- PROS:
  - Constructor and `from_smpl_model` signature include `missing_joint_fill` and `warn_fn`, consistent with spec’s simplified factory (no override of model metadata).
  - `_normalize_inputs`, `_unify_joints`, `_compute_full_pose`, `forward`, `to`, `eval`, `train`, `get_joint_names`, `_get_raw_joint_count`, `select_joints` are structured per spec.
- GAPS/ISSUES:
  - `from_smpl_model` body is empty; should store refs and config, and validate wrapped instance.
  - `_detect_model_type` has the right heuristics, but is unimplemented.
  - `model_type`, `num_betas`, `num_expressions`, `dtype`, `device`, `faces` properties are unimplemented; spec requires dynamic access to underlying model buffers/params.
  - `_normalize_inputs` must:
    - Convert `PoseByKeypoints` using `UnifiedSmplInputs.from_keypoint_pose`.
    - Validate via `check_valid`.
    - Zero-fill missing segments according to `model_type`.
    - Build kwargs in the exact structure the wrapped model expects.
  - `_unify_joints` must map raw to unified SMPL-X joint set; on SMPL-H/SMPL, fill missing face joints with NaN/zero per config, and return mapping in extras.
  - `_compute_full_pose` must concatenate according to model type (P=156/165 etc).
  - `forward` must handle wrapped output structure (names may vary) and include `v_shaped` optionally.
  - `to`, `eval`, `train` should delegate to wrapped model for mode (spec says proxy eval/train) and move only adapter tensors on `to`.
  - `get_joint_names` should return unified vs raw names; `_get_raw_joint_count` should reflect model type.
  - `select_joints` must implement indices/names mapping and error cases.

## Spec consistency checklist
- Accept `PoseByKeypoints` in `forward` and convert via `UnifiedSmplInputs.from_keypoint_pose`: PARTIAL (structure present, not implemented).
- Factory `from_smpl_model(deformable_model, *, missing_joint_fill, warn_fn)`: PARTIAL (signature OK, body missing; ensure no metadata overrides).
- Dynamic properties read from wrapped model: MISSING.
- Input validation and normalization per model type: MISSING.
- Joint set unification to SMPL-X with placeholders for missing: MISSING.
- Full pose composition (P sizes): MISSING.
- Output container properties: MISSING.
- Aliases for eye fields honored: MISSING (to be in from_keypoint_pose).
- Warnings via `warn_fn` for ignored extra inputs: MISSING.

## Suggestions (no code changes)
1. Implement UnifiedSmplInputs methods first (small surface area, unlocks validation):
   - `batch_size()`, `hand_pose`, `eyes_pose`, `from_kwargs`, `check_valid()` per spec rules.
   - `from_keypoint_pose()` with exact ordering (21-body + 15-finger orders) and alias handling.
2. Implement PoseByKeypoints validation and batch sizing; add a tiny mapping table listing body and finger orders to drive composition.
3. Fill UnifiedSmplOutput computed properties; provide safe defaults and assert shapes for clarity.
4. Complete UnifiedSmplModel dynamic properties by probing wrapped model buffers: `v_template.dtype`, `faces_tensor`, `shapedirs`, `betas`, `expression`, and first param/buffer device.
5. `_detect_model_type()` with the spec heuristics and type name lowercasing.
6. `_normalize_inputs()` path:
   - If PoseByKeypoints → `from_keypoint_pose` → `check_valid` → kwargs for wrapped model.
   - Ensure zeros for missing segments and move to correct device/dtype.
7. `_compute_full_pose()` concatenation based on `model_type` (root/body/face/eyes/hands).
8. `_unify_joints()`:
   - SMPL-X: pass-through mapping to unified (also return raw mapping).
   - SMPL-H: map body + hands; inject face placeholders per `m_missing_joint_fill`, and record `missing_joints` in extras.
   - Consider caching a mapping tensor on first use.
9. `forward()`:
   - Be robust to different model outputs (e.g., namedtuple vs. dataclass) and field names (`vertices`/`verts`, `joints` present?).
   - Attach `v_shaped` to extras if available and spec flag is later added.
10. `eval()`/`train()` should proxy directly to the wrapped model, returning self for chaining.
11. `get_joint_names()`:
   - Unified names from `context/refcode/smplx/smplx/joint_names.py` (SMPL-X list) vs raw names per type; ensure exposed names match unification target ordering.
12. Add inline docstrings to methods indicating shape contracts and error modes; this will ease implementing tests later.

## Potential pitfalls / edge cases
- Batch size inference when only betas or expression provided.
- Mismatched dtype/device across provided tensors; normalize to wrapped model device/dtype.
- Partial hand specification; zero-fill missing fingers, optionally warn.
- SMPL models with different joint regressors; ensure deterministic mapping.
- `faces_tensor` sometimes on CPU; do not move faces unnecessarily.

## References
- SMPL-X repo (Context7): /vchoutas/smplx — body models and joints, pose layout and joint naming.
- Local reference: `context/refcode/smplx/smplx/joint_names.py` — JOINT_NAMES, SMPLH_JOINT_NAMES, SMPL_JOINT_NAMES.

## Conclusion
The implementation file is a good scaffold but currently incomplete relative to the spec. Prioritize input container and conversion utilities, then normalization and pose composition, followed by joint unification and dynamic properties. Once these are in place, add unit tests outlined in the spec to guide further refinements.
