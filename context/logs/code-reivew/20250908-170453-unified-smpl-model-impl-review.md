# Code Review Report — Unified SMPL Family Model Implementation

Target file: [src/smplx_toolbox/core/unified_model.py](src/smplx_toolbox/core/unified_model.py)
Spec: [context/tasks/features/unified-smpl-model/task-unified-smpl.md](context/tasks/features/unified-smpl-model/task-unified-smpl.md)
Review method: [.magic-context/instructions/review-code-by-mem.md](.magic-context/instructions/review-code-by-mem.md)

Executive summary

- Overall alignment: High-level API matches spec; several validation and compatibility gaps remain.
- Risk level: Medium due to potential forward signature incompatibility with official smplx and missing validations.
- Priority fixes: remove unsupported kwargs in forward normalization; add shape checks for betas/expression; add partial-hand/eye warnings; replace placeholder joint names; define deterministic joint mapping for SMPL and SMPL-H.

Compliance matrix (spec vs. implementation)

- Factory and constructor: Implemented as classmethod [UnifiedSmplModel.from_smpl_model()](src/smplx_toolbox/core/unified_model.py:468) returning configured adapter. __init__ is no-arg, as required.
- Dynamic properties: model_type [UnifiedSmplModel.model_type](src/smplx_toolbox/core/unified_model.py:512), num_betas [UnifiedSmplModel.num_betas](src/smplx_toolbox/core/unified_model.py:517), num_expressions [UnifiedSmplModel.num_expressions](src/smplx_toolbox/core/unified_model.py:529), dtype [UnifiedSmplModel.dtype](src/smplx_toolbox/core/unified_model.py:541), device [UnifiedSmplModel.device](src/smplx_toolbox/core/unified_model.py:551), faces [UnifiedSmplModel.faces](src/smplx_toolbox/core/unified_model.py:569) implemented per spec heuristics.
- Inputs container: [UnifiedSmplInputs](src/smplx_toolbox/core/unified_model.py:38) and validation [UnifiedSmplInputs.check_valid()](src/smplx_toolbox/core/unified_model.py:94) exist; see gaps below.
- Keypoint container and conversion: [PoseByKeypoints](src/smplx_toolbox/core/unified_model.py:269) and [UnifiedSmplInputs.from_keypoint_pose()](src/smplx_toolbox/core/unified_model.py:172) implemented; body and hand joint order matches spec; eye aliases supported.
- Forward normalization: [_normalize_inputs()](src/smplx_toolbox/core/unified_model.py:582), [_compute_full_pose()](src/smplx_toolbox/core/unified_model.py:705) align with spec ordering; see kwargs compatibility risk below.
- Joint unification: [_unify_joints()](src/smplx_toolbox/core/unified_model.py:647) produces SMPL-X-like 55 joints; SMPL-H fills face placeholders; SMPL fills missing; see mapping notes below.
- Output container: [UnifiedSmplOutput](src/smplx_toolbox/core/unified_model.py:400) with properties for counts and body/hand/face segments; OK.
- Utilities: [UnifiedSmplModel.get_joint_names()](src/smplx_toolbox/core/unified_model.py:828) and [UnifiedSmplModel.select_joints()](src/smplx_toolbox/core/unified_model.py:858) present; unified names currently placeholders.

Key issues and gaps

1) Forward kwargs incompatibility with official smplx models
   - The normalizer injects normalized["return_joints"] = True in [_normalize_inputs()](src/smplx_toolbox/core/unified_model.py:641-644). Official implementations do not accept a return_joints kwarg:
     - SMPL.forward signature shows return_verts and return_full_pose only [SMPL.forward](context/refcode/smplx/smplx/body_models.py:315-325).
     - SMPLH.forward is similar [SMPLH.forward](context/refcode/smplx/smplx/body_models.py:696-708).
     - SMPLX.forward supports return_verts, return_full_pose, return_shaped [SMPLX.forward](context/refcode/smplx/smplx/body_models.py:1122-1139).
   - Passing return_joints will raise TypeError on official models.

2) Missing shape validations in inputs
   - betas: [UnifiedSmplInputs.check_valid()](src/smplx_toolbox/core/unified_model.py:94-171) does not verify betas shape against model.num_betas (spec section Inputs Validation).
   - expression: For SMPL-X, provided expression is not checked against [UnifiedSmplModel.num_expressions](src/smplx_toolbox/core/unified_model.py:529). [_normalize_inputs()](src/smplx_toolbox/core/unified_model.py:636-640) creates zeros if missing but does not validate provided tensors.

3) Partial-hands/partial-eyes warnings not implemented per spec
   - [PoseByKeypoints.check_valid_by_keypoints()](src/smplx_toolbox/core/unified_model.py:355-398) handles basic shapes and ignores face on SMPL-H, but:
     - Does not warn when only some hand finger joints are provided on SMPL-H.
     - Does not warn when only one eye is provided on SMPL-X; conversion currently allows None which zero-fills but no warning.

4) SMPL joint mapping is naïve and may misalign semantics
   - [_unify_joints() smpl path](src/smplx_toolbox/core/unified_model.py:686-703) copies the first min(J, 23) raw joints into unified indices 0..num_body−1. Official SMPL returns 23 body joints via regressor, then vertex_joint_selector may append dataset landmarks before mapping. A deterministic mapping using joint names is preferable.
   - The spec calls for a documented, deterministic mapping and to record missing indices in extras.

5) Unified joint names are placeholders
   - [get_joint_names(unified=True)](src/smplx_toolbox/core/unified_model.py:837-841) returns joint_i. Spec references official joint name lists in [joint_names.py](context/refcode/smplx/smplx/joint_names.py:19) and expects unified names to be exposed.

6) Extras completeness
   - Only joints_raw and optional v_shaped are added in forward [UnifiedSmplModel.forward()](src/smplx_toolbox/core/unified_model.py:783-793). Spec suggests including joint_names_raw and joint_mapping; also warnings list if any.

7) faces dtype/device hygiene
   - [faces property](src/smplx_toolbox/core/unified_model.py:569-580) returns faces_tensor if present, else converts numpy to torch.long. Ensure faces is torch.long and, if needed by downstream, on CPU; current behavior is acceptable but should be documented.

8) select_joints by names depends on placeholder names
   - [select_joints()](src/smplx_toolbox/core/unified_model.py:858-893) will not work reliably until unified names reflect [JOINT_NAMES](context/refcode/smplx/smplx/joint_names.py:19).

9) Minor documentation clarity
   - [UnifiedSmplModel.to()](src/smplx_toolbox/core/unified_model.py:795-815) warns that only adapter tensors are moved; consider clarifying that users must call deformable_model.to(device).

What is working well

- Model type detection heuristics align with spec [UnifiedSmplModel._detect_model_type()](src/smplx_toolbox/core/unified_model.py:492-511).
- full_pose composition matches spec ordering [UnifiedSmplModel._compute_full_pose()](src/smplx_toolbox/core/unified_model.py:705-741).
- Keypoint-to-segment conversion uses the exact 21 body joint order and 15-finger order; eye aliases supported [UnifiedSmplInputs.from_keypoint_pose()](src/smplx_toolbox/core/unified_model.py:205-265).
- Output container properties for body/hand/face slices match shapes [UnifiedSmplOutput.body_joints](src/smplx_toolbox/core/unified_model.py:431-434), [hand_joints](src/smplx_toolbox/core/unified_model.py:436-439), [face_joints](src/smplx_toolbox/core/unified_model.py:441-444).

Risk assessment

- High: return_joints kwarg will break with official smplx models (TypeError).
- Medium: Silent acceptance of wrong betas/expression shapes can produce subtle geometry errors.
- Low-Medium: Joint misalignment for SMPL could affect evaluation metrics or downstream selectors if indices shift.

Recommendations (prioritized)

P0 — Forward compatibility
- Remove return_joints from [_normalize_inputs()](src/smplx_toolbox/core/unified_model.py:641-644). Official models already return joints; no kwarg needed.

P0 — Input shape validations
- In [UnifiedSmplInputs.check_valid()](src/smplx_toolbox/core/unified_model.py:94-171):
  - If betas provided, enforce betas.shape[1] == adapter.num_betas; raise ValueError otherwise.
  - If expression provided (SMPL-X), enforce expression.shape[1] == adapter.num_expressions; raise ValueError.

P1 — Keypoint validation behaviors
- In [PoseByKeypoints.check_valid_by_keypoints()](src/smplx_toolbox/core/unified_model.py:355-398):
  - SMPL-H: if any hand joint provided, check that both hands have the same set; warn and zero-fill missing fingers; strict=True raises.
  - SMPL-X: if only one eye provided, warn and zero-fill the other.

P1 — Deterministic joint mapping and metadata
- Construct raw→unified index mapping using joint name lists in [joint_names.py](context/refcode/smplx/smplx/joint_names.py:168-242, context/refcode/smplx/smplx/joint_names.py:244-269) and SMPL-X [JOINT_NAMES](context/refcode/smplx/smplx/joint_names.py:19-65).
- Add extras["joint_mapping"] and extras["joint_names_raw"] (if available from model).

P1 — Unified names
- Replace placeholder unified names in [get_joint_names(unified=True)](src/smplx_toolbox/core/unified_model.py:837-841) with official SMPL-X names from [JOINT_NAMES](context/refcode/smplx/smplx/joint_names.py:19-65).

P2 — faces and docs
- Ensure faces is torch.long; document device expectations. Minor doc clarifications in [to()](src/smplx_toolbox/core/unified_model.py:795-815).

Suggested tests to add or adjust

- test_unified_model_forward_signature_compat: Call UnifiedSmplModel with official smplx models and assert no TypeError on forward (i.e., no unexpected kwargs).
- test_betas_expression_validation: Provide mismatched betas/expression sizes and assert ValueError (SMPL-H/SMPL-X).
- test_pose_by_keypoints_partial_warnings: Verify warnings emitted for partial hands (SMPL-H) and single-eye (SMPL-X), and zero-filling behavior.
- test_joint_mapping_extras: Assert extras contains joint_mapping and that missing_joints indices are correct for SMPL-H and SMPL.
- test_joint_names_unified: Assert get_joint_names(unified=True) equals the first 55 elements of [JOINT_NAMES](context/refcode/smplx/smplx/joint_names.py:19-65).

Migration and compatibility notes

- No change to public data formats of underlying models; adapter-only changes.
- Downstream callers relying on placeholder joint names may see name changes; this is desirable for alignment with spec/tests.

References

- Official model forwards: [SMPL.forward](context/refcode/smplx/smplx/body_models.py:315-399), [SMPLH.forward](context/refcode/smplx/smplx/body_models.py:696-762), [SMPLX.forward](context/refcode/smplx/smplx/body_models.py:1122-1302).
- Joint names: [context/refcode/smplx/smplx/joint_names.py](context/refcode/smplx/smplx/joint_names.py).
- Current implementation: [UnifiedSmplModel._normalize_inputs()](src/smplx_toolbox/core/unified_model.py:582-645), [UnifiedSmplModel._unify_joints()](src/smplx_toolbox/core/unified_model.py:647-704), [UnifiedSmplInputs.check_valid()](src/smplx_toolbox/core/unified_model.py:94-171), [PoseByKeypoints.check_valid_by_keypoints()](src/smplx_toolbox/core/unified_model.py:355-398).

Appendix — Mapping strategy outline (SMPL/SMPL-H to unified SMPL-X indices)

- Use Body helper concept in [joint_names.Body](context/refcode/smplx/smplx/joint_names.py:272-320) as a reference for name-based projections.
- Build a raw_names list from model if available; else assume canonical order from official implementations.
- For SMPL-H: identity map for body and hands, set face indices 52:55 to NaN/0 per missing_joint_fill; record extras["missing_joints"] = [52,53,54].
- For SMPL: map first 22 body joints plus pelvis/head etc. according to [SMPL_JOINT_NAMES](context/refcode/smplx/smplx/joint_names.py:244-269); mark all hand and face indices missing.