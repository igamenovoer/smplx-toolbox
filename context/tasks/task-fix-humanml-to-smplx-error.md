# Task: Fix HumanML3D → SMPL‑X Conversion Errors

Goal
- Resolve orientation/mapping issues causing “lying on ground” visuals and odd bone poses when converting HumanML3D (T2M) outputs to SMPL‑X `UnifiedSmplInputs`.

Context
- Current converter builds `NamedPose` from T2M 6D and composes root from recovered quaternion/translation.
- Symptoms: Y‑up vs Z‑up mix, potential bone‑axis mismatch, collar/shoulder handling, and order/inversion pitfalls.
- Reference mapping doc: context/refcode/FlowMDM/explain/howto-interpret-flowmdm-output.md

Repro (baseline)
- Generate HML3D: `pixi run flowmdm-gen-humanml` → `tmp/flowmdm-out/humanml3d`
- Convert: `pixi run -e dev python scripts/cvt_flowmdm_humanml_to_smpl_animation.py --input_dir tmp/flowmdm-out/humanml3d`
- View: `pixi run -e dev python scripts/show-animation-unified-model.py --anim-file tmp/flowmdm-out/humanml3d/unified_smpl_animation.pkl --body-model-dir data/body_models --backend basic`

Plan (systematic investigation)
1) Isolate intrinsic rotations
- Zero root (orient/trans) and compare 22‑joint positions vs SMPL‑X body joints.
- Export both: T2M joints (Z‑up) and SMPL‑X joints from retargeted pose only.

2) Orientation study (Y‑up↔Z‑up)
- Validate Rx(±90°) on both root orientation and translation; confirm which sign/order makes pelvis vertical and gait in X/Z plane.
- Report MPJPE (22 joints) before/after orientation fix.

3) Bone‑axis correction prototype
- Implement `apply_bone_axis_correction()` in `src/smplx_toolbox/utils/humanml_mapping.py` (fixed per‑joint Rcorr).
- Start with shoulders/collars/wrists; verify reduction in MPJPE.

4) Decode/mapping parity checks
- Confirm 6D→R decode parity with FlowMDM `cont6d_to_matrix` (no axis flips).
- Re‑verify `humanml_joint_mapping()` ordering and collar handling.

5) Fit vs Retarget consistency
- Create neutral T2M and one frame sample (no root motion). Fit SMPL‑X by keypoints vs retargeted pose; diff should be ≪ 2cm avg.
- Neutral helpers:
  - Python API: `src/smplx_toolbox/utils/humanml_mapping.py:create_neutral_t2m_skeleton()` → returns `T2MSkeleton` (local joints, identity root).
  - From global joints: `T2MSkeleton.from_global_joints(joints_global, root_orient6d, trans)` to obtain local joints for analysis.
  - CLI (optional): `scripts/make_t2m_neutral.py --out tmp/t2m_neutral.pkl` writes a neutral bundle for quick inspection.

Retargeting Strategy (neutral alignment → delta application)
- Observation: The T2M (HumanML3D) local joint frames differ from SMPL‑X. Directly copying per‑joint AA (or 6D) produces incorrect poses.
- Approach:
  1. Fit SMPL‑X to the T2M neutral skeleton by 3D keypoints → obtain `init-smplx-pose` (per‑joint neutral rotations in SMPL‑X space). This captures the bone‑axis/frame alignment implicitly.
  2. For a new T2M pose, compute per‑joint rotation deltas relative to T2M neutral: Δ_T2M_j = R_T2M_pose_j · (R_T2M_neutral_j)^−1.
  3. Map deltas into the SMPL‑X local basis via per‑joint conjugation using the neutral alignment:
     - C_j = R_SMPLX_neutral_j · (R_T2M_neutral_j)^−1 (fixed per joint)
     - Δ_SMPLX_j = C_j · Δ_T2M_j · C_j^−1
  4. Compose the final SMPL‑X pose per joint: R_SMPLX_pose_j = Δ_SMPLX_j · R_SMPLX_neutral_j.
  5. Convert back to AA and assemble `UnifiedSmplInputs` (root handled separately via recovered quaternion + translation with Y‑up→Z‑up fix).
- This ensures T2M deltas are applied in the correct SMPL‑X local frames, eliminating “lying on ground” and bone twisting artifacts.

Deliverables
- Converter flags: `--report-mpjpe` and optional `--export-debug` (dump aligned T2M/SMPLX joints).
- Mapping utils: `apply_bone_axis_correction()` wired behind `--axis-corr`.
- Smoke tests: mapping/orientation unit tests (deterministic synthetic cases).
- Short doc note in roadmap describing final conventions.

Acceptance Criteria
- Upright character (Z‑up) in viewer; pelvis path horizontal.
- MPJPE(T2M vs SMPL‑X 22 joints, no root) < 3cm on simple sequences.
- Fit vs retarget MPJPE < 2cm after corrections on tested frames.

Notes
- Use `pixi run flowmdm-exec -- ...` to run FlowMDM commands inside its workspace.
- Keep changes focused; no unrelated refactors.
