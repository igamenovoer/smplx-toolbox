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

Retargeting Strategy (bind calibration → world‑space delta)
- Goal: Given two skeletons matched/characterized in a reference state, drive the target (SMPL‑X) from the source (T2M) by transferring per‑joint motion while accounting for different local joint frames and proportions.
- References (industry‑standard patterns):
  - SideFX KineFX: “Match rest poses, map joints, transfer motion” (FBIK) — emphasizes conforming rest/bind poses before transfer and handling offsets during solve.
  - NVIDIA Omniverse: “Retarget uses rest transforms; adjust retarget pose if the defaults differ; root joint recommended.”
  - Wicked Engine: practical algorithm mapping animation differences in world space, then converting back to target local space.
  - Khronos (forum): derive target joint transforms by equating source and target in a common reference pose and walking the hierarchy; warns that local‑only deltas fail when local frames differ.

Recommended algorithm (rotations‑only; robust to local‑axis mismatches)
1) Calibrate in a shared reference (bind/neutral)
   - Source bind: G_src0_j = world transform of T2M neutral/bind for joint j.
   - Target bind: G_tgt0_j = world transform of SMPL‑X neutral/bind for joint j (obtained by fitting SMPL‑X to the neutral T2M skeleton).
   - Precompute per joint:
     - invG_src0_j = inverse(G_src0_j)
     - Optionally, cache joint parent indices for FK/topo order.

2) For each animation frame t
   - Build source world transforms G_src_j(t) by FK from T2M local rotations (6D→R) and offsets. Include the root orientation/translation in the FK after applying the Y‑up→Z‑up fix to the root.
   - Compute the world‑space rotation delta (left‑multiply convention):
       δ^W_j(t) = R_src^W_j(t) · (R_src^W_j(0))^{-1}
     Equivalently, using world matrices: δ^W_j(t) = G_src_j(t) · (G_src0_j)^{-1} with translation cleared.
   - Apply the same delta to the target bind world rotation and recover target local:
       R_tgt^W_j(t) = δ^W_j(t) · R_tgt^W_j(0)
       R_tgt^L_j(t) = (R_tgt^W_parent(j)(t))^{-1} · R_tgt^W_j(t)  (parent handled in topo order)
     Convert R_tgt^L_j(t) to axis–angle for the `NamedPose`.
   - Assemble `NamedPose` for SMPL‑X from these local rotations (axis‑angle), keeping pelvis/root as `global_orient` + `transl` handled separately (see Orientation below).

Why this works
- It transfers motion as a difference from the source bind pose in world space (what you see on screen), then expresses that motion in the target’s coordinate frames. Under left‑multiply conventions, this is exactly:
    R_tgt^W_j(t) = R_src^W_j(t) · (R_src^W_j(0))^{-1} · R_tgt^W_j(0)
  Equivalently, if you prefer a fixed relative rotation per joint:
    M_j = (R_src^W_j(0))^{-1} · R_tgt^W_j(0)  (constant)
    R_tgt^W_j(t) = R_src^W_j(t) · M_j
  Both forms are equivalent and avoid assuming identical local axes.

Implementation notes (toolbox)
- Source FK: use the self‑contained T2M FK in `smplx_toolbox.utils.humanml_mapping` to compute G_src_j(t) from 6D + offsets.
- Bind caches: cache G_src0_j, G_tgt0_j and parents once per sequence (or once per rig) for speed.
- Rotations only: zero the translation component when extracting L_tgt_j(t) for pose; translation is handled at the root.
- Numerical stability: use orthonormalized rotation matrices from 6D before FK; re‑orthonormalize after multiplications if needed.

Alternative (local delta with change‑of‑basis)
- When you prefer local‑space math, compute fixed change‑of‑basis per joint from binds and conjugate the local deltas:
  - Δ_src^L_j(t) = R_src^L_j(t) · (R_src^L_j(0))^{-1}
  - C_j = R_tgt^L_j(0) · (R_src^L_j(0))^{-1}
  - R_tgt^L_j(t) = C_j · Δ_src^L_j(t) · R_tgt^L_j(0)
- This becomes equivalent to the world‑space method when bind transforms are consistent and parents are handled in topo order; the world‑space form is typically more forgiving.

Orientation and Up‑Axis
- HumanML3D/FlowMDM uses Y‑up for root motion; our SMPL‑X viewer/pipeline is Z‑up. Apply a fixed X‑axis rotation at the root to map Y‑up→Z‑up consistently to both `global_orient` and `transl` before FK on the source side, or equivalently post‑compose on the target side. Validate the sign by ensuring pelvis is upright and forward motion lies in the ground plane.

Scale and Units (T2M vs SMPL‑X)
- T2M skeleton is unit‑normalized: average bone length across its 21 edges is exactly 1.0 (unitless) in the neutral helper.
- SMPL‑X is in meters: for neutral shape (betas=0), the 21 core body bone lengths have avg ≈ 0.2004 m (min ≈ 0.0605, max ≈ 0.4115).
- Implications:
  - Rotations are unit‑free; the world‑rotation delta method remains valid.
  - Translations/root trajectories and MPJPE comparisons must be on a common unit scale.
  - Recommended: compute a global scale `s` from the bind alignment and apply it to T2M translations (or equivalently to T2M joints for error reporting):
    - Simple ratio (edges): `s = median_j ( ||e_tgt0_j||_2 / ||e_src0_j||_2 )` where `e_*` are corresponding bone vectors.
    - Or Procrustes‑style scalar fit on centered joint sets (without rotation): `s = trace(A^T B) / trace(A^T A)` with `A = P_src0 - c_src`, `B = P_tgt0 - c_tgt`.
  - During FK of the source, scale the root translation: `T_root(src, meters) = s · T_root(src, t2m_units)` so world positions are comparable.
  - Allow SMPL‑X shape (betas) to deform during the initial fit to reduce residual limb length mismatch; `s` handles gross unit conversion, betas handle morphology.

Sketch (per frame, per joint j)
- Input: source local 6D rotations R_src^L_j(t), root (R_root_src(t), T_root_src(t)) in Y‑up; bind caches R_src^W_j(0), R_tgt^W_j(0); joint parent indices.
- Steps:
  - R_root = Rx(+/−90°) · R_root_src(t)
  - T_root = s · Rx(+/−90°) · T_root_src(t)  (apply global scale `s` to root translation)
  - Run FK to get R_src^W_j(t)
  - δ^W_j = R_src^W_j(t) · (R_src^W_j(0))^{-1}
  - R_tgt^W_j(t) = δ^W_j · R_tgt^W_j(0)
  - R_tgt^L_j(t) = (R_tgt^W_parent(j)(t))^{-1} · R_tgt^W_j(t)
  - Convert R_tgt^L_j(t) → axis‑angle → `NamedPose`

Cross‑checks
- Root‑only test: with identity local rotations, only root motion should move the target. Character must be upright in Z‑up.
- Bind‑pose test: when R_src_local(t) == bind, then R_tgt_local(t) should equal target bind.
- Round‑trip sanity: small random deltas at elbows/knees should map without twisting or collapsing collars.

Pointers to references
- SideFX KineFX retargeting workflow (rest pose match, map, transfer): https://www.sidefx.com/docs/houdini/character/kinefx/retargeting.html
- NVIDIA Omniverse retargeting (rest transforms, retarget pose, root joint recommendation): https://docs.omniverse.nvidia.com/extensions/latest/ext_animation-retargeting.html
- Wicked Engine retargeting blog (world‑space delta mapping with bind/inverse‑parent matrices and decomposition): https://wickedengine.net/2022/09/animation-retargeting/
- Khronos forum thread (derive target transforms via global equality; caution on local‑only deltas): https://community.khronos.org/t/skeleton-animation-retargeting-orientation-only-same-skeleton/75872

Deliverables
- Converter flags: `--report-mpjpe` and optional `--export-debug` (dump aligned T2M/SMPLX joints).
- Mapping utils: `apply_bone_axis_correction()` wired behind `--axis-corr`.
- Add `--scale` (float) and `--auto-scale` options:
  - When `--auto-scale`, compute `s` from bind alignment (median edge ratio or Procrustes) and log it; default to applying it to source root translations and evaluation joints.
- Smoke tests: mapping/orientation unit tests (deterministic synthetic cases).
- Short doc note in roadmap describing final conventions.

Acceptance Criteria
- Upright character (Z‑up) in viewer; pelvis path horizontal.
- MPJPE(T2M vs SMPL‑X 22 joints, no root) < 3cm on simple sequences.
- Fit vs retarget MPJPE < 2cm after corrections on tested frames.

Notes
- Use `pixi run flowmdm-exec -- ...` to run FlowMDM commands inside its workspace.
- Keep changes focused; no unrelated refactors.
