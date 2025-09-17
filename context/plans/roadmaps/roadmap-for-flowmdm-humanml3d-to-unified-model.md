# Roadmap for Implementing FlowMDM HumanML3D → Unified Model (SMPL‑X) Pipeline

## Current Implementing Feature
Build an end‑to‑end path to transform FlowMDM HumanML3D generation outputs into `UnifiedSmplInputs` consumable by `src/smplx_toolbox/core/unified_model.py`.

This includes: exporting richer artifacts from FlowMDM for HumanML3D (root transforms, de‑normalized features, joint rotations, metadata), a converter that reads those artifacts, and an optimization/retargeting step to recover SMPL/SMPL‑X parameters per frame.

### Tasks
- [x] `runners/generate-ex.py`: export extended HumanML3D artifacts (`results_ext.npy`, root transforms, de‑norm feats)
- [x] `save_hml_extended(sample_denorm)`: save `r_quat`, `r_pos`, `cont6d_params`, `joints_world`, `meta` (integrated in generator)
- [x] `scripts/cvt_flowmdm_humanml_to_smpl_animation.py`: convert extended outputs → `UnifiedSmplInputs`
- [ ] `humanml_joint_mapping()`: map T2M 22 joints ↔ SMPL‑X unified joints (subset)
- [ ] `fit_hml_joints_to_smplx()`: sequence fitting (global_orient, body_pose, transl[, betas])
- [x] `pixi task`: add converter task (`flowmdm-cvt-humanml`); extended export is implicit during gen
- [ ] `docs`: author “HumanML3D conversion & fitting” guide and update hints
- [ ] `tests`: small smoke tests on synthetic/short sequences

Next Steps
- Manual validation on fresh generations (end‑to‑end run + pickle inspection)
- Implement visualization integration (open converted pickle in unified viewer)
- Add mapping utilities and optional bone‑axis corrections
- Implement/iterate fitting pipeline (IK/optimization) with priors
- Write docs and add smoke tests

### feat: humanml3d-to-unified-model-bridge

## Overview
HumanML3D output from FlowMDM is 3D keypoints and short MP4s. Unlike Babel, it lacks direct SMPL/SMPL‑X parameters. To use our unified model, we need either a direct retargeting of HumanML3D’s internal joint rotations to SMPL‑X or a robust fitting pipeline from 3D joints to SMPL‑X parameters. This roadmap implements a robust path that first persists essential HumanML3D internals (root yaw/translation, de‑normalized 263‑D features, continuous‑6D joint rotations, foot contacts, and world‑space joints), and then converts them into `UnifiedSmplInputs` via a mapping + fitting/IK procedure.

## Features
- `export-extended-artifacts-humanml3d`: Persist complete decode‑time info from FlowMDM for HumanML3D.
- `converter-humanml3d-to-unifiedsmplinputs`: Read extended artifacts and construct `UnifiedSmplInputs` per frame.
- `fitting-pipeline-humanml3d-3d-joints-to-smplx`: Optimize SMPL‑X params against HumanML3D joints with priors.
- `joint-mapping-and-retargeting`: Utilities to map T2M joints to SMPL‑X joint set and build reasonable initial poses.
- `viewer-and-validation`: Visual checks and quantitative error summaries.
- `cli-integration`: Pixi tasks and scripts for generation and conversion.
- `documentation-and-tests`: Docs and smoke tests.

## Breakdown of Features

### feat: export-extended-artifacts-humanml3d

Add a HumanML3D “extended” save path in FlowMDM so that conversion does not re‑implement internals.

#### Requirements
- `save de‑normalized features`: 263‑D per‑frame features post `X = X * std + mean` used in decode.
- `save root transform`: `r_quat (w,x,y,z)` and `r_pos (x,y,z)` from `recover_root_rot_pos()`.
- `save joint rotations`: continuous‑6D joint rotations for joints 1..21; include root as 6D (yaw) for completeness.
- `save world joints`: `[22, 3, T]` joints already written to `results.npy` must be duplicated into the extended bundle with metadata for direct consumption.
- `save meta`: fps=20, text, lengths, stats source, coordinate system notes, kinematic chain id.

#### Tasks
- [ ] `runners/generate-ex.py`: add a HumanML3D branch to persist `results_ext.npy` alongside `results.npy`.
- [ ] `save_hml_extended(sample_denorm)`: helper that computes and saves:
  - `feats_denorm` (T,263)
  - `r_quat` (T,4), `r_pos` (T,3)
  - `cont6d_params` (T,22,6) including root (convert root yaw to 6D)
  - `joints_world` (T,22,3) copied from decode
  - `foot_contacts` (T,4) if available
  - `meta.json` (fps, text, lengths, mean/std paths, code versions)

### feat: converter-humanml3d-to-unifiedsmplinputs

Convert extended HumanML3D outputs into `UnifiedSmplInputs` for `UnifiedSmplModel`.

#### Requirements
- `read extended bundle`: accept directory with `results_ext.npy` + `meta.json`.
- `fallback`: if only `results.npy` is present, reconstruct minimal info (positions + length + text) and warn about missing root quaternion.
- `output`: pickle a `list[UnifiedSmplInputs]` with `named_pose`, `global_orient`, `trans` per frame; optionally `betas` and hands if estimated.

#### Tasks
- [x] `scripts/cvt_flowmdm_humanml_to_smpl_animation.py`: implement converter CLI mirroring the Babel script style.
- [ ] `--method (fit|retarget)`, `--init (root_only|retargeted|identity)` options.
- [x] `UnifiedSmplInputs`: fill `global_orient` and `trans` from `r_quat`, `r_pos`; fill `named_pose` from retargeted body pose.

### feat: joint-mapping-and-retargeting

Provide utilities to map the Text2Motion (T2M) 22‑joint skeleton to the SMPL‑X body joint set and build reasonable initial poses.

#### Requirements
- `mapping`: static mapping table T2M→SMPL‑X for core body joints; document any offsets or missing joints.
- `retarget init`: convert T2M 6D rotations to axis‑angle per joint in SMPL‑X order; note that bone axes differ, so this is a heuristic initializer, not final output.

#### Tasks
- [ ] `humanml_joint_mapping()`: define mapping and selection order matching `CoreBodyJoint` used by the toolbox.
- [ ] `cont6d_to_axis_angle_t2m()`: decode 6D to AA per T2M joint, then remap to SMPL‑X order.
- [ ] `apply_bone_axis_correction()`: optional fixed transforms to better align T2M local frames to SMPL‑X.

### feat: fitting-pipeline-humanml3d-3d-joints-to-smplx

Fit SMPL‑X parameters to HumanML3D 3D joints using our optimization builders and priors.

#### Requirements
- `objective`: L2/robust 3D joint distance between model joints and T2M joints (mapped subset).
- `priors`: angle prior, VPoser body prior; optional temporal smoothing; optional foot contact penalties.
- `initialization`: root from `r_quat/r_pos`; body from retargeted 6D or zeros.
- `outputs`: per‑frame `UnifiedSmplInputs` with `global_orient`, `body_pose` (21×3), and `trans`; optionally `betas` if optimized.

#### Tasks
- [ ] `fit_hml_joints_to_smplx(joints, init, priors, cfg)`: core optimization routine (sequence or per‑frame).
- [ ] Integrate toolbox priors: `src/smplx_toolbox/optimization/*` builders.
- [ ] `artifact dump`: save a small `.pkl` with target joints, init, optimized params for quick inspection via `scripts/show-keypoint-match.py`.

### feat: viewer-and-validation

Visual sanity checks and numeric summaries to validate the pipeline.

#### Requirements
- `viewer`: visualize converted animation with `scripts/show-animation-unified-model.py`.
- `metrics`: report average joint error vs T2M joints; check pelvis path & yaw alignment.

#### Tasks
- [ ] Add converter flag `--viz` to open the unified viewer after writing the pickle.
- [ ] Compute/report mean per‑joint position error (MPJPE) against T2M joints.

### feat: cli-integration

Ease‑of‑use wrappers with Pixi tasks.

#### Requirements
- `generation`: `flowmdm-gen-humanml-extended` to save the extended artifacts.
- `conversion`: `flowmdm-cvt-humanml` to run the converter and write the unified pickle.

#### Tasks
- [x] `pyproject.toml`: add Pixi converter task (`flowmdm-cvt-humanml`).
- [ ] Shortcuts: `flowmdm-show-humanml-anim` to open the viewer on the converted pickle.

### feat: documentation-and-tests

Author docs and add minimal tests to prevent regressions.

#### Requirements
- `docs`: describe artifacts, mapping, fitting options, and expected outputs.
- `tests`: smoke test on a short sequence; verify pickle loads and first/last frames are finite and sensible.

#### Tasks
- [ ] `docs/`: “HumanML3D → SMPL‑X conversion” page; update `context/hints/` cross‑links.
- [ ] `tests/`: add a focused test that runs the converter on a tiny sample and asserts output shapes and basic error bounds.

---

References
- Target API: `src/smplx_toolbox/core/unified_model.py`
- Babel example: `scripts/cvt_flowmdm_babel_to_smpl_animation.py`
- FlowMDM generator: `context/refcode/FlowMDM/runners/generate-ex.py`
- HumanML3D decode internals: `context/refcode/FlowMDM/data_loaders/humanml/scripts/motion_process.py`
- Stats usage (denorm): `context/refcode/FlowMDM/runners/generate.py:93`
- Related task: see Task 2.4 in `context/tasks/task-anything.md` (HumanML3D visualization and conversion notes)
