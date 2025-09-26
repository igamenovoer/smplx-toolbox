# ROADMAP

## Overview

The SMPL-X Toolbox is a Python library designed to provide a comprehensive set of utilities for working with SMPL-X and SMPL-H human parametric models. This roadmap outlines our iterative development plan for delivering essential tools that streamline workflows in 3D human body modeling, animation, and conversion.

## Vision

To become the go-to toolkit for researchers, developers, and artists working with SMPL-X and SMPL-H models by providing:

- **Robust model support** for both SMPL-X and SMPL-H.
- **Model conversion utilities** to and from various DCC-oriented formats (UE5, Maya, Blender).
- **Seamless interoperability** with existing ecosystems (PyTorch, Trimesh).
- **Optimization frameworks** for fitting models to 2D images and 3D keypoints.
- **Format converters** for popular animation and motion capture formats.

### Design Philosophy

**Iterative Development:** The library is developed in an iterative manner, with each round touching on all aspects of the functional requirements. This approach ensures that the library is continuously improving and that new features are added in a structured and organized way.

## Next Step (Priority)

- Text2Motion skeleton interop: develop conversion utilities between the
  Text2Motion skeleton and SMPL/SMPL-X skeletons (bidirectional where
  possible), including joint name mapping, scale/axis conventions, and
  robust fallbacks for missing joints. This enables reusing Text2Motion
  datasets and outputs with SMPL-based pipelines. (Reference: search for
  “Text2Motion skeleton” for canonical joint layouts and conventions.)

Progress toward this interop:
- [x] FlowMDM environment installed and set up (SpaCy + chumpy) via Pixi tasks.
- [x] Body models symlink created for FlowMDM
      (`context/refcode/FlowMDM/body_models -> ../../../data/body_models`).
- [x] Motion generation verified:
  - [x] Babel with SMPL/SMPL‑X exports → `tmp/flowmdm-out/babel`.
  - [x] HumanML3D skeleton/videos → `tmp/flowmdm-out/humanml3d`.
- [x] Added convenience tasks: `flowmdm-gen-babel`, `flowmdm-gen-humanml`.
- [x] Wrote hint with exact commands and paths:
      `context/hints/howto-run-flowmdm-and-redirect-output.md`.

Next up under this priority:
- [ ] Define joint name mapping between FlowMDM/BABEL/HumanML3D outputs and
      our `NamedPose`/Unified SMPL inputs (AA, global_orient, transl).
- [ ] Implement conversion utilities and tests for direct AA transfer.
- [ ] Prototype 3D keypoint matching path for joints not present in FlowMDM.

## Currently Working / Recent Changes

Current Working Task
- See: `context/tasks/task-fix-humanml-to-smplx-error.md` (HumanML3D → SMPL‑X retargeting: neutral alignment + delta application; orientation and mapping validation.)

Keypoint-Based Fitting (completed; validate on real data next)
- [x] Loss builders (under `src/smplx_toolbox/optimization/`)
  - [x] `KeypointMatchLossBuilder` (3D, unified names + packed native order)
  - [x] `ProjectedKeypointMatchLossBuilder` (2D via camera projection)
  - [x] `ShapePriorLossBuilder` (L2 on betas)
  - [x] `AnglePriorLossBuilder` (heuristic knees/elbows)
- [x] VPoser prior
  - [x] `VPoserPriorLossBuilder.by_pose_latent(z, weight)` (L2 on latent)
  - [x] `VPoserPriorLossBuilder.by_pose(pose, w_pose_fit, w_latent_l2)` (self‑reconstruction + latent L2)
  - [x] Convenience: `encode_pose_to_latent`, `decode_latent_to_pose`
- [x] Robustification utilities: `GMoF` module and functional `gmof`
- [x] Unified model exposes `extras['betas']` for shape prior
- [x] VPoser runtime + interop helpers
  - [x] Minimal `VPoserModel` (v2‑compatible) encode/decode
  - [x] `convert_named_pose_to_pose_body(npz) -> (B,63)` and back
- [x] Documentation
  - [x] `context/tasks/features/keypoint-match/task-vposer-prior.md` (single source of truth)
  - [x] `docs/vposer.md` (runtime usage and mapping helpers)
  - [x] `docs/fitting/helper.md` (end-to-end keypoint fitting helper usage)
  - [x] Updated smoke scripts under `tests/fitting/` to include visualization and
        DOF toggles; helper scripts now align iteration budget for parity.
  - [x] Updated docs on `NamedPose` (root-aware) and VPoser integration using
        `VPoserModel.from_checkpoint`.

Model Input Refactor
- [x] Introduced `NamedPose` as the single intrinsic pose source (pelvis excluded)
- [x] Separated `global_orient` as a standalone `(B, 3)` input
- [x] Removed segmented pose fields from `UnifiedSmplInputs` (root_orient/pose_body/hands/face)
- [x] Updated examples/docs to NamedPose-first pattern; clarified VPoser vs global_orient separation
- [x] Smoke tests refactored to optimize `NamedPose.packed_pose` + `global_orient`; excluded global_orient from L2 reg

Hints
- [x] Added guidance: applying VPoser without affecting global orientation
      (`context/hints/smplx-kb/howto-apply-vposer-with-global-orient.md`)

## Development Timeline

### Round 1: Core Functionality

- [x] **Model Loading and Conversion:**
  - [x] Load SMPL-X and SMPL-H models.
  - [x] Convert models to Trimesh format for visualization and processing.
- [ ] **Text2Motion Conversion:**
  - [ ] Convert text2motion skeletons to SMPL-H models.
- [x] **Visualization:**
  - [x] Visualize SMPL-X and SMPL-H models (PyVista)
  - [x] SMPL-specific visualizer (`SMPLVisualizer`) with mesh, joints, skeleton
  - [x] Joints labeling and axis overlays (`add_axes`)
  - [x] Manual smoke script under `tests/smoke_test_show_mesh.py`

### Round 2: Advanced Features

- [ ] **Model Conversion:**
  - [ ] Convert SMPL-H models to SMPL-X for both static and animated models.
- [x] **Keypoint Fitting Helper:**
  - [x] Implemented `SmplKeypointFittingHelper` to manage data terms, priors
        (VPoser, L2 pose/shape), DOF toggles, custom terms, and iteration.
        See `docs/fitting/helper.md` and tests under `unittests/fitting`.
- [ ] **Animation:**
  - [ ] Define animation format for SMPL-H and SMPL-X.
  - [ ] Convert animation to and from BVH format.

### Round 3: To Be Determined

The goals for the third round of development will be determined based on the progress of the first two rounds and feedback from the community.
