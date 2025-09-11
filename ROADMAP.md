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

## Currently Working

Keypoint-Based Fitting (implemented; pending validation on real data)
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
  - [x] `convert_struct_to_pose(PoseByKeypoints) -> (B,63)`
  - [x] `convert_pose_to_struct((B,63)/(B,21,3)) -> PoseByKeypoints`
- [x] Documentation
  - [x] `context/tasks/features/keypoint-match/task-vposer-prior.md` (single source of truth)
  - [x] `docs/vposer.md` (runtime usage and mapping helpers)

Pending next
- [ ] Validate on real 2D/3D detections and full optimization schedules
- [ ] Add targeted unit tests and minimal examples for each builder
- [ ] Tune robustifier scales/default weights based on validation results

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
- [ ] **Keypoint Fitting:**
  - [ ] Develop keypoint fitting for all SMPL model variants (SMPL, SMPL-H, SMPL-X) optimization.
- [ ] **Animation:**
  - [ ] Define animation format for SMPL-H and SMPL-X.
  - [ ] Convert animation to and from BVH format.

### Round 3: To Be Determined

The goals for the third round of development will be determined based on the progress of the first two rounds and feedback from the community.
