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

**Keypoint-Based Fitting Implementation (In Progress)**
- Implementing unified keypoint-based fitting system for all SMPL model variants (SMPL, SMPL-H, SMPL-X)
- Developing optimization framework for fitting model parameters to 2D/3D keypoint data
- Creating consistent API for pose estimation and body shape inference across model types

## Development Timeline

### Round 1: Core Functionality

- [x] **Model Loading and Conversion:**
  - [x] Load SMPL-X and SMPL-H models.
  - [x] Convert models to Trimesh format for visualization and processing.
- [ ] **Text2Motion Conversion:**
  - [ ] Convert text2motion skeletons to SMPL-H models.
- [ ] **Visualization:**
  - [ ] Visualize SMPL-X and SMPL-H models.

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