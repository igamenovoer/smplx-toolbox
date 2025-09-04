# ROADMAP

## Overview

SMPL-X Toolbox aims to provide a comprehensive collection of developer utilities for working with the SMPL-X human parametric model. This roadmap outlines our planned features and development timeline for delivering essential tools that streamline workflows in 3D human body modeling and animation.

## Vision

To become the go-to toolkit for researchers, developers, and artists working with SMPL-X models, providing:
- **SMPL-X focused APIs** - all core functionality built specifically for SMPL-X models
- **Model conversion utilities** - convert SMPL/SMPLH to SMPL-X before processing
- **Seamless interoperability** with existing ecosystems (PyTorch, Trimesh, DCC tools)
- **Optimization frameworks** for fitting SMPL-X to various data sources
- **Format converters** for popular animation and 3D content creation pipelines

### Design Philosophy
**SMPL-X First:** Our core APIs are designed exclusively for SMPL-X models. Other model variants (SMPL, SMPLH) should be converted to SMPL-X format before use. This approach ensures:
- Consistent API surface with full expressiveness (body + hands + face)
- Simplified maintenance and testing
- Better performance and reliability
- Future-proof architecture for advanced features

## Development Timeline

### 🚀 Phase 1: Foundation (Q1 2025)
**Core SMPL-X Operations**

- [x] Project setup and structure
- [ ] SMPL-X model loader (focused exclusively on SMPL-X format)
- [ ] Model conversion utilities (SMPL/SMPLH → SMPL-X)
- [ ] SMPL-X parameter management system (pose, shape, expression, translation)
- [ ] Mesh generation and vertex extraction
- [ ] Trimesh integration and basic export utilities (OBJ, PLY)
- [ ] Model introspection tools (joint names, topology info, DOF summary)

**Key Deliverables:**
- `smplx_toolbox.core.load_smplx_model()` API (SMPL-X only)
- `smplx_toolbox.convert.smpl_to_smplx()` and `smplh_to_smplx()` utilities
- `SMPLXParameters` dataclass with device management
- Basic mesh export functionality
- Comprehensive test suite

**Model Conversion Features:**
- Automatic parameter mapping from SMPL/SMPLH to SMPL-X space
- Zero-padding for missing parameters (hands, face expressions)
- Validation and compatibility checking
- Batch conversion utilities for datasets

### 🎯 Phase 2: Optimization Framework (Q2 2025)
**Objective Function Builder**

- [ ] Modular objective term system
- [ ] Keypoint matching objectives (2D/3D)
- [ ] VPoser pose prior integration (with licensing compliance)
- [ ] Configurable weight management
- [ ] Optimization utilities (Adam, L-BFGS wrappers)
- [ ] Loss term composition and reporting

**Key Deliverables:**
- `objectives/` module with pluggable terms
- Keypoint fitting examples and tutorials
- Prior abstraction layer for extensibility
- Single-frame optimization demos

### 🔄 Phase 3: Animation Pipeline (Q3 2025)
**BVH Conversion & Text2Motion Bridge**

- [ ] BVH reader/writer with multiple format support
- [ ] SMPL-X ↔ BVH bidirectional conversion
- [ ] Rotation format handling (axis-angle, Euler, quaternion)
- [ ] Text2Motion skeleton mapping system
- [ ] Keypoint sequence to SMPL-X pose conversion
- [ ] Temporal interpolation and smoothing utilities
- [ ] Frame rate and timing preservation

**Key Deliverables:**
- `io/bvh_*` modules with robust format handling
- `interop/text2motion_mapper` for skeleton conversion
- Animation retargeting examples
- Round-trip conversion validation

### 🎨 Phase 4: Visualization & Export (Q4 2025)
**Advanced Visualization & DCC Integration**

- [ ] Interactive 3D mesh viewer
- [ ] Animation playback and scrubbing
- [ ] Parameter space visualization tools
- [ ] Before/after optimization comparisons
- [ ] Export utilities for DCC tools:
  - [ ] Blender add-on/scripts
  - [ ] Maya/3ds Max compatibility
  - [ ] Unity/Unreal Engine pipelines
- [ ] Advanced format support (FBX, glTF, USD)

**Key Deliverables:**
- Interactive visualization toolkit
- DCC tool integration packages
- Professional animation export workflows
- User documentation and tutorials

## Architecture Principles

### SMPL-X Unified Approach

**Why SMPL-X Only?**
Our decision to focus exclusively on SMPL-X models in core APIs provides several key advantages:

1. **Unified Interface**: Single, consistent API surface reduces complexity
2. **Full Expressiveness**: Access to body, hands, and facial expressions in all operations
3. **Performance**: Optimized code paths without branching for different model types
4. **Future-Proof**: Built for the most advanced parametric model available
5. **Simplified Testing**: Comprehensive coverage without model variant combinatorics

**Model Conversion Strategy:**
- **Automatic Conversion**: Built-in utilities convert SMPL/SMPLH to SMPL-X seamlessly
- **Parameter Mapping**: Intelligent mapping preserves original pose and shape
- **Extension Handling**: Missing parameters (hands, face) initialized with neutral defaults
- **Backward Compatibility**: Convert SMPL-X back to simpler models when needed

**Supported Conversion Paths:**
```
SMPL → SMPL-X (body pose + shape, neutral hands/face)
SMPLH → SMPL-X (body + hands, neutral face)  
SMPL-X → SMPL (body only, hands/face discarded)
SMPL-X → SMPLH (body + hands, face discarded)
```