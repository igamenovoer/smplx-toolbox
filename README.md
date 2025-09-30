# SMPL-X Toolbox

A comprehensive collection of developer utilities for working with the SMPL-X human parametric model. This toolbox aims to provide essential tools for researchers, developers, and artists working with 3D human body modeling and animation.

## Overview

SMPL-X (SMPL eXpressive) is an expressive body model that extends SMPL with articulated hands and an expressive face. This toolbox provides utilities to streamline common workflows when working with SMPL-X models, making it easier to integrate human body modeling into various applications and pipelines.

## Planned Features

### 🎯 Optimization Tools
- Parameter optimization algorithms for fitting SMPL-X to various data sources
- Pose and shape optimization utilities
- Custom loss function implementations
- Constraint-based optimization helpers

### 📊 Objective Generation
- Automatic objective function generation for common fitting scenarios
- Landmark-based objectives
- Silhouette matching objectives
- Motion capture alignment objectives
- Multi-view consistency objectives

### 🎨 Visualization
- Interactive 3D visualization tools
- Mesh rendering utilities
- Animation playback and scrubbing
- Parameter space visualization
- Comparison tools for before/after optimization

### 🔄 Format Conversion
- Export utilities for popular DCC (Digital Content Creation) tools:
  - Blender integration
  - Maya/3ds Max compatibility
  - Unreal Engine/Unity support
  - Houdini workflows
- Standard format converters (FBX, OBJ, glTF, USD)
- Animation data conversion tools

### 🛠️ Utilities
- SMPL-X model loading and manipulation
- Batch processing tools
- Configuration management
- Data pipeline helpers
- Validation and debugging utilities

## Target Applications

- **Research**: Academic research in computer vision, graphics, and human motion analysis
- **Animation**: Character animation and rigging for films, games, and VR/AR
- **Fitness & Health**: Body measurement and posture analysis applications
- **Fashion**: Virtual try-on and garment fitting simulations
- **Motion Capture**: Processing and retargeting motion capture data

## SMPL-X Background

SMPL-X is a unified model that represents:
- **Body**: Based on the SMPL model with 10 shape parameters and 23 joint rotations
- **Hands**: Detailed finger articulation with MANO hand model integration
- **Face**: Expressive facial modeling with flame-based face model
- **Eyes**: Separate eye pose parameters for realistic gaze modeling

The model uses a learned linear blend skinning approach with pose-dependent corrective blend shapes, making it differentiable and suitable for optimization-based fitting.

## Getting Started

> **Note**: This project is currently in development. The package does **NOT** include PyTorch as a dependency - users must install PyTorch separately.

### Prerequisites
- Python 3.11+
- **PyTorch 2.0+** (must be installed separately - see installation instructions below)
- NumPy, SciPy (for numerical computations)
- Trimesh (for 3D mesh processing)

### Installation

**Step 1: Install PyTorch**

The package requires PyTorch but does not install it automatically. Choose the appropriate installation based on your hardware:

```bash
# For NVIDIA GPU users (CUDA 12.6)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# For CPU-only users
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For other CUDA versions, visit: https://pytorch.org/get-started/locally/
```

**Step 2: Install SMPL-X Toolbox**

```bash
pip install smplx-toolbox
```

**Step 3: Verify Installation**

```python
import torch
import smplx_toolbox

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("✅ SMPL-X Toolbox ready!")
```

### Development Setup

For development with pixi (includes automatic PyTorch configuration):

```bash
# Clone the repository
git clone https://github.com/igamenovoer/smplx-toolbox.git
cd smplx-toolbox

# Install development environment (PyTorch configured automatically)
pixi install -e dev

# Verify setup
pixi run -e dev python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

For detailed PyTorch installation instructions, see [docs/pytorch-installation.md](docs/pytorch-installation.md).

## GenericSkeleton (MJCF) and Topology Graphs

The toolbox provides a lightweight MJCF-backed skeleton wrapper for reusable motion/retargeting workflows:

```python
from smplx_toolbox.core.skeleton import GenericSkeleton
import networkx as nx

# Load an MJCF skeleton (human rigs typically have only actuated joints)
skel = GenericSkeleton.from_mjcf_file("path/to/skeleton.xml", name="humanoid")

# Names and base
print("base:", skel.base_link_name)
print("joints:", skel.joint_names)   # preferred canonical set for human workflows
print("links:", skel.link_names)     # optional; segment/geometry-centric tasks

# Forward kinematics (dict[link_name -> 4x4])
cfg = {j: 0.0 for j in skel.joint_names}
fk_links = skel.link_fk(cfg)

# 1) Joint topology (nodes = joints). Each edge carries the child link name.
JG = skel.get_joint_topology()
order_j = list(nx.topological_sort(JG))
for parent, child in JG.edges():
    child_link = JG[parent][child]["link"]  # child body (link) between joints
    print(parent, "->", child, "via link:", child_link)

# 2) Link topology (nodes = links). Each edge carries the connecting joint name(s).
LG = skel.get_link_topology()
order_l = list(nx.topological_sort(LG))
for parent_link, child_link in LG.edges():
    joint_names = LG[parent_link][child_link]["joints"]
    print(parent_link, "->", child_link, "via joints:", joint_names)

# 3) Full mixed topology (links and joints). Nodes have node_type: "link" or "joint".
FG = skel.get_full_topology()
for n, data in FG.nodes(data=True):
    print(n, data.get("node_type"))
```

Notes
- Human skeleton pipelines typically use joint names (pelvis/hip/wrist, etc.); link names are available when you need segment-level metadata/geometry.
- The root/global pose in SMPL(-X) maps to the base link world transform (apply as an external SE(3) after FK).

## Contributing

We welcome contributions from the community! Whether you're interested in:
- Adding new optimization algorithms
- Improving visualization tools
- Creating DCC tool integrations
- Writing documentation and tutorials
- Reporting bugs and suggesting features

Please feel free to open issues or submit pull requests.

## Related Projects

- [SMPL-X Official Repository](https://smpl-x.is.tue.mpg.de/)
- [SMPL Model](https://smpl.is.tue.mpg.de/)
- [PyTorch SMPL-X](https://github.com/vchoutas/smplx)

## License

This project will be released under an appropriate open-source license (to be determined based on dependencies and community feedback).

## Citation

If you use this toolbox in your research, please cite:

```bibtex
@software{smplx_toolbox,
  title={SMPL-X Toolbox: Developer Utilities for Human Parametric Modeling},
  author={[Author names to be added]},
  year={2025},
  url={https://github.com/igamenovoer/smplx-toolbox}
}
```

## Acknowledgments

This project builds upon the excellent work of the SMPL-X team and the broader computer vision and graphics research community. We acknowledge the contributions of all researchers and developers who have made human parametric modeling accessible and practical.

---

**Status**: 🚧 In Development - Project structure and initial implementations coming soon!
