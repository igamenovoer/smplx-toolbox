# SMPL-X Toolbox

A comprehensive collection of developer utilities for working with the SMPL-X human parametric model.

## Overview

SMPL-X (SMPL eXpressive) is an expressive body model that extends SMPL with articulated hands and an expressive face. This toolbox provides utilities to streamline common workflows when working with SMPL-X models, making it easier to integrate human body modeling into various applications and pipelines.

## Installation

```bash
pip install smplx-toolbox
```

### Development Installation

```bash
git clone https://github.com/yourusername/smplx-toolbox.git
cd smplx-toolbox
pixi install
pixi run install-dev
```

## Quick Start

```python
import smplx_toolbox as sxt

# Load SMPL-X model
model = sxt.load_smplx_model('path/to/smplx/model')

# Create parameters
params = sxt.SMPLXParameters()

# Generate mesh
mesh = model.forward(params)

# Visualize
viewer = sxt.InteractiveViewer()
viewer.show(mesh)
```

## Features

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
- Export utilities for popular DCC (Digital Content Creation) tools
- Standard format converters (FBX, OBJ, glTF, USD)
- Animation data conversion tools

## Command Line Interface

```bash
# Optimize SMPL-X parameters
smplx-toolbox optimize input.json --output optimized.json

# Convert between formats
smplx-toolbox convert model.obj output.fbx --format fbx

# Launch interactive visualizer
smplx-toolbox visualize parameters.json
```

## Documentation

- [Installation Guide](https://yourusername.github.io/smplx-toolbox/installation/)
- [Quick Start Tutorial](https://yourusername.github.io/smplx-toolbox/quickstart/)
- [API Reference](https://yourusername.github.io/smplx-toolbox/api/)
- [Examples](https://yourusername.github.io/smplx-toolbox/examples/)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this toolbox in your research, please cite:

```bibtex
@software{smplx_toolbox,
  title={SMPL-X Toolbox: Developer Utilities for Human Parametric Models},
  author={SMPL-X Toolbox Contributors},
  year={2025},
  url={https://github.com/yourusername/smplx-toolbox}
}
```
