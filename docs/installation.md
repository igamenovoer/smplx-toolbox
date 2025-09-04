# Installation

## Requirements

- Python 3.8 or higher
- Operating System: Windows, macOS, or Linux

## Installation Methods

### PyPI Installation (Recommended)

```bash
pip install smplx-toolbox
```

### Development Installation

For development or to get the latest features:

```bash
git clone https://github.com/yourusername/smplx-toolbox.git
cd smplx-toolbox
```

#### Using Pixi (Recommended for Development)

```bash
# Install Pixi if you haven't already
curl -fsSL https://pixi.sh/install.sh | bash

# Install project dependencies
pixi install

# Install in development mode
pixi run install-dev
```

#### Using pip directly

```bash
pip install -e .[dev]
```

## Optional Dependencies

### Visualization Features

```bash
pip install smplx-toolbox[viz]
```

### Documentation Building

```bash
pip install smplx-toolbox[docs]
```

### All Features

```bash
pip install smplx-toolbox[all]
```

## Verify Installation

```python
import smplx_toolbox as sxt
print(sxt.__version__)
```

Or from command line:

```bash
smplx-toolbox --version
```

## Troubleshooting

### Common Issues

**Import Error**: Make sure you're using Python 3.8 or higher:
```bash
python --version
```

**Missing Dependencies**: Reinstall with all dependencies:
```bash
pip install --upgrade --force-reinstall smplx-toolbox[all]
```

**CUDA Issues**: For GPU acceleration, ensure PyTorch is properly installed:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Environment Issues

If you encounter environment conflicts, consider using a virtual environment:

```bash
python -m venv smplx-env
source smplx-env/bin/activate  # On Windows: smplx-env\Scripts\activate
pip install smplx-toolbox
```

## Development Setup

For contributors and developers:

```bash
# Clone the repository
git clone https://github.com/yourusername/smplx-toolbox.git
cd smplx-toolbox

# Install Pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Install development environment
pixi install
pixi run install-dev

# Run tests to verify setup
pixi run test

# Install pre-commit hooks
pixi run pre-commit-install
```
