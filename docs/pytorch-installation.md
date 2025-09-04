# PyTorch Installation Guide

## Overview

The `smplx-toolbox` package does **NOT** include PyTorch as a dependency. Users must install PyTorch themselves based on their hardware and requirements.

## PyTorch Installation Instructions

### For NVIDIA GPU Users (CUDA >= 12.6)

```bash
# Install PyTorch with CUDA 12.6 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Then install smplx-toolbox
pip install smplx-toolbox
```

### For NVIDIA GPU Users (Other CUDA Versions)

Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) and select your CUDA version:

```bash
# Example for CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Example for CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install smplx-toolbox
pip install smplx-toolbox
```

### For CPU-Only Users

```bash
# Install CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Then install smplx-toolbox
pip install smplx-toolbox
```

### For AMD GPU Users (ROCm)

```bash
# Install PyTorch with ROCm support
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# Then install smplx-toolbox
pip install smplx-toolbox
```

## Development Setup

For development with pixi, PyTorch is automatically configured based on your platform:

- **Windows/Linux**: GPU version with CUDA >= 12.6
- **macOS**: CPU version

```bash
# Development environment (includes PyTorch)
pixi install -e dev

# Check PyTorch installation
pixi run -e dev python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Verification

After installation, verify PyTorch is working:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

## Why This Approach?

1. **Flexibility**: Users can choose the exact PyTorch version for their hardware
2. **Compatibility**: Avoids conflicts between different PyTorch builds
3. **Size**: Keeps the package lightweight without large PyTorch dependencies
4. **Hardware-specific**: Users get the optimal PyTorch build for their system

## Troubleshooting

### Import Error: "No module named 'torch'"

This means PyTorch is not installed. Follow the installation instructions above.

### CUDA Not Available

1. Verify you installed the CUDA version of PyTorch
2. Check your NVIDIA driver supports the CUDA version
3. Verify CUDA runtime is installed on your system

### Performance Issues

- For GPU workloads, ensure you installed the CUDA version, not CPU-only
- For CPU workloads, the CPU version may be faster than CUDA on CPU

## Example Installation Commands

```bash
# Complete setup for CUDA users
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install smplx-toolbox
python -c "import torch; import smplx_toolbox; print('✅ Setup complete!')"

# Complete setup for CPU users  
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install smplx-toolbox
python -c "import torch; import smplx_toolbox; print('✅ Setup complete!')"
```
