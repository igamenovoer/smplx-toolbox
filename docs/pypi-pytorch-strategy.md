# PyTorch GPU/CPU Fallback Configuration for PyPI Publication

## Overview

When publishing your package to PyPI, you need a strategy for users to automatically get the right PyTorch version (GPU or CPU) based on their hardware. Here are several approaches:

## Solution 1: Optional Dependencies with User Choice (Recommended)

### PyProject.toml Configuration

Your `pyproject.toml` should have:

```toml
[project]
dependencies = [ 
    "rich>=14.1.0,<15", 
    "attrs>=25.3.0,<26",
    # No PyTorch in base dependencies - let users choose
]

[project.optional-dependencies]
# GPU support (explicit choice)
gpu = [
    "torch>=2.0.0,<3.0.0",
    "torchvision>=0.15.0,<1.0.0",
]
# CPU-only explicit option
cpu = [
    "torch>=2.0.0,<3.0.0",
    "torchvision>=0.15.0,<1.0.0"
]
```

### Installation Instructions for Users

**For GPU users:**
```bash
# Option 1: Let PyTorch installer detect GPU
pip install smplx-toolbox[gpu]
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Option 2: One-line with specific CUDA version
pip install smplx-toolbox[gpu] torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**For CPU users:**
```bash
# Option 1: Use CPU extra
pip install smplx-toolbox[cpu]

# Option 2: Explicit CPU installation
pip install smplx-toolbox[cpu] torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Solution 2: Automatic Detection with Helper Script

### Include Detection Script in Package

Users can run:
```bash
pip install smplx-toolbox
smplx-install-pytorch  # Our custom detection script
```

The script (`pytorch_installer.py`) will:
1. Detect GPU capability
2. Install appropriate PyTorch version
3. Verify installation

### Benefits:
- Automatic hardware detection
- No user decision required
- Graceful fallback to CPU

## Solution 3: Environment Markers (Advanced)

For more sophisticated control:

```toml
[project]
dependencies = [ 
    "rich>=14.1.0,<15", 
    "attrs>=25.3.0,<26",
    # Conditional PyTorch installation
    "torch>=2.0.0; platform_machine != 'unknown'",
]

[project.optional-dependencies]
gpu = [
    "torch>=2.0.0; sys_platform != 'darwin' or platform_machine != 'arm64'",
]
```

## Solution 4: Multiple Package Variants (Heavyweight)

Create separate packages:
- `smplx-toolbox` (base package, no PyTorch)
- `smplx-toolbox-gpu` (includes GPU PyTorch)
- `smplx-toolbox-cpu` (includes CPU PyTorch)

## Recommended Approach for Your Package

### 1. Update pyproject.toml

```toml
[project]
dependencies = [ 
    "rich>=14.1.0,<15", 
    "attrs>=25.3.0,<26",
    # Note: No PyTorch here - users choose via extras
]

[project.optional-dependencies]
gpu = [
    "torch>=2.0.0,<3.0.0",
    "torchvision>=0.15.0,<1.0.0",
]
cpu = [
    "torch>=2.0.0,<3.0.0", 
    "torchvision>=0.15.0,<1.0.0"
]
all = ["smplx-toolbox[gpu]"]

[project.scripts]
smplx-install-pytorch = "smplx_toolbox.utils.pytorch_installer:main"
```

### 2. Documentation for Users

Include in your README:

```markdown
## Installation

### Quick Install (Automatic GPU Detection)
```bash
pip install smplx-toolbox
smplx-install-pytorch  # Detects GPU and installs appropriate PyTorch
```

### Manual Installation

**For NVIDIA GPU users:**
```bash
pip install smplx-toolbox[gpu] torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**For CPU-only users:**
```bash
pip install smplx-toolbox[cpu] torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**For development:**
```bash
pip install smplx-toolbox[dev]
```
```

### 3. Runtime Checks in Your Code

Add to your main module:

```python
def verify_pytorch():
    try:
        import torch
        if not torch.cuda.is_available():
            print("Warning: CUDA not available. Running in CPU mode.")
        return True
    except ImportError:
        print("PyTorch not found. Install with: smplx-install-pytorch")
        return False
```

## Benefits of This Approach

1. **User Control**: Users explicitly choose GPU or CPU
2. **Clear Documentation**: Installation instructions are straightforward
3. **Automatic Helper**: Detection script for convenience
4. **PyPI Compatible**: Works with standard PyPI infrastructure
5. **No Surprises**: Users get exactly what they request

## Testing the Configuration

Test locally with:
```bash
# Test CPU installation
pip install -e .[cpu] torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Test GPU installation  
pip install -e .[gpu] torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Test detection script
python -m smplx_toolbox.utils.pytorch_installer
```

This approach balances automation with user control, ensuring compatibility across different deployment scenarios while providing clear upgrade paths.
