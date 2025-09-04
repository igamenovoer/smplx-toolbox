# Configuration Summary: PyTorch Installation Strategy

## Changes Made

### 1. PyProject.toml (PyPI Publication)

**Before**: Included PyTorch as a dependency with conditional installation logic
**After**: **Removed PyTorch completely** from dependencies

```toml
# OLD (removed)
dependencies = [ 
    "torch>=2.0.0; extra != 'gpu'",
    "torchvision>=0.15.0; extra != 'gpu'"
]

# NEW (current)
dependencies = [ 
    "rich>=14.1.0,<15", 
    "attrs>=25.3.0,<26",
    # Note: PyTorch is NOT included as a dependency
    # Users must install PyTorch themselves based on their hardware
]
```

**Result**: Users are responsible for installing PyTorch themselves

### 2. Pixi Configuration (Development)

**Added platform-specific PyTorch installation**:

- **Windows & Linux**: GPU version with CUDA >= 12.6
- **macOS (Intel & ARM)**: CPU version from PyPI

```toml
# GPU PyTorch for Windows and Linux with CUDA >= 12.6
[tool.pixi.feature.dev.target.win-64.dependencies]
pytorch = {version = ">=2.7", channel = "pytorch"}
torchvision = {channel = "pytorch"}
pytorch-cuda = ">=12.6"

[tool.pixi.feature.dev.target.linux-64.dependencies]
pytorch = {version = ">=2.7", channel = "pytorch"}
torchvision = {channel = "pytorch"}
pytorch-cuda = ">=12.6"

# CPU PyTorch for macOS
[tool.pixi.feature.dev.target.osx-64.pypi-dependencies]
torch = {version = ">=2.0.0", index = "https://download.pytorch.org/whl/cpu"}
torchvision = {version = ">=0.15.0", index = "https://download.pytorch.org/whl/cpu"}

[tool.pixi.feature.dev.target.osx-arm64.pypi-dependencies]
torch = {version = ">=2.0.0", index = "https://download.pytorch.org/whl/cpu"}
torchvision = {version = ">=0.15.0", index = "https://download.pytorch.org/whl/cpu"}
```

### 3. Documentation Updates

- **Updated**: `docs/pytorch-installation.md` with clear user instructions
- **Updated**: `README.md` with installation steps
- **Removed**: PyTorch installer script and CLI command

## User Experience

### For PyPI Users

1. **Install PyTorch first** (based on their hardware):
   ```bash
   # GPU users
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   
   # CPU users  
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Then install the package**:
   ```bash
   pip install smplx-toolbox
   ```

### For Developers (Pixi)

1. **Just run pixi install**:
   ```bash
   pixi install -e dev
   ```

2. **PyTorch is automatically configured**:
   - Windows/Linux: GPU with CUDA >= 12.6
   - macOS: CPU version

## Benefits of This Approach

1. **✅ User Control**: Users choose the exact PyTorch version for their hardware
2. **✅ Lightweight**: Package doesn't include large PyTorch dependencies  
3. **✅ Compatibility**: Avoids conflicts between different PyTorch builds
4. **✅ Development Convenience**: Pixi automatically configures PyTorch for development
5. **✅ Platform Optimization**: Each platform gets the optimal PyTorch build

## Key Files Modified

- `pyproject.toml` - Removed PyTorch dependencies, added platform-specific pixi config
- `docs/pytorch-installation.md` - Complete user installation guide
- `README.md` - Updated installation instructions
- Removed: `src/smplx_toolbox/utils/pytorch_installer.py`

## Verification Commands

### For Development
```bash
# Install dev environment
pixi install -e dev

# Check PyTorch
pixi run -e dev python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### For Users
```bash
# Install PyTorch + package
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install smplx-toolbox

# Verify
python -c "import torch; import smplx_toolbox; print('✅ Setup complete!')"
```

This configuration follows the principle: **Development convenience with user flexibility**.
