# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The SMPL-X Toolbox is a comprehensive Python library for working with SMPL-X human parametric models. It provides optimization tools, visualization capabilities, format conversion utilities, and a professional development environment.

## Project Structure

```
smplx-toolbox/
├── src/smplx_toolbox/     # Main package source code
├── scripts/               # Command-line interface tools
├── tests/                 # Test suite
├── docs/                  # Documentation source
├── context/               # AI assistant workspace
├── tmp/                   # Temporary working files
├── .github/workflows/     # CI/CD automation
└── [config files]        # Project configuration
```

## Key Resources

### Context Directory
The `context/` directory is your primary workspace for understanding and contributing to this project. It follows the [Context Directory Guide](.magic-context/general/context-dir-guide.md) and contains:

- **design/** - Technical specifications and architecture
- **plans/** - Implementation roadmaps and strategies  
- **tasks/** - Current and planned work items
- **logs/** - Development session records
- **roles/** - Role-based AI assistant configurations

### Important Files
- `context/tasks/goal.md` - Primary project objectives and success criteria
- `pyproject.toml` - Python packaging and dependency configuration
- `pixi.toml` - Development environment and task definitions
- `mkdocs.yml` - Documentation build configuration

## Common Development Commands

This project uses **Pixi** for environment and task management. Essential commands:

```bash
# Environment setup
pixi install          # Install dependencies
pixi run install-dev  # Install in development mode

# Testing
pixi run test         # Run test suite  
pixi run test-cov     # Run tests with coverage report

# Code quality
pixi run lint         # Run flake8 linting
pixi run format       # Format code with black
pixi run format-check # Check formatting without changes
pixi run sort-imports # Sort imports with isort
pixi run type-check   # Run mypy type checking
pixi run qa           # Run all quality checks (format-check, sort-imports-check, lint, type-check, test)

# Documentation
pixi run docs-serve   # Start documentation server

# Build and package
pixi run build        # Build package
pixi run clean        # Clean build artifacts
```

## AI Assistant Roles

When working on this project, consider adopting specific roles based on the task:

- **Backend Developer** - Python development, API design, optimization algorithms
- **Computer Vision Specialist** - 3D modeling, mesh processing, parameter fitting
- **Documentation Writer** - Technical writing, API docs, user guides
- **Test Engineer** - Quality assurance, testing strategies, validation

## Architecture

### Core Module Structure
- **`src/smplx_toolbox/core/`** - SMPL-X model loading, parameter management, and core functionality
- **`src/smplx_toolbox/optimization/`** - Parameter fitting algorithms, objective functions, and constraint handling
- **`src/smplx_toolbox/visualization/`** - 3D rendering, interactive tools, and animation playback
- **`src/smplx_toolbox/utils/`** - Format conversion, validation utilities, and helper functions

### Key Dependencies
- **PyTorch** - Deep learning framework for optimization and differentiable operations
- **Trimesh** - 3D mesh processing and geometry operations
- **OpenCV** - Computer vision operations and image processing
- **NumPy/SciPy** - Numerical computing and scientific calculations

### Target SMPL-X Components
- **Body parameters** - 10 shape parameters and 23 joint rotations from SMPL
- **Hand parameters** - Detailed finger articulation via MANO integration
- **Facial parameters** - Expressive modeling with FLAME-based face model
- **Eye parameters** - Separate gaze control for realistic eye movements

## Current Status

The project is in its **foundation phase**, focusing on:
- Core SMPL-X model handling and parameter management
- Basic optimization framework setup
- Project structure and development tooling
- Initial documentation and testing infrastructure

## Working Guidelines

### Before Starting
1. Check `context/tasks/` for current work items and priorities
2. Review relevant `context/design/` documents for technical context
3. Look at `context/logs/` for recent development history
4. Consider your role and expertise area

### During Development
1. Document decisions and findings in appropriate `context/` directories
2. Update task status and progress regularly
3. Follow the established code quality standards (typing, testing, documentation)
4. Test changes thoroughly and update relevant tests

### After Completion
1. Log the development session outcome in `context/logs/`
2. Update task status and any affected plans
3. Consider what knowledge should be preserved for future reference
4. Update documentation if user-facing changes were made

## Code Quality Standards

- **Type Hints**: All functions should have complete type annotations
- **Documentation**: Docstrings for all public functions and classes
- **Testing**: Maintain >90% test coverage
- **Formatting**: Use Black for code formatting, isort for imports
- **Linting**: Code should pass flake8 and mypy checks

## Python Coding Guidelines

This project follows specific Python coding conventions for consistency and maintainability. **All Python classes must adhere to these standards.**

### Class Design Patterns

#### 1. Member Variable Naming
- **Prefix all member variables with `m_`**
- **Initialize all member variables in `__init__()`** 
- **Default to `None` with proper typing**

```python
from typing import Optional, Any, Dict

class SMPLModel:
    def __init__(self) -> None:
        self.m_vertices: Optional[Any] = None
        self.m_faces: Optional[Any] = None  
        self.m_parameters: Optional[Dict[str, Any]] = None
        self.m_device: Optional[str] = None
```

#### 2. Read-Only Properties
- **Use `@property` decorators for read-only access**
- **No property setters allowed**
- **Type annotate all return values**

```python
@property
def vertices(self) -> Optional[Any]:
    """Get the mesh vertices."""
    return self.m_vertices

@property 
def parameter_count(self) -> int:
    """Get the number of model parameters."""
    return len(self.m_parameters) if self.m_parameters else 0
```

#### 3. Explicit Setter Methods
- **Use `set_xxx()` methods for all modifications**
- **Include validation when needed**
- **Type annotate parameters and return values**

```python
def set_parameters(self, parameters: Dict[str, Any]) -> None:
    """Set model parameters with validation."""
    if not isinstance(parameters, dict):
        raise TypeError("Parameters must be a dictionary")
    self.m_parameters = parameters

def set_device(self, device: str) -> None:
    """Set the compute device (cpu/cuda)."""
    if device not in ['cpu', 'cuda']:
        raise ValueError("Device must be 'cpu' or 'cuda'")
    self.m_device = device
```

#### 4. Constructor and Factory Pattern
- **Constructors take no arguments (except `self`)**
- **Use factory methods `cls.from_xxx()` for initialization**
- **Type annotate with `TypeVar` for proper return typing**

```python
from typing import Type, TypeVar

T = TypeVar('T', bound='SMPLModel')

class SMPLModel:
    def __init__(self) -> None:
        # Initialize all member variables to None
        pass
    
    @classmethod
    def from_file(cls: Type[T], model_path: str) -> T:
        """Create model by loading from file."""
        instance = cls()
        # Load and set data using setter methods
        return instance
    
    @classmethod
    def from_parameters(cls: Type[T], params: Dict[str, Any]) -> T:
        """Create model from parameter dictionary."""
        instance = cls()
        instance.set_parameters(params)
        return instance
```

### Documentation Standards

#### 1. NumPy Style Docstrings
- **Use NumPy documentation style for all docstrings**
- **Include Parameters, Returns, Raises, and Examples sections**
- **Document all public methods, functions, and classes**

```python
def optimize_parameters(self, target_vertices: Any, max_iterations: int = 100) -> Dict[str, Any]:
    """
    Optimize model parameters to fit target vertices.
    
    Uses gradient-based optimization to minimize the distance between
    model output and target vertices.
    
    Parameters
    ----------
    target_vertices : array_like
        Target vertex positions to fit
    max_iterations : int, optional
        Maximum optimization iterations (default is 100)
        
    Returns
    -------
    dict
        Dictionary containing optimized parameters and convergence info
        
    Raises
    ------
    ValueError
        If target_vertices shape is incompatible with model
    RuntimeError
        If optimization fails to converge
        
    Examples
    --------
    >>> model = SMPLModel.from_file("model.pkl")
    >>> target = load_target_mesh("target.obj")
    >>> result = model.optimize_parameters(target.vertices)
    >>> print(f"Converged: {result['converged']}")
    """
```

#### 2. Module Documentation
- **Add module-level docstring at the top of each file**
- **Explain module purpose and main components**
- **List key classes/functions**

```python
"""
Core SMPL-X model implementation.

This module provides the primary SMPLModel class for loading, manipulating,
and optimizing SMPL-X human parametric models. It includes utilities for
parameter management, mesh generation, and pose optimization.

Classes
-------
SMPLModel : Main SMPL-X model class
ParameterValidator : Validation utilities for model parameters

Functions
---------
load_smpl_model : Load model from various file formats
validate_pose_parameters : Validate pose parameter constraints
"""
```

### Import Style Guidelines
- **Use absolute imports whenever possible**
- **Group imports: standard library, third-party, local modules** 
- **Avoid relative imports unless absolutely necessary**

```python
# Standard library
from typing import Any, Optional, Dict, List, Type, TypeVar
from pathlib import Path
import json

# Third-party
import numpy as np
import torch
import trimesh

# Local modules  
from smplx_toolbox.core.base_model import BaseModel
from smplx_toolbox.utils.validation import validate_parameters
from smplx_toolbox.utils.io import load_model_data
```

### Complete Class Example

```python
"""
SMPL-X model implementation for human parametric modeling.

This module provides the core SMPLXModel class following project coding
standards with member variable prefixes, read-only properties, explicit
setters, and factory methods.
"""

from typing import Any, Optional, Dict, Type, TypeVar
import torch
import numpy as np

T = TypeVar('T', bound='SMPLXModel')

class SMPLXModel:
    """
    SMPL-X human parametric model implementation.
    
    Provides functionality for loading SMPL-X models, managing parameters,
    and generating meshes with pose, shape, and expression controls.
    
    Attributes
    ----------
    vertices : array_like or None
        Generated mesh vertices
    faces : array_like or None  
        Mesh face connectivity
    parameters : dict or None
        Model parameters including pose, shape, expression
    """
    
    def __init__(self) -> None:
        """Initialize empty SMPL-X model instance."""
        self.m_vertices: Optional[torch.Tensor] = None
        self.m_faces: Optional[torch.Tensor] = None
        self.m_parameters: Optional[Dict[str, torch.Tensor]] = None
        self.m_device: Optional[str] = None
    
    @property
    def vertices(self) -> Optional[torch.Tensor]:
        """Get the mesh vertices."""
        return self.m_vertices
    
    @property
    def faces(self) -> Optional[torch.Tensor]:
        """Get the mesh faces."""
        return self.m_faces
        
    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Set model parameters with validation."""
        if not isinstance(parameters, dict):
            raise TypeError("Parameters must be a dictionary")
        self.m_parameters = parameters
    
    @classmethod
    def from_file(cls: Type[T], model_path: str) -> T:
        """Create model by loading from file."""
        instance = cls()
        # Implementation details...
        return instance
```

### Key Benefits
- **Clear State Management**: `m_` prefix makes internal state obvious
- **Controlled Access**: Read-only properties prevent accidental modification
- **Explicit Changes**: Setter methods provide validation points
- **Flexible Creation**: Factory methods support different initialization scenarios  
- **Type Safety**: Complete type annotations prevent runtime errors
- **Consistent API**: Predictable patterns across all classes

## Common Tasks

### Adding New Features
1. Create task document in `context/tasks/features/`
2. Review or create design documents in `context/design/`
3. Implement with tests and documentation
4. Update relevant configuration files if needed

### Bug Fixes
1. Document the issue in `context/tasks/fixes/`
2. Investigate and document findings
3. Implement fix with test to prevent regression
4. Log the resolution process

### Documentation Updates
1. Make changes in `docs/` directory
2. Test with `pixi run docs-serve`
3. Ensure all examples and code snippets work
4. Consider updating API reference if needed

## Integration Points

### SMPL-X Model
- The core focus is on SMPL-X human parametric models
- Support for pose, shape, facial expression, and hand parameters
- Integration with existing SMPL-X implementations and data formats

### Target Applications
- Research in computer vision and graphics
- Character animation and rigging for entertainment
- VR/AR applications requiring human modeling
- Data analysis and machine learning with human pose/shape data

### External Dependencies
- PyTorch for deep learning and optimization
- Trimesh for 3D mesh processing
- OpenCV for computer vision operations
- Open3D for advanced 3D operations and visualization

## Help and Resources

- Check `context/hints/` for project-specific troubleshooting guides
- Review `context/summaries/` for analysis and knowledge consolidation
- Look at `context/refcode/` for implementation examples and patterns
- Consult the main README.md for user-facing information

Remember: The `context/` directory is your knowledge base. Use it actively to understand the project's history, current state, and future direction.
