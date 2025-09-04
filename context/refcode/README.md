# Reference Code

Reference implementations and examples for the SMPL-X Toolbox project.

## Contents

This directory contains working code examples, reference implementations, and third-party integrations that serve as patterns for new development. These examples demonstrate best practices and proven approaches.

## Document Types

### Implementation Examples
- **smplx-model-loading-example.py** - Reference implementation for model loading
- **optimization-algorithm-example.py** - Working optimization algorithm implementation
- **mesh-visualization-example.py** - 3D visualization and rendering examples
- **parameter-validation-example.py** - Input validation and error handling patterns
- **batch-processing-example.py** - Efficient batch operation implementations

### Integration Patterns
- **blender-integration-example.py** - Blender addon integration pattern
- **pytorch-optimization-example.py** - PyTorch-based optimization implementation
- **opencv-preprocessing-example.py** - Image processing and computer vision patterns
- **trimesh-operations-example.py** - 3D mesh manipulation using trimesh
- **open3d-visualization-example.py** - Advanced visualization with Open3D

### Third-Party Code
- **external-smplx-implementations/** - Reference SMPL-X implementations
- **optimization-libraries-examples/** - Usage patterns for optimization libraries
- **visualization-frameworks-examples/** - Examples from various 3D visualization tools
- **format-conversion-examples/** - File format conversion reference code

### Testing Patterns
- **unit-test-examples.py** - Comprehensive unit testing patterns
- **integration-test-examples.py** - End-to-end testing approaches
- **performance-benchmark-examples.py** - Performance testing and profiling code
- **mock-and-fixture-examples.py** - Test data generation and mocking patterns

## Purpose

Reference code serves to:
- **Provide working examples** of complex implementations
- **Demonstrate best practices** for common development patterns
- **Reduce implementation time** by offering proven solutions
- **Ensure consistency** across different parts of the project
- **Document integration approaches** with external libraries
- **Preserve successful experiments** for future reference

## Code Organization

### File Structure
```
refcode/
├── core/                    # Core functionality examples
├── optimization/            # Optimization algorithm implementations
├── visualization/           # Rendering and display examples
├── integrations/           # Third-party tool integrations
├── testing/                # Testing and validation examples
├── performance/            # Performance optimization examples
└── experiments/            # Research and experimental code
```

### Naming Convention
- Use descriptive names that clearly indicate the functionality
- Include version numbers for evolving implementations
- Add status indicators (working/experimental/deprecated)
- Group related examples in subdirectories

Examples:
- `core/smplx-model-loader-v2-working.py`
- `optimization/gradient-descent-optimizer-experimental.py`
- `visualization/interactive-viewer-deprecated.py`

## Code Quality Standards

### Documentation Requirements
- **Clear purpose statement** at the top of each file
- **Usage examples** with sample inputs and expected outputs
- **Parameter documentation** for all functions and classes
- **Integration notes** explaining how to use with the main project
- **Performance characteristics** and limitations

### Code Standards
- Follow project coding conventions (Black formatting, type hints)
- Include comprehensive error handling
- Add performance profiling where relevant
- Provide test data or generators for examples
- Comment complex algorithms and design decisions

### Example Format
```python
"""
Reference Implementation: SMPL-X Model Loading

Purpose: Demonstrates efficient SMPL-X model loading with error handling
Status: Working (tested with PyTorch 1.10+)
Dependencies: torch, numpy, pickle
Performance: ~50ms load time for standard model

Usage:
    loader = SMPLXModelLoader('path/to/model')
    model = loader.load()
    mesh = model.forward(parameters)

Integration:
    This pattern is used in src/smplx_toolbox/core/model.py
    Modify CACHE_SIZE constant for different memory constraints
"""

import torch
import numpy as np
from typing import Optional, Dict, Any

# [Implementation code with detailed comments]
```

## Categories

### Core Examples
Fundamental functionality implementations:
- Model loading and initialization
- Parameter creation and validation
- Basic mesh generation and manipulation
- File I/O and serialization patterns

### Algorithm Examples
Complex algorithm implementations:
- Optimization algorithms with convergence criteria
- Geometric processing and mesh operations
- Mathematical utilities and transformations
- Performance-critical computational kernels

### Integration Examples
External library and tool integrations:
- DCC tool plugins and scripts
- Machine learning framework integrations
- Visualization library usage patterns
- File format conversion implementations

### Pattern Examples
Software design and architecture patterns:
- Factory patterns for model creation
- Observer patterns for parameter updates
- Strategy patterns for optimization algorithms
- Adapter patterns for external integrations

## Usage Guidelines

### For Developers
- **Study examples** before implementing similar functionality
- **Copy and adapt** patterns rather than starting from scratch
- **Contribute improvements** when finding better approaches
- **Document modifications** when adapting examples for specific needs

### For AI Assistants
- **Reference examples** when implementing similar functionality
- **Suggest relevant examples** when users need implementation guidance
- **Update examples** based on implementation experience
- **Extract patterns** from successful implementations for reuse

## Testing and Validation

### Example Requirements
- All examples should be **runnable** with minimal setup
- Include **test data** or data generators where needed
- Provide **expected outputs** for validation
- Include **performance benchmarks** where relevant

### Validation Process
- Test examples with each major dependency update
- Verify examples work with different input sizes and edge cases
- Check performance characteristics under various conditions
- Validate integration examples with actual external tools

## Integration with Main Project

### Code Reuse
- Examples serve as **prototypes** for main project implementations
- **Extract common patterns** into reusable utilities
- **Maintain alignment** between examples and production code
- **Update examples** when main project patterns evolve

### Documentation Links
- Reference examples from main project documentation
- Link to examples from API documentation
- Include examples in tutorials and guides
- Cross-reference with design documents and plans

## Maintenance

### Regular Updates
- Update examples when dependencies change
- Refresh examples with improved patterns discovered during development
- Archive obsolete examples with migration notes
- Add new examples for emerging patterns and use cases

### Quality Assurance
- Regularly test all examples for functionality
- Review examples for code quality and documentation completeness
- Ensure examples follow current project conventions
- Validate performance characteristics and update benchmarks

Reference code is a living resource that evolves with the project, providing concrete guidance and proven solutions for common development challenges.
