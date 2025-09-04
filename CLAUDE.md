# AI Assistant Guide for SMPL-X Toolbox

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

## Development Environment

This project uses **Pixi** for environment management. Key commands:

```bash
pixi install          # Install dependencies
pixi run install-dev  # Install in development mode
pixi run test         # Run test suite
pixi run qa           # Run quality assurance checks
pixi run docs-serve   # Start documentation server
```

## AI Assistant Roles

When working on this project, consider adopting specific roles based on the task:

- **Backend Developer** - Python development, API design, optimization algorithms
- **Computer Vision Specialist** - 3D modeling, mesh processing, parameter fitting
- **Documentation Writer** - Technical writing, API docs, user guides
- **Test Engineer** - Quality assurance, testing strategies, validation

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
