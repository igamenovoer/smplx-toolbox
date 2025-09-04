# Project Configuration Cleanup - Remove Viz and Docs Features

## HEADER
- **Purpose**: Remove unnecessary visualization and documentation features from project configuration
- **Status**: Completed
- **Date**: 2025-09-04
- **Dependencies**: Previous pixi migration to pyproject.toml
- **Target**: AI assistants and developers

## Objective

Simplify the project configuration by removing visualization (`viz`) and documentation (`docs`) features and environments, focusing the project on core functionality and development tooling only.

## Rationale

The `viz` and `docs` features were determined to be unnecessary for the current project scope:

- **Visualization dependencies** (matplotlib, plotly, open3d, pyglet) can be added on-demand when needed
- **Documentation dependencies** (mkdocs, mkdocs-material, mkdocstrings) are not immediately required for core development
- **Simpler configuration** reduces complexity and maintenance overhead
- **Faster environment resolution** with fewer dependencies to solve

## Changes Made

### Removed PyPI Optional Dependencies
```toml
# Removed from [project.optional-dependencies]
docs = [
    "mkdocs>=1.4.0",
    "mkdocs-material>=8.0.0", 
    "mkdocstrings[python]>=0.19.0",
]
viz = [
    "matplotlib>=3.5.0",
    "plotly>=5.0.0",
    "open3d>=0.13.0",
    "pyglet>=1.5.0",
]

# Updated "all" extra to only include dev
all = ["smplx-toolbox[dev]"]  # Previously: ["smplx-toolbox[dev,docs,viz]"]
```

### Removed Pixi Features
```toml
# Removed entire sections
[tool.pixi.feature.docs.dependencies]
[tool.pixi.feature.viz.dependencies]
```

### Removed Pixi Environments
```toml
# Removed from [tool.pixi.environments]
docs = {features = ["docs"], solve-group = "default"}
viz = {features = ["viz"], solve-group = "default"}
all = {features = ["dev", "docs", "viz"], solve-group = "default"}
```

### Removed Documentation Tasks
```toml
# Removed tasks
docs-serve = "mkdocs serve"
docs-build = "mkdocs build"
docs-deploy = "mkdocs gh-deploy"
```

## Results

### Simplified Environment Structure
After cleanup, only essential environments remain:

**Default Environment** (10 dependencies):
- Core project dependencies for basic functionality
- Python, numpy, scipy, pytorch, trimesh, opencv, pillow, tqdm, pyyaml, click

**Dev Environment** (17 dependencies):
- Default dependencies + development tools
- Additional: pytest, pytest-cov, black, isort, flake8, mypy, pre-commit

### Streamlined Task List
Available tasks reduced to essential development operations:
- **Code Quality**: format, format-check, lint, sort-imports, sort-imports-check, type-check
- **Testing**: test, test-cov
- **Development**: install-dev, pre-commit-install, pre-commit-run
- **Build**: build, clean
- **Utility**: qa (quality assurance suite), upgrade-deps

### Performance Benefits
- **Faster environment resolution** - fewer dependencies to solve
- **Reduced disk usage** - no visualization or documentation packages
- **Simpler dependency management** - fewer potential conflicts
- **Cleaner development workflow** - focused on core functionality

## Verification

### Environment Check
```bash
pixi info
# Shows only 2 environments: default (10 deps) and dev (17 deps)
# Previously: 5 environments with up to 23 dependencies
```

### Task Verification
```bash
pixi task list
# Shows 15 essential tasks
# Documentation tasks (docs-serve, docs-build, docs-deploy) removed
```

### Functionality Preserved
- ✅ All core development tasks available
- ✅ Testing framework intact
- ✅ Code quality tools preserved
- ✅ Build and packaging tasks maintained

## Future Considerations

### If Visualization Features Needed
Can be re-added on-demand:
```bash
pip install matplotlib plotly open3d pyglet
# Or add back to pyproject.toml optional dependencies when needed
```

### If Documentation Tools Needed
Can be installed separately:
```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
# Or add back documentation feature when ready for formal documentation
```

### Modular Approach Benefits
- **Add features when needed** rather than carrying unused dependencies
- **Easier onboarding** for new contributors with simpler setup
- **Clearer project focus** on core SMPL-X functionality
- **Reduced cognitive overhead** in configuration management

## Impact Assessment

### Positive Impacts
- ✅ **Simplified setup** - fewer dependencies to install and manage
- ✅ **Faster CI/CD** - smaller environments to build and cache
- ✅ **Clearer project scope** - focus on core functionality
- ✅ **Reduced maintenance** - fewer dependencies to keep updated

### Minimal Risk
- ⚠️ **Future documentation work** will require adding back dependencies
- ⚠️ **Visualization features** need manual installation when needed
- ✅ **Easy to restore** - configuration can be added back when required

## Success Criteria

✅ **Environment Simplification** - Reduced from 5 to 2 environments
✅ **Dependency Reduction** - Maximum dependencies reduced from 23 to 17
✅ **Task Streamlining** - Removed 3 documentation tasks
✅ **Functionality Preservation** - All essential development tasks retained
✅ **Configuration Cleanup** - Cleaner, more focused pyproject.toml

The cleanup successfully simplified the project configuration while maintaining all essential development capabilities. The project now has a cleaner, more focused setup that can be extended when specific features (visualization, documentation) are actually needed.
