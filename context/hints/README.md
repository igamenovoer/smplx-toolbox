# Hints

How-to guides and troubleshooting tips for the SMPL-X Toolbox project.

## Contents

This directory contains practical guidance for common development tasks, known issues, and solutions discovered during development.

## Document Types

### How-To Guides
- **howto-setup-development-environment.md** - Step-by-step environment setup
- **howto-debug-optimization-issues.md** - Troubleshooting optimization convergence problems
- **howto-handle-mesh-loading-errors.md** - Common mesh loading and validation issues
- **howto-configure-visualization.md** - Setting up and customizing 3D visualization
- **howto-export-to-dcc-tools.md** - Exporting models to Blender, Maya, etc.

### Troubleshooting Guides
- **troubleshoot-build-failures.md** - Common build and packaging issues
- **troubleshoot-dependency-conflicts.md** - Resolving package dependency problems
- **troubleshoot-performance-issues.md** - Identifying and fixing performance bottlenecks
- **troubleshoot-import-errors.md** - Python import and module loading problems

### Error Solutions
- **why-optimization-fails.md** - Common reasons for optimization convergence failures
- **why-mesh-rendering-breaks.md** - Visualization and rendering issues
- **why-parameter-validation-errors.md** - Input validation and parameter format issues

## Purpose

These documents serve to:
- **Prevent repeated mistakes** by documenting known issues and solutions
- **Speed up development** by providing step-by-step guidance for common tasks
- **Share knowledge** between team members and AI assistants
- **Document workarounds** for external library limitations or bugs
- **Capture tribal knowledge** that might otherwise be lost

## Naming Convention

Follow these prefixes for clear organization:
- `howto-` - Step-by-step instructions for accomplishing specific tasks
- `troubleshoot-` - Diagnostic guides for investigating problems
- `why-` - Explanations of common error conditions and their causes

## Document Format

Each hint document should include:
- **Problem/Task Description** - Clear statement of what is being addressed
- **Symptoms** - How to recognize the issue (for troubleshooting guides)
- **Solution** - Step-by-step instructions or explanation
- **Prevention** - How to avoid the issue in the future
- **Related Issues** - Links to similar problems or additional context

## Usage Guidelines

### For Developers
- Check hints before starting complex tasks
- Add new hints when discovering solutions to non-obvious problems
- Update existing hints when finding better solutions

### For AI Assistants
- Reference relevant hints when users encounter known issues
- Suggest creating new hints for novel problems and solutions
- Use hints to provide context-aware troubleshooting guidance

## Maintenance

- Keep hints current with the latest project state
- Remove obsolete hints when underlying issues are fixed
- Consolidate related hints to avoid duplication
- Test instructions periodically to ensure they still work
