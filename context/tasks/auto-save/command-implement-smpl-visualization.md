# Command: Implement SMPL Visualization Module

## Task Description
Implement SMPL visualization module in `@src/smplx_toolbox/visualization/` using `pyvista` and compatible `pyvistaqt`.

## Requirements
- Create a class that:
  1. Takes unified SMPL model `@src/smplx_toolbox/core/unified_model.py` as input
  2. Adds visualization objects to a pyvista plot (given by user, or created internally)
  3. Supports points, lines, meshes, etc.
  4. Returns pyvista plotter for user to call `show()` and customize further

## Key Benefits
- Easy integration with existing plotting scripts
- User can add custom elements we don't anticipate
- Flexible visualization pipeline

## Referenced Files
- `@src/smplx_toolbox/core/unified_model.py` - unified SMPL model input
- `@src/smplx_toolbox/visualization/` - target implementation directory
- `@.magic-context/instructions/search-proactively.md` - search guidelines
- `@context/tasks/features/smpl-vis` - design document location

## Libraries
- `pyvista` - main 3D visualization library
- `pyvistaqt` - Qt integration for interactive plotting

## Next Steps
1. Create design document in `@context/tasks/features/smpl-vis`
2. Define classes, functions, functionalities, data structures
3. Research pyvista API and best practices