# Visualization

This section covers the visualization utilities built on top of PyVista for
SMPL/SMPL-H/SMPL-X models.

The main entrypoint is `SMPLVisualizer`, which adds meshes, joints, and
skeletal connections to a PyVista plotter. A small set of utilities are also
provided for creating `PolyData` and drawing reference axes at one or more
origins.

## Components

- `SMPLVisualizer` — Add model mesh, joints (with optional labels), and
  skeleton (as lines or merged cylinders) to a PyVista plotter.
- `add_axes(plotter, origins, ...)` — Draw XYZ axes at the world origin or
  at any list of positions (e.g., per-joint axes).
- `create_polydata_from_vertices_faces(vertices, faces)` — Convert SMPL-style
  vertices/faces to `pyvista.PolyData`.

## Requirements

- PyVista installed (the toolbox defaults to `pyvista.Plotter`).
- To use a background (non-blocking) plotter in notebooks, install
  `pyvistaqt` and pass `background=True` to `SMPLVisualizer.from_model(...)`.

