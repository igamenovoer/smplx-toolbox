# Code Review — SMPL Visualization Module (PyVista)

Scope: `src/smplx_toolbox/visualization/` (config.py, utils.py, plotter.py, __init__.py)
Design basis: `context/tasks/features/smpl-vis/design-visualization-module.md`

## Summary

Overall, the implementation aligns well with the design doc and uses PyVista APIs correctly for creating `PolyData`, adding meshes, points, labels, and constructing skeletons as either lines or merged cylinders. Types and batching are handled sensibly. A few small docstring inconsistencies and practical improvement opportunities exist (not blockers).

## Structure vs Design

- Files and exports match the proposed module layout and public interface.
- `SMPLVisualizer` exposes: `from_model`, `add_mesh`, `add_smpl_joints`, `add_smpl_skeleton`, `get_plotter` as designed.
- `VisualizationConfig` provides default colors via properties.
- `utils.py` includes bone connections for SMPL/SMPL-H/SMPL-X, `create_polydata_from_vertices_faces`, and `resolve_joint_selection` per spec.

## Third‑Party APIs Used

- PyVista: `pv.Plotter`, `plotter.add_mesh`, `plotter.add_points`, `plotter.add_point_labels`, `pv.PolyData`, `pv.Cylinder`, `PolyData.lines` property.
- Optional: `pyvistaqt.BackgroundPlotter`.
- NumPy: array conversion/stacking.
- Torch: tensor → NumPy conversion for model outputs.

## API Usage Validation (PyVista)

- PolyData construction
  - Code builds faces as flat array `[n_verts, v0, v1, v2, ...]` before `pv.PolyData(points, faces)`. This matches PyVista requirements. [api-doc]
- Adding meshes/points/labels
  - `plotter.add_mesh(mesh, **kwargs)` is standard. [api-doc]
  - `plotter.add_points(points, point_size=..., color=...)` is correct; point_size is in screen pixels (not world units). The code scales a “size” parameter by 1000 to get a perceptible default (~20 px). [api-doc]
  - `plotter.add_point_labels(points, labels, font_size=..., point_size=..., shape_opacity=...)` is valid; `shape_opacity=0` hides the label background while keeping text. [api-doc]
- Skeleton as lines
  - `polydata = pv.PolyData(joints); polydata.lines = np.hstack([[2, i, j], ...])` is the canonical way to add line cells. Rendering with `add_mesh(..., line_width=...)` is correct. [api-doc]
- Skeleton as cylinders
  - `pv.Cylinder(center=..., direction=..., height=..., radius=...)` returns a cylinder surface; merging multiple cylinder meshes via `mesh.merge(other)` is a supported pattern. [api-doc]
- Background plotting
  - `pyvistaqt.BackgroundPlotter()` for non-blocking windows is correct with graceful fallback to `pv.Plotter` when import fails.

## Correctness & Design Observations

- Batch handling: Vertices/joints accept batched arrays or tensors and consistently select batch 0. Faces are unbatched as expected. Good.
- Joint selection: `resolve_joint_selection` supports names, indices, and keywords (`body`, `hands`, `face`, `all`) while honoring per‑model ranges (SMPL: 24, SMPL-H: 52, SMPL-X: 55). Duplicates removed order‑preservingly. Good.
- Default colors: Pulled from `VisualizationConfig` when not supplied. Good.
- Defensive checks: Plotter/model presence validated; index bounds checked when building skeletons. Good.

## Potential Issues / Edge Cases

- Docstring inconsistency in `VisualizationConfig` example uses `config.m_mesh_color` rather than the public property `mesh_color` (minor documentation nit).
- `add_smpl_joints(size=0.02)`: the parameter is described as a “sphere size”, but rendering uses `add_points` (screen‑space points). Consider clarifying units or enabling `render_points_as_spheres=True` to better match user expectations.
- Lines building: `np.hstack(lines)` is fine; ensure dtype is integer (it is by default from Python ints). Optionally cast to `np.int64` for VTK compatibility on some platforms.
- Cylinder merging: Merging many cylinders per call can be slow for large skeleton sets. An alternative is rendering lines and applying `.tube(radius=...)` once, which is often faster and simpler to manage.
- Mesh faces dtype: Faces are cast to `np.int32`; PyVista/VTK generally accept this, but some platforms prefer `np.int64`. If users encounter warnings, consider making dtype configurable or upcasting as needed.

## Suggestions (Non‑blocking)

- Joints visualization
  - Add optional `render_points_as_spheres: bool = True` default when calling `add_points` for better appearance and alignment with “sphere size” verbiage.
  - Expose `point_size` directly or compute from `size` with a clearer mapping (e.g., `point_size=int(round(size_px))`) and document that units are pixels.
- Skeleton performance
  - Provide a code path that constructs a line set and uses `.tube(radius)` once for a cylinder‑like look with fewer merges.
- Mesh creation
  - Use `np.ascontiguousarray` on flattened faces before passing to `pv.PolyData` to avoid hidden copies.
- Typing
  - Where possible, annotate return types as `vtk.vtkActor | None` (or PyVista’s `Actor` proxy) instead of `Any` to improve mypy coverage, if feasible within project constraints.
- Docs
  - Fix the `VisualizationConfig` example property name.
  - Note optional dependency on `pyvistaqt` in user‑facing docs/examples.

## Quick Self‑Check Scenarios

- Mesh only: Triangular faces `(F, 3)`; verify `add_mesh` shows surface with default color.
- Joints selection: `['body']`, `['hands']`, `['face']` across SMPL/SMPL-H/SMPL-X; confirm indices resolved and points render with labels.
- Skeleton as lines/cylinders: Confirm both paths render without index errors and respect `line_width`/`radius`.
- Tensor inputs: Pass `torch.Tensor` outputs for `vertices`, `joints`, `faces` to verify detaching and CPU conversion paths.

## References

api-doc (Context7):
- /pyvista/pyvista — Plotting and core mesh API

online-examples (PyVista docs):
- PolyData faces format and construction: https://github.com/pyvista/pyvista/blob/main/doc/source/user-guide/data_model.rst#_snippet_0
- Plotter.add_mesh basic usage: https://github.com/pyvista/pyvista/blob/main/doc/source/user-guide/simple.rst#_snippet_7
- Point clouds and add_points: https://github.com/pyvista/pyvista/blob/main/doc/source/user-guide/what-is-a-mesh.rst#_snippet_1
- Point labels: https://github.com/pyvista/pyvista/blob/main/doc/source/user-guide/data_model.rst#_snippet_22
- Building lines in PolyData: examples using `PolyData.lines` and plotting lines via `add_mesh`: https://github.com/pyvista/pyvista/blob/main/doc/source/user-guide/data_model.rst#_snippet_27
- Cylinder primitive: general mesh primitives usage (Cylinder available via `pv.Cylinder`): https://docs.pyvista.org/api/utilities/_autosummary/pyvista.Cylinder.html (general reference)

— End of report —

