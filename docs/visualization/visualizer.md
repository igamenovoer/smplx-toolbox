# SMPLVisualizer

The `SMPLVisualizer` class provides a convenient interface to add SMPL family
visualizations to a PyVista plotter.

## Quick Start

```python
import smplx
import pyvista as pv
from smplx_toolbox.core.unified_model import UnifiedSmplModel, UnifiedSmplInputs
from smplx_toolbox.visualization import SMPLVisualizer

# 1) Load base SMPL-X (or SMPL-H/SMPL) model
base = smplx.create("data/body_models", model_type="smplx", gender="neutral", use_pca=False)

# 2) Wrap with unified model
model = UnifiedSmplModel.from_smpl_model(base)

# 3) Create visualizer (defaults to pyvista.Plotter)
viz = SMPLVisualizer.from_model(model)

# 4) Run a neutral forward pass
out = model.forward(UnifiedSmplInputs())

# 5) Add visual elements
viz.add_mesh(out, style="wireframe", opacity=1.0)
viz.add_smpl_joints(out, labels=True)
viz.add_smpl_skeleton(out, as_lines=True)

# 6) Show
viz.get_plotter().show()
```

## Construction

```python
viz = SMPLVisualizer.from_model(model, plotter=None, background=False)
```

- `plotter=None` — If omitted, a new `pyvista.Plotter` is created.
- `background=False` — When `True`, and if `pyvistaqt` is installed, a
  `BackgroundPlotter` is created. Otherwise falls back to `pyvista.Plotter`.

## Methods

### add_mesh(output=None, *, style=None, color=None, opacity=None, **kwargs)

Add the model mesh to the plotter.

- `style` — One of `{"surface", "wireframe", "points"}`.
- `color` — Tuple of floats `(r, g, b)` in `[0, 1]` or a `(3,)` NumPy array.
- `opacity` — Float in `[0, 1]`.
- Additional keyword args are passed through to `plotter.add_mesh`.

If `output` is `None`, a neutral output is generated from the current model.

### add_smpl_joints(output=None, joints=None, size=0.02, color=None, labels=False, label_font_size=12, **kwargs)

Add selected joints to the plotter as points (rendered as spheres by default).

- `joints` — `None` (all), names/indices, or keywords like `"body"`, `"hands"`,
  `"face"`, `"all"`.
- `labels` — When `True`, joint names are shown and kept visible when possible.
- `size` — Scales `point_size` in pixels for visibility.

### add_smpl_skeleton(output=None, connections=None, radius=0.005, color=None, as_lines=False, **kwargs)

Add the skeleton as either fast line segments (`as_lines=True`) or merged
cylinders (`as_lines=False`).

## Notes

- The visualizer does not call `show()` for you; use `viz.get_plotter().show()`
  after adding your custom overlays.
- Label visibility is improved by requesting `always_visible=True` when the
  running PyVista version supports it.

