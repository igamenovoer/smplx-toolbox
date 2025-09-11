# Visualization Utilities

Helper functions for building common visualization elements.

## add_axes(plotter, origins, *, scale=0.1, as_arrows=False, line_width=2, labels=False, label_font_size=10, label_prefix=None)

Draw red/green/blue (X/Y/Z) axes at one or more origins.

```python
import numpy as np
import pyvista as pv
from smplx_toolbox.visualization import add_axes

pl = pv.Plotter()
add_axes(pl, origins=(0, 0, 0), scale=0.25, labels=True)

# Multiple axes (e.g., at joint positions)
origins = np.array([[0.0, 0.0, 0.0], [0.5, 0.1, 0.0]])
add_axes(pl, origins=origins, scale=0.05, labels=False)

pl.show()
```

Parameters:

- `origins` — Either a single `(3,)` position or an `(N, 3)` array.
- `scale` — Length of each axis segment.
- `as_arrows` — If `True`, draw arrows instead of line segments.
- `labels` — Add `X/Y/Z` labels near positive ends; kept visible when possible.

## create_polydata_from_vertices_faces(vertices, faces)

Convert vertices/faces arrays to a `pyvista.PolyData`, ensuring correct face
layout and contiguity for VTK.

