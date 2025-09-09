# SMPL-X Adapter

`SMPLXModel` is a thin wrapper around a user‑provided SMPL‑X model instance. It doesn’t re‑expose the `smplx` API; it just helps convert outputs to meshes.

## Example

```python
import smplx
from smplx_toolbox.core import SMPLXModel

base = smplx.create("/path/to/models", model_type="smplx", gender="neutral")
wrapper = SMPLXModel.from_smplx(base)
output = wrapper.base_model(return_verts=True)
mesh = wrapper.to_mesh(output)
```

## API Reference

::: smplx_toolbox.core.smplx_model.SMPLXModel

