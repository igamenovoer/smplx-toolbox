# SMPL-H Adapter

`SMPLHModel` is a thin wrapper around a user‑provided SMPL‑H model instance. It provides helpers to extract body/hand joints and convert outputs to meshes.

## Example

```python
import smplx
from smplx_toolbox.core import SMPLHModel

base = smplx.create("/path/to/models", model_type="smplh", gender="neutral")
wrapper = SMPLHModel.from_smplh(base)
output = wrapper.base_model(return_verts=True)
mesh = wrapper.to_mesh(output)
```

## API Reference

::: smplx_toolbox.core.smplh_model.SMPLHModel

