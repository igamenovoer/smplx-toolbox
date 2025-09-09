# Unified SMPL Family Model

The `UnifiedSmplModel` wraps any SMPL/SMPL-H/SMPL-X model created via `smplx.create`, providing a consistent interface and normalized 55‑joint SMPL‑X outputs.

## Key ideas

- Auto‑detects model type (`smpl`, `smplh`, `smplx`).
- Accepts `UnifiedSmplInputs` or `PoseByKeypoints` and prepares model‑specific kwargs.
- Unifies joints to the first 55 SMPL‑X joints, filling missing ones (configurable as zeros or NaNs).
- Always returns vertices and a flattened full pose vector used for LBS.

## Example

```python
import torch
import smplx
from smplx_toolbox.core import UnifiedSmplModel, PoseByKeypoints

base = smplx.create("/path/to/models", model_type="smplx", gender="neutral", use_pca=False)
umodel = UnifiedSmplModel.from_smpl_model(base)

pose = PoseByKeypoints(left_shoulder=torch.tensor([[0.0, 0.0, -1.5]]))
out = umodel(pose)
print(out.vertices.shape, out.joints.shape, umodel.model_type)
```

## API Reference

::: smplx_toolbox.core.unified_model.UnifiedSmplModel

Related containers used by the unified model:

::: smplx_toolbox.core.containers.UnifiedSmplInputs

::: smplx_toolbox.core.containers.PoseByKeypoints

::: smplx_toolbox.core.containers.UnifiedSmplOutput

