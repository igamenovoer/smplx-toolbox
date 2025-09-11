# Usage

This section outlines common workflows using the unified model wrapper and thin adapters.

## Unified SMPL Family Wrapper

The wrapper provides a single API for SMPL, SMPL-H, and SMPL-X by normalizing inputs and outputs to a 55-joint SMPL-X layout.

```python
import torch
import smplx
from smplx_toolbox.core import UnifiedSmplModel, UnifiedSmplInputs

# 1) Create base model (e.g., SMPL-X)
base = smplx.create("/path/to/models", model_type="smplx", gender="neutral", use_pca=False, batch_size=1)

# 2) Wrap it
model = UnifiedSmplModel.from_smpl_model(base)

# 3) Prepare segmented inputs (axis-angle)
inputs = UnifiedSmplInputs(
    root_orient=torch.zeros(1, 3),
    pose_body=torch.zeros(1, 63),  # 21 body joints * 3
)

# 4) Forward
out = model(inputs)
print(out.vertices.shape, out.joints.shape)

# 5) Select joints by name (unified 55-joint space)
shoulders = model.select_joints(out.joints, names=["left_shoulder", "right_shoulder"])
```

See the API reference in Core → Unified Model.

## Thin Adapters (SMPL-X / SMPL-H)

Adapters are minimal helpers around an existing `smplx` model to quickly get a `trimesh.Trimesh`.

```python
import smplx
from smplx_toolbox.core import SMPLXModel

base = smplx.create("/path/to/models", model_type="smplx", gender="neutral")
wrapper = SMPLXModel.from_smplx(base)
output = wrapper.base_model(return_verts=True)
mesh = wrapper.to_mesh(output)
mesh.show()
```

For SMPL-H, use `SMPLHModel` and `SMPLHModel.from_smplh(...)`.
