"""Manual smoke script — SMPL-X wireframe mesh + skeleton.

This script constructs an SMPL-X model in neutral pose/shape, wraps it with the
unified model, and uses the SMPLVisualizer to add a wireframe mesh and a
line-based skeleton. Use it for quick manual inspection in terminals or Jupyter.

Notes
-----
- Skips gracefully if `data/body_models` is missing.
- No `if __name__ == '__main__'` guard (per project guidance).
- In Jupyter, `SMPLVisualizer.from_model(..., background=True)` prefers
  `pyvistaqt.BackgroundPlotter` for a non-blocking window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import smplx

pv.set_jupyter_backend('client')  # Prefer client-side rendering in Jupyter

from smplx_toolbox.core.unified_model import UnifiedSmplInputs, UnifiedSmplModel
from smplx_toolbox.visualization import SMPLVisualizer, add_axes

print("[smoke] SMPL-X wireframe + skeleton visualization")

model_path = Path("data/body_models")
if not model_path.exists():
    print("[smoke] data/body_models not found — skipping.")
    raise SystemExit(0)

print(f"[smoke] Using model path: {model_path}")

# Create base SMPL-X model (neutral gender, no PCA for hands)
base_model = smplx.create(
    str(model_path), model_type="smplx", gender="neutral", use_pca=False, batch_size=1
)

# Wrap with unified model
uni_model: UnifiedSmplModel = UnifiedSmplModel.from_smpl_model(base_model)

# Prefer a background plotter for notebooks; users can still supply their own
viz: SMPLVisualizer = SMPLVisualizer.from_model(uni_model)

# Neutral inputs and forward pass
inputs: UnifiedSmplInputs = UnifiedSmplInputs()
output = uni_model.forward(inputs)

# Add wireframe mesh (default color from VisualizationConfig), and skeleton as lines
mesh_actor: Any = viz.add_mesh(output, style="wireframe", opacity=1.0)
skel_actor: Any = viz.add_smpl_skeleton(output, as_lines=True)

# Add joints with labels (joint names)
_joint_actors = viz.add_smpl_joints(output, labels=True, size=0.012, label_font_size=24)

# Add world axes at the origin
plotter = viz.get_plotter()
add_axes(plotter, origins=(0.0, 0.0, 0.0), scale=0.2, labels=True)

# Add per-joint local axes (small) at each joint position
import torch

if isinstance(output.joints, torch.Tensor):
    joints_np = output.joints[0].detach().cpu().numpy()
else:
    joints_np = np.asarray(output.joints)
    if joints_np.ndim == 3:
        joints_np = joints_np[0]

add_axes(plotter, origins=joints_np, scale=0.02, labels=False)

print(f"[smoke] Mesh actor: {type(mesh_actor).__name__ if mesh_actor is not None else None}")
print(f"[smoke] Skeleton actor: {type(skel_actor).__name__ if skel_actor is not None else None}")

print("[smoke] Plotter ready. In Jupyter, call `plotter.show()` to view.")

# Ensure clipping planes include all new props (can help prevent far-plane culling)
plotter.reset_camera_clipping_range()

plotter.show()
