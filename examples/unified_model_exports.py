#!/usr/bin/env python
"""
Unified SMPL model export examples.

This script demonstrates the revised `UnifiedSmplModel` and `UnifiedSmplInputs`
APIs by loading different SMPL-family models via `smplx.create`, applying a
few simple parameterizations (pose, hands, face, expressions), and exporting the
resulting meshes to OBJ files under `tmp/smpl-export/`.

What it does
------------
- Loads SMPL (neutral), SMPL-H (male), SMPL-X (neutral) using the official `smplx` API
- Exports default meshes
- Exports lightly posed meshes (body / hands / face depending on model)
- Exports lightly shaped meshes by sampling betas (and expressions for SMPL-X)

Notes
-----
- For simplicity we construct SMPL-H/SMPL-X with `use_pca=False` so hand poses
  are provided in axis-angle (45 dims). The unified adapter can bridge AA<->PCA
  if needed, but AA inputs are recommended for clarity.
"""

import sys
from pathlib import Path
import torch
from torch import Tensor

# Make local smplx implementation importable
_SMPXLIB_ROOT = Path("context/refcode/smplx")
if _SMPXLIB_ROOT.exists():
    sys.path.insert(0, str(_SMPXLIB_ROOT.resolve()))

try:
    from smplx import create as smplx_create
except ImportError as e:
    print(f"Failed to import smplx: {e}")
    sys.exit(1)

from smplx_toolbox.core import UnifiedSmplModel, UnifiedSmplInputs, PoseByKeypoints

# --- Configuration ---
MODEL_ROOT = Path("data/body_models").resolve()
OUTPUT_DIR = Path("tmp/smpl-export").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- OBJ Export Utility ---
def export_obj(vertices: Tensor, faces: Tensor, file_path: Path) -> None:
    """Export a mesh to an .obj file."""
    if vertices.ndim == 3:
        vertices = vertices.squeeze(0)
    if faces.ndim == 3:
        faces = faces.squeeze(0)

    with file_path.open("w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    print(f"Exported mesh to {file_path}")

def _device_dtype(t: Tensor | None, *, fallback_B: int = 1) -> tuple[torch.device, torch.dtype, int]:
    if t is not None:
        return (t.device, t.dtype, t.shape[0])
    return (torch.device("cpu"), torch.float32, fallback_B)


def process_model(model_type: str, gender: str, model_name: str) -> None:
    """Load, process, and export a specified SMPL-family model.

    - Exports default (identity) mesh
    - Exports lightly posed mesh
    - Exports lightly shaped mesh
    - For SMPL-X, also adds light face and expression changes
    """
    print(f"--- Processing {model_name} ---")

    # 1) Load base model via official smplx API
    try:
        create_kwargs = dict(
            model_path=str(MODEL_ROOT),
            model_type=model_type,
            gender=gender,
            use_pca=False,  # prefer AA hands
            batch_size=1,
        )
        base_model = smplx_create(**create_kwargs)
        model = UnifiedSmplModel.from_smpl_model(base_model)
    except Exception as e:
        print(f"Could not load {model_name}: {e}")
        return

    # 2) Default (identity) mesh
    out_default = model(UnifiedSmplInputs())
    export_obj(out_default.vertices.detach().cpu(), model.faces.detach().cpu(), OUTPUT_DIR / f"{model_name}.obj")

    # 3) Lightly posed mesh (use PoseByKeypoints for readability)
    # Minimal body articulation: small elbows twist; add face/hands depending on family
    rnd = torch.randn(1, 3) * 1e-2
    kpts = {
        "root": torch.zeros(1, 3),
        "left_elbow": rnd.clone(),
        "right_elbow": -rnd.clone(),
    }
    if model_type in ("smplh", "smplx"):
        # A little curl on index1 fingers
        kpts["left_index1"] = torch.randn(1, 3) * 1e-2
        kpts["right_index1"] = torch.randn(1, 3) * 1e-2
    if model_type == "smplx":
        # Add slight jaw open and small eye rotations
        kpts["jaw"] = torch.tensor([[1e-2, 0.0, 0.0]])
        kpts["left_eye"] = torch.tensor([[0.0, 1e-2, 0.0]])
        kpts["right_eye"] = torch.tensor([[0.0, -1e-2, 0.0]])

    posed_output = model(PoseByKeypoints.from_kwargs(**kpts))
    export_obj(posed_output.vertices.detach().cpu(), model.faces.detach().cpu(), OUTPUT_DIR / f"{model_name}-posed.obj")

    # 4) Lightly shaped mesh (betas and expressions for SMPL-X)
    betas = torch.randn(1, model.num_betas) * 5e-2
    inputs_shape = {"betas": betas}
    if model_type == "smplx" and model.num_expressions > 0:
        inputs_shape["expression"] = torch.randn(1, model.num_expressions) * 5e-2
    shaped_output = model(UnifiedSmplInputs.from_kwargs(**inputs_shape))
    export_obj(shaped_output.vertices.detach().cpu(), model.faces.detach().cpu(), OUTPUT_DIR / f"{model_name}-shaped.obj")


if __name__ == "__main__":
    # Process SMPL neutral
    process_model(model_type="smpl", gender="neutral", model_name="smpl-neutral")

    # Process SMPL-H male
    process_model(model_type="smplh", gender="male", model_name="smplh-male")

    # Process SMPL-X neutral
    process_model(model_type="smplx", gender="neutral", model_name="smplx-neutral")

    print("\n--- All processing complete. Output directory:", OUTPUT_DIR)
