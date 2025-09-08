#!/usr/bin/env python
"""
This script demonstrates how to use the UnifiedSmplModel to load different
SMPL family models, manipulate their parameters (pose and shape), and export
the resulting meshes to .obj files.

The script performs the following actions:
1.  Loads a neutral SMPL-X model and exports its default mesh.
2.  Loads a male SMPL-H model and exports its default mesh.
3.  Randomizes the body pose of each model and exports the posed meshes.
4.  Randomizes the shape parameters (betas) of each model and exports the shaped meshes.

All output files are saved to the `tmp/smpl-export/` directory.
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

from smplx_toolbox.core import UnifiedSmplModel, UnifiedSmplInputs

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

# --- Main Processing Function ---
def process_model(model_type: str, gender: str, model_name: str) -> None:
    """Load, process, and export a specified SMPL model."""
    print(f"--- Processing {model_name} ---")

    # 1. Load the base model
    try:
        create_kwargs = dict(
            model_path=str(MODEL_ROOT),
            model_type=model_type,
            gender=gender,
            use_pca=False,
            batch_size=1,
        )
        # For SMPL-H use default PKL models
        base_model = smplx_create(**create_kwargs)
        model = UnifiedSmplModel.from_smpl_model(base_model)
    except Exception as e:
        print(f"Could not load {model_name}: {e}")
        return

    # 2. Export the default T-pose mesh
    default_inputs = UnifiedSmplInputs()
    default_output = model(default_inputs)
    export_obj(default_output.vertices, model.faces, OUTPUT_DIR / f"{model_name}.obj")

    # 3. Randomize pose and export
    posed_inputs = UnifiedSmplInputs(
        pose_body=torch.randn(1, 63) * 1e-2
    )
    posed_output = model(posed_inputs)
    export_obj(posed_output.vertices, model.faces, OUTPUT_DIR / f"{model_name}-posed.obj")

    # 4. Randomize shape and export
    shaped_inputs = UnifiedSmplInputs(
        betas=torch.randn(1, model.num_betas) * 1e-1
    )
    shaped_output = model(shaped_inputs)
    export_obj(shaped_output.vertices, model.faces, OUTPUT_DIR / f"{model_name}-shaped.obj")


if __name__ == "__main__":
    # Process SMPL-X neutral model
    process_model(model_type="smplx", gender="neutral", model_name="smplx-neutral")

    # Process SMPL-H male model
    process_model(model_type="smplh", gender="male", model_name="smplh-male")

    print("\n--- All processing complete. ---")
