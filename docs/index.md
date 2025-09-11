# SMPL-X Toolbox

A concise toolbox for working with SMPL family models (SMPL, SMPL-H, SMPL-X) in Python. It focuses on a few high-value utilities:

- Unified SMPL family wrapper with a single, consistent API (55-joint SMPL-X layout)
- Strict, typed input/output containers and a user-friendly per-joint pose spec
- Thin adapters for SMPL-X and SMPL-H to quickly convert model outputs to meshes

This documentation is built with Material for MkDocs.

## Quick Start

- Install the dev environment (includes MkDocs and mkdocstrings):
  - `pixi install -e dev`
  - Or auto-detect GPU/CPU and install: `python scripts/auto-install.py`
- Run the docs locally:
  - `pixi run -e dev mkdocs serve` (or `pixi run -e dev docs-serve`)
- Build the static site:
  - `pixi run -e dev mkdocs build` (or `pixi run -e dev docs-build`)

## Project Highlights

- Unified model API: `UnifiedSmplModel.from_smpl_model(base_model)` wraps any SMPL/SMPL-H/SMPL-X model from `smplx.create` and normalizes inputs/outputs.
- Containers:
  - `UnifiedSmplInputs` – segmented inputs (root/body/hands/face), all AA-space
  - `PoseByKeypoints` – per-joint AA by name, auto-converted to segments
  - `UnifiedSmplOutput` – unified vertices, faces, 55-joint positions, full pose
- Adapters:
  - `SMPLXModel` – SMPL-X only, quick `to_mesh(output)` helper
  - `SMPLHModel` – SMPL-H only, body/hand joint helpers + `to_mesh(output)`

## Code Layout

- `src/smplx_toolbox/core/` – unified model, adapters, containers, constants
- `src/smplx_toolbox/utils/` – utilities (placeholder)
- `src/smplx_toolbox/visualization/` – visualization helpers (PyVista-based)

See Installation and Usage for environment setup and examples.
