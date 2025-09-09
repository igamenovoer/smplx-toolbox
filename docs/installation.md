# Installation

The project uses Pixi to manage environments and dev tools.

## Quick setup

- Full dev toolchain (PyTorch pinned per platform, docs, tests):
  - `pixi install -e dev`
- Auto-detect NVIDIA GPU and install a suitable env:
  - `python scripts/auto-install.py`

Notes
- PyTorch is not declared in `pyproject.toml` runtime deps; Pixi installs it per platform.
- Model assets are not bundled. Place SMPL/SMPL-X assets under `data/body_models`.

## Validate setup

- Run tests: `pixi run -e dev pytest -q -m "not slow"`
- Lint/format: `pixi run -e dev ruff check .` and `pixi run -e dev ruff format .`
- Type check: `pixi run -e dev mypy src`

## Docs commands (Material for MkDocs)

- Serve locally: `pixi run -e dev mkdocs serve` (or `pixi run -e dev docs-serve`)
- Build static site: `pixi run -e dev mkdocs build` (or `pixi run -e dev docs-build`)

