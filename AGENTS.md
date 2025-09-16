# Repository Guidelines

## Project Structure & Module Organization
Core source lives in `src/smplx_toolbox/`: `core/` for models, `utils/` for shared helpers, `visualization/` for rendering, and `optimization/` for fitting routines. Tests sit in `tests/` with extended checks in `unittests/`. Dataset assets belong in `data/` (e.g., `data/body_models`). Temporary artifacts go to `tmp/`. CLI and setup helpers are in `scripts/`, while user docs and MkDocs config stay under `docs/`.

## Build, Test, and Development Commands
Run `pixi install -e dev` once per machine to provision the pinned toolchain. Use `python scripts/auto-install.py` to auto-detect CUDA/CPU and configure PyTorch wheels quickly. Execute `pixi run -e dev pytest` for the full suite or scope to a test like `pytest -q -m "not slow"`. Lint and format with `pixi run -e dev ruff check .` and `pixi run -e dev ruff format .`. Type-check with `pixi run -e dev mypy src` before merging.

## FlowMDM Integration
FlowMDM lives in `context/refcode/FlowMDM` and is driven through Pixi wrapper tasks. Install its latest environment via `pixi run flowmdm-install`, then run `pixi run flowmdm-setup` for SpaCy models and legacy `chumpy`.

- Create a symlink once so FlowMDM finds body models: `ln -s ../../../data/body_models context/refcode/FlowMDM/body_models`.
- Generate motion:
  - Babel (with SMPL/SMPL-X export): `pixi run flowmdm-gen-babel` → writes to `tmp/flowmdm-out/babel`.
  - HumanML3D (skeleton + videos): `pixi run flowmdm-gen-humanml` → writes to `tmp/flowmdm-out/humanml3d`.
- Use `pixi run flowmdm-shell` for an interactive prompt, `pixi run flowmdm-exec <cmd>` to run commands from the FlowMDM root, or `pixi run flowmdm-exec-local <cmd>` when you need the FlowMDM Python env while staying in the workspace root.

## Coding Style & Naming Conventions
Target Python 3.11+, four-space indents, max line width 88, and prefer double quotes. Keep functions small, fully typed, and documented per `.magic-context/general/python-coding-guide.md`. Maintain sorted imports via Ruff (isort). Modules and packages use `snake_case`; classes are `CamelCase`; constants remain `UPPER_SNAKE_CASE`.

## Testing Guidelines
Pytest is the standard; mark scenarios with `unit`, `integration`, or `slow` as needed. Name files `test_*.py` or `*_test.py`, classes `Test*`, and functions `test_*`. Ensure tests are deterministic and avoid external I/O unless explicitly guarded. Run focused checks, e.g., `pytest tests/test_unified_model.py::test_unified_model_forward`, before submitting.

## Commit & Pull Request Guidelines
Use Conventional Commit headers (e.g., `feat:`, `fix:`, `docs:`) with descriptive scopes. Every PR should explain intent, link issues, note required assets under `data/`, document reproduction steps, and include screenshots for visual changes. Confirm lint, typing, and tests pass locally; reference the exact commands in the PR description.

## Security & Configuration Tips
Do not commit model binaries or secrets; place secrets in `.env` and keep it private. PyTorch stays outside `pyproject.toml`, so rely on Pixi or follow `docs/pytorch-installation.md`. Before running examples, verify `data/body_models` exists and matches the expected license.
