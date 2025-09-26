# Repository Guidelines

## Project Structure & Module Organization
- Core source: `src/smplx_toolbox/`
  - `core/` models, `utils/` shared helpers, `visualization/` rendering, `optimization/` fitting.
- Tests: `tests/` (unit/integration) and `unittests/` (extended checks).
- Data assets: `data/` (e.g., `data/body_models`). Temp artifacts: `tmp/`.
- CLI/setup: `scripts/`. Docs and MkDocs config: `docs/`.

## Build, Test, and Development Commands
- Provision toolchain (once per machine): `pixi install -e dev`.
- Quick PyTorch setup (CUDA/CPU autodetect): `python scripts/auto-install.py`.
- Run tests: `pixi run -e dev pytest` or focused: `pytest -q -m "not slow"`.
- Lint/format: `pixi run -e dev ruff check .` and `pixi run -e dev ruff format .`.
- Type-check: `pixi run -e dev mypy src`.

## FlowMDM Integration
- Install env: `pixi run flowmdm-install` → then `pixi run flowmdm-setup` (SpaCy, chumpy).
- One-time symlink so models resolve: `ln -s ../../../data/body_models context/refcode/FlowMDM/body_models`.
- Generate motion: `pixi run flowmdm-gen-babel` → `tmp/flowmdm-out/babel`; `pixi run flowmdm-gen-humanml` → `tmp/flowmdm-out/humanml3d`.
- Use shell/exec: `pixi run flowmdm-shell`, `pixi run flowmdm-exec-local <cmd>`.

## Coding Style & Naming Conventions
- Python 3.11+, 4-space indents, max line width 88, prefer double quotes.
- Keep functions small, fully typed; maintain sorted imports (Ruff/isort).
- Naming: snake_case modules/packages, CamelCase classes, UPPER_SNAKE_CASE constants.

## Testing Guidelines
- Framework: Pytest; mark `unit`, `integration`, `slow` as needed.
- Names: files `test_*.py` or `*_test.py`; classes `Test*`; functions `test_*`.
- Tests must be deterministic; avoid external I/O unless explicitly guarded.
- Example: `pytest tests/test_unified_model.py::test_unified_model_forward`.

## Commit & Pull Request Guidelines
- Conventional Commits (e.g., `feat:`, `fix:`, `docs:`) with descriptive scopes.
- PRs: explain intent, link issues, note required assets under `data/`, add repro steps; include screenshots for visual changes.
- Verify lint, typing, and tests pass locally; reference commands used in the PR description.

## Security & Configuration Tips
- Do not commit model binaries or secrets; use `.env` for private config.
- PyTorch is not in `pyproject.toml`; install via Pixi or see `docs/pytorch-installation.md`.
- Before running examples, ensure `data/body_models` exists and license terms are respected.

