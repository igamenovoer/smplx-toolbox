# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/smplx_toolbox/` (core models in `core/`, helpers in `utils/`, `visualization/`, `optimization/`).
- Tests: primary suite in `tests/` (fast/unit + examples), additional rigorous cases in `unittests/`.
- Assets/data: `data/` (e.g., `data/body_models` for SMPL/SMPL-X files), temp outputs in `tmp/`.
- Scripts: `scripts/` (CLI `scripts/smplx-toolbox`, env helper `auto-install.py`).
- Docs: `docs/` with MkDocs config `mkdocs.yml`.

## Build, Test, and Development Commands
- Environment: `pixi install -e dev` (installs deps incl. tools; PyTorch index set by platform). 
- Quick GPU/CPU setup: `python scripts/auto-install.py` (detects NVIDIA GPU and installs matching env).
- Run tests: `pixi run -e dev pytest` (coverage to `htmlcov/` and `coverage.xml`). Examples:
  - `pytest -q -m "not slow"`
  - `pytest tests/test_unified_model.py::test_unified_model_forward`
- Lint/format: `pixi run -e dev ruff check .` and `pixi run -e dev ruff format .`.
- Type check: `pixi run -e dev mypy src`.
- CLI help: `pixi run -e dev python scripts/smplx-toolbox --help`.

## Coding Style & Naming Conventions
- Python 3.11+, 4-space indents (`.editorconfig`). Line length 88 (ruff).
- Use double quotes; import order via ruff-isort. Keep functions small and typed (mypy strict).
- Modules/packages: `snake_case`; classes: `CamelCase`; constants: `UPPER_SNAKE_CASE`.

## Testing Guidelines
- Framework: pytest. Locations: `tests/` (default), optional `unittests/`.
- Naming: files `test_*.py` or `*_test.py`; classes `Test*`; functions `test_*`.
- Markers: `unit`, `integration`, `slow` (e.g., `-m "not slow"`).
- Aim for coverage on changed code; ensure determinism and no external I/O without guards.

## Commit & Pull Request Guidelines
- Commits: Conventional style preferred: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`.
- PRs: include a clear summary, linked issues, test coverage for changes, and screenshots for visual output when relevant. Note any `data/` paths used.

## Security & Configuration Tips
- Do not commit model assets or large binaries; keep secrets in `.env` (never commit).
- PyTorch is not a runtime dependency in `pyproject.toml`; install via pixi or follow `docs/pytorch-installation.md`.
- Validate `data/body_models` exists before running examples or tests that need it.
