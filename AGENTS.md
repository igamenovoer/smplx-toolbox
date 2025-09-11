# Repository Guidelines

## Project Structure & Module Organization
- Source code: `src/smplx_toolbox/`
  - Core models: `core/`; utilities: `utils/`; plus `visualization/` and `optimization/`.
- Tests: main suite `tests/`; extra rigorous cases `unittests/`.
- Data/assets: `data/` (e.g., `data/body_models` for SMPL/SMPL-X); temp outputs `tmp/`.
- Scripts: `scripts/` (CLI entry `scripts/smplx-toolbox`, env helper `scripts/auto-install.py`).
- Docs: `docs/` with MkDocs config `mkdocs.yml`.

## Build, Test, and Development Commands
- Install dev env: `pixi install -e dev` (tools + pinned PyTorch per platform).
- Quick CUDA/CPU setup: `python scripts/auto-install.py` (detects NVIDIA GPU, configures env).
- Run tests: `pixi run -e dev pytest` (examples: `pytest -q -m "not slow"`, `pytest tests/test_unified_model.py::test_unified_model_forward`).
- Lint/format: `pixi run -e dev ruff check .` and `pixi run -e dev ruff format .`.
- Type check: `pixi run -e dev mypy src`.
- CLI help: `pixi run -e dev python scripts/smplx-toolbox --help`.

## Coding Style & Naming Conventions
- Python 3.11+, 4-space indents, max line length 88, prefer double quotes.
- Keep functions small and fully typed (mypy strict). Imports sorted via ruff-isort.
- Naming: modules/packages `snake_case`; classes `CamelCase`; constants `UPPER_SNAKE_CASE`.

## Testing Guidelines
- Framework: pytest. Test locations: `tests/` (default) and `unittests/`.
- Naming: files `test_*.py` or `*_test.py`; classes `Test*`; functions `test_*`.
- Markers: `unit`, `integration`, `slow` (e.g., `-m "not slow"`). Ensure determinism; avoid external I/O unless explicitly guarded.

## Commit & Pull Request Guidelines
- Commits: Conventional (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`) with clear scope.
- PRs: include description, linked issues, tests for changes, and screenshots for visual output. Note any `data/` paths used and how to reproduce locally.

## Security & Configuration Tips
- Do not commit model assets or large binaries. Keep secrets in `.env` (never commit).
- PyTorch is not a runtime dependency in `pyproject.toml`; install via Pixi or follow `docs/pytorch-installation.md`.
- Validate `data/body_models` exists before running examples or tests that need it.

## Additional Guides
- Python coding patterns and expectations: `.magic-context/general/python-coding-guide.md` (covers docstrings, imports, properties, and factory patterns).
