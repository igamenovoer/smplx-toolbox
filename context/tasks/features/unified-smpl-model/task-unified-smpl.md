# Unified SMPL Model — Revision Plan

This plan revises the unified SMPL family model to cleanly support SMPL, SMPL-H (including MANO-based variants), and SMPL-X using models loaded via the `smplx` library. The scope is limited to API/behavior changes and tests; no implementation is included here.

## Goals
- Keep a single high-level API that runs over SMPL/SMPL-H/SMPL-X (PKL/NPZ variants).
- Accept a superset of parameters via `UnifiedSmplInputs` and ignore unsupported ones per model.
- Provide explicit input conversion functions:
  - `to_smpl_inputs()`
  - `to_smplh_inputs(with_hand_shape: bool)`
  - `to_smplx_inputs()`
- Use `PoseByKeypoints` for pose construction and map to the correct per-model inputs.
- Unify outputs to a consistent joint set (55 SMPL-X joints) and expose faces/pose.
- Preserve device/dtype and batch semantics; improve error and warning handling.

## Non-Goals
- Parameter-space conversion between model families (e.g., SMPL → SMPL-X fitting). See `context/hints/smplx-kb/about-smplx-transfer-model.md` for that pipeline.
- Changing how `smplx` models are constructed. We assume they are created externally via `smplx.create`.

## Current State (Quick Audit)
- `UnifiedSmplModel` exists and handles detection, basic normalization, and unification.
- `UnifiedSmplInputs` and `PoseByKeypoints` are present; `UnifiedSmplInputs.from_keypoint_pose` exists.
- Hand PCA handling in normalization is incomplete: it assumes the correct hand pose dimensionality without converting from AA↔PCA.
- Body pose uses 63-DoF (21×3) and pads to 69 for SMPL, but logic and documentation need to be made explicit.
- Tests use mock models; they don’t exercise real `smplx` hand PCA/expr paths.

## Design Decisions
- Prefer `use_pca=False` for SMPL-H/SMPL-X at model construction for simplicity. If `use_pca=True` is detected, provide a best-effort conversion (see below) and warn.
- Keep `UnifiedSmplInputs` focused on a single unified shape vector (`betas`) plus optional `expression`. For SMPL-H MANO shape support, expose an optional `hand_betas` field (see below) and only pass it when `with_hand_shape=True` and the underlying model supports it.
- Maintain `PoseByKeypoints` as the primary UX for composing poses by joint names, mapped consistently across model families.

## API Changes (Proposed)
1) `UnifiedSmplInputs` additions
   - New optional fields:
     - `hand_betas: Tensor | None` (B, H) — optional MANO hand shape for SMPL-H MANO variant.
     - `use_hand_pca: bool | None` and `num_hand_pca_comps: int | None` — optional hints for conversion when the base model uses PCA.
   - New conversion methods:
     - `to_smpl_inputs(self) -> dict[str, Tensor]`
     - `to_smplh_inputs(self, with_hand_shape: bool) -> dict[str, Tensor]`
     - `to_smplx_inputs(self) -> dict[str, Tensor]`
   - Validation helpers updated to allow extra fields that may be ignored by some models.

2) `UnifiedSmplModel` changes
   - Refactor input normalization to route through `UnifiedSmplInputs.to_*` methods based on detected `model_type`.
   - Make SMPL body-pose padding explicit: 63-DoF body pose (21 joints) → pad 2×3 AA zeros for `left_hand` and `right_hand` to reach 69-DoF expected by SMPL.
   - Add a guarded hand PCA bridge:
     - If base model `use_pca=True` and AA(45) hand poses are provided, compute PCA coefficients using a pseudo-inverse of `left/right_hand_components` and subtract/add the hand means appropriately. Warn for potential minor discrepancies.
     - If `use_pca=False`, pass AA(45) directly.
   - Expressions: infer `num_expression_coeffs` from the model and pad/truncate `expression` accordingly for SMPL-X.
   - Keep output unification to 55-joint SMPL-X; improve missing joint fill policy (`nan`/`zero`).

3) Error, warnings, and typing
   - Strengthen shape checks and consistency across fields.
   - Emit actionable warnings on ignored parameters, PCA fallback, and partial hands/eyes.
   - Ensure full type coverage (mypy strict) and small, testable functions.

## Conversion Specifications

Reference: `context/hints/smplx-kb/compare-smpl-models.md`, `compare-smpl-skeleton.md`, `compare-smpl-shape.md`.

- Shared keys across families:
  - `global_orient: (B, 3)` AA
  - `body_pose: (B, 63)` AA for 21 joints; SMPL requires padding to `(B, 69)`
  - `betas: (B, N_b)` (pad/truncate per model’s `num_betas`)
  - `transl: (B, 3)` when provided

- Hands:
  - Unified input: left/right hand as AA 45 dims each (15 joints × 3) when available.
  - SMPL-H/SMPL-X, `use_pca=False`: pass AA(45) directly.
  - SMPL-H/SMPL-X, `use_pca=True`: convert AA→PCA using pseudo-inverse:
    - Let `C ∈ R^{K×45}` be `hands_components{l,r}`; AA is `A ∈ R^{B×45}`.
    - Solve `A ≈ P @ C` for `P ∈ R^{B×K}` via `P = A @ C^T @ (C @ C^T)^{-1}` or `torch.linalg.lstsq` with regularization; adjust for means if needed.
    - Warn when performing conversion; allow override via `use_hand_pca` hints.

- Expressions (SMPL-X only):
  - `expression: (B, N_e)`; derive `N_e` from model (`num_expression_coeffs`). Pad/truncate as needed.

- SMPL-H MANO shape (optional):
  - If `with_hand_shape=True` and `hand_betas` provided, forward to model only when the underlying variant supports it. If unsupported, warn and ignore.

## Detailed Tasks

1) `UnifiedSmplInputs`
   - Add fields: `hand_betas`, `use_hand_pca`, `num_hand_pca_comps`.
   - Implement `to_smpl_inputs()`:
     - Keys: `betas`, `global_orient`, `body_pose(69)`, `transl`, `return_verts=True`.
     - Pad body pose by 2×3 AA zeros at the end for `left_hand`/`right_hand` (SMPL-only body joints).
   - Implement `to_smplh_inputs(with_hand_shape)`:
     - Keys: `betas`, `global_orient`, `body_pose(63)`, `left_hand_pose`, `right_hand_pose`, `transl`, `return_verts=True`.
     - If base model `use_pca=True`, convert AA→PCA using the model’s components via helper on the wrapper (see (2)).
     - If `with_hand_shape` and `hand_betas` present and supported, include them (conditional pass-through; otherwise warn/ignore).
   - Implement `to_smplx_inputs()`:
     - Keys: `betas`, `expression`, `global_orient`, `body_pose(63)`, `jaw_pose`, `leye_pose`, `reye_pose`, `left_hand_pose`, `right_hand_pose`, `transl`, `return_verts=True`.
     - Handle hand PCA conversion if needed (as above).
   - Update validation: allow ignored fields depending on `model_type`.

2) `UnifiedSmplModel`
   - Detect model type and flags: `use_pca`, `num_pca_comps`, `num_betas`, `num_expression_coeffs`.
   - Replace `_normalize_inputs` body with switch:
     - `smpl` → `inputs.to_smpl_inputs()` + dtype/device alignment + betas/expression shaping.
     - `smplh` → `inputs.to_smplh_inputs(with_hand_shape=...)` including PCA handling.
     - `smplx` → `inputs.to_smplx_inputs()` including PCA/expr shaping.
   - Extract and cache hand PCA buffers for conversion (`left/right_hand_components`, `left/right_hand_mean`) when present.
   - Keep `_compute_full_pose` to expose a unified flat AA pose, expanding PCA to AA for reporting when base model uses PCA.
   - Keep `_unify_joints` mapping; verify indices against `SMPL_JOINT_NAMES`, `SMPLH_JOINT_NAMES`, `SMPLX_JOINT_NAMES`.
   - Improve missing-joint filling (nan/zero) and expose `extras` metadata including `joint_mapping`, `raw_joint_names`, and PCA flags.

3) Utilities
   - Add small helpers for AA↔PCA conversion with numerically stable pseudo-inverse and mean handling.
   - Centralize pad/truncate logic for `betas` and `expression` with explicit warnings when truncating.

4) Tests
   - Unit tests for conversion methods without real `smplx` dependency (shape-only validation, warnings, padding, ignored fields).
   - Integration tests with real `smplx` models:
     - SMPL (PKL), SMPL-H (PKL/NPZ, `use_pca=False`), SMPL-X (NPZ/PKL), optional SMPL-H with `use_pca=True` if available.
     - Verify forward succeeds with AA hand inputs under both PCA and non-PCA models; check vertex/joint shapes and no device/dtype mismatches.
     - Validate expression shaping for SMPL-X (pad/truncate to model’s `num_expression_coeffs`).
   - Back-compat tests for current public calls in `tests/test_unified_model.py`.

5) Documentation
   - Update module docstrings and README sections to describe the new conversion methods.
   - Document hand PCA handling and the recommendation to create models with `use_pca=False` for simplicity; outline the fallback conversion and its caveats.
   - Add a short example showing `PoseByKeypoints` → `UnifiedSmplInputs` → `UnifiedSmplModel` forward for each family.

6) Quality
   - Ensure `ruff` and `mypy` compliance; keep functions small and fully typed.
   - Maintain determinism in tests; avoid external I/O unless guarded with markers.

## Acceptance Criteria
- `UnifiedSmplInputs` exposes `to_smpl_inputs`, `to_smplh_inputs(with_hand_shape)`, `to_smplx_inputs` and passes strict shape checks.
- `UnifiedSmplModel` forwards via those conversion methods; SMPL body pose padding is correct and documented.
- Hand PCA models run from AA(45) inputs via internal conversion (with warnings) or natively when `use_pca=False`.
- SMPL-X expressions are padded/truncated to the model’s `num_expression_coeffs`.
- Output joints are always unified to 55 SMPL-X joints with consistent missing-joint policy and metadata about mappings.
- Tests cover conversion logic, warnings, padding, and real `smplx` paths where feasible.

## Risks / Open Questions
- SMPL-H MANO hand shape inputs vary by source; availability in all distributions isn’t guaranteed. We’ll gate by capability detection and warn otherwise.
- PCA components’ orthogonality assumptions may vary; we’ll use `lstsq`/`pinv` and warn about approximation when projecting AA→PCA.
- Some legacy PKL models expose only 10 betas; we will truncate/pad with explicit warnings and document it.

## Implementation Outline (Phased)
1) Add fields + conversion methods to `UnifiedSmplInputs`; extend validation and docs.
2) Implement AA↔PCA helpers; read components/means from base model when available.
3) Refactor `UnifiedSmplModel._normalize_inputs` to delegate to `to_*` methods; wire PCA/expr shaping; keep `_compute_full_pose` and `_unify_joints`.
4) Update/extend tests (unit + optional integration) and docs.
5) Polish warnings, typing, and ruff/mypy compliance.

## References
- Models and relationships: `context/hints/smplx-kb/compare-smpl-models.md`, `compare-smpl-skeleton.md`, `compare-smpl-shape.md`.
- Transfer pipeline: `context/hints/smplx-kb/about-smplx-transfer-model.md`.
- Current implementation: `src/smplx_toolbox/core/unified_model.py` and `containers.py`.
## Note: PoseByKeypoints Status (Outdated)

- PoseByKeypoints is no longer part of the public API. Code now accepts `UnifiedSmplInputs` only.
- For optional editing/inspection of packed poses, use `NamedPose` with `ModelType`.
- Joint name→index mappings are sourced from `core.constants` and follow the unified SMPL‑X 55‑joint scheme.
- This design doc is retained for archival purposes; references to PoseByKeypoints reflect an older iteration.
