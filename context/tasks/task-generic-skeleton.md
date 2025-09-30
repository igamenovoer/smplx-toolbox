# Task: Generic NetworkX‑based Skeleton (Joint Graph)

Goal
- Replace the MJCF‑backed `GenericSkeleton` with a lightweight, framework‑agnostic implementation built on `networkx` that models a human skeleton purely as a directed joint graph with per‑joint 4×4 local transforms.

Why
- MJCF/URDF bring heavy dependencies and constraints (joint types/DoFs) that are unnecessary for human retargeting workflows here.
- Human pipelines (SMPL/SMPL‑X/T2M) primarily work with joint names and local transforms; link semantics are rarely needed.
- A simple joint graph with explicit transforms is easier to reason about, serialize, and integrate for retargeting.

Design Overview
- Core idea: a DAG (`nx.DiGraph`) whose nodes are joints. Each node holds a `JointNode` attrs object containing:
  - `name: str` — joint name (node key)
  - `T_local: Transmat_4x4` — 4×4 homogeneous transform from parent joint to this joint
  - Optional metadata: `index: int | None`, `tags: set[str]`, `attrs: dict[str, Any]` (future‑proof)
- Global/world transform is computed recursively by FK:
  - `T_global[root] = T_local[root]` (root’s parent is world)
  - `T_global[child] = T_global[parent] @ T_local[child]`
- The graph stores only joints. Link/body nodes are not modeled; this matches community practice where “joint names” are canonical (pelvis/hip/knee/wrist…).

Type Conventions
- `Transmat_4x4 = np.ndarray` with shape `(4, 4)`, dtype `float32` or `float64`.
- All APIs accept/produce `float32` by default; inputs are validated and cast.
- No runtime dependency on MuJoCo/URDF libraries.

Public API (proposed)
- `class JointNode` (attrs)
  - Fields: `name: str`, `T_local: Transmat_4x4`, `index: int | None = None`, `tags: set[str] = field(factory=set)`
  - Validation: `T_local.shape == (4, 4)`, last row `[0,0,0,1]` within tolerance.

- `class GenericSkeleton` (regular class, not attrs)
  - Fields:
    - `name: str` — arbitrary user label; defaults to `uuid4().hex`
    - `_graph: nx.DiGraph` — nodes keyed by joint name; node attr `joint: JointNode`
    - `_root: str` — name of root joint (e.g., `pelvis`)
  - Properties (read‑only):
    - `joint_names: list[str]` — from graph nodes (no caching)
    - `base_joint_name: str` — root joint (same as `_root`)
  - Construction:
    - `from_nodes_edges(nodes: dict[str, Transmat_4x4], edges: list[tuple[str,str]], root: str, name: str | None = None)`
      - Builds graph; validates single root, acyclicity, connectivity to root, unique names, and transforms.
    - `from_chain(names: list[str], offsets: list[np.ndarray], name: str | None = None)`
      - Convenience for a simple chain; constructs `T_local` from translation offsets and identity rotation.
  - Topology helpers:
    - `get_joint_topology() -> nx.DiGraph`
      - Returns a read‑only copy (or view) of the joint DAG; nodes are joint names; edges parent→child. Edge may include optional `segment` metadata if provided by caller.
    - `parents() -> dict[str, str | None]` and `children() -> dict[str, list[str]]` for convenience.
    - `topo_order() -> list[str]` — joint names in topological order rooted at `base_joint_name`.
  - Kinematics:
    - `get_global_transforms() -> dict[str, Transmat_4x4]`
      - Computes FK in topo order. Optionally caches results and invalidates on updates to `T_local`.
    - `get_global_transform(name: str) -> Transmat_4x4`
    - `set_local_transform(name: str, T_local: Transmat_4x4) -> None`
    - `get_local_transform(name: str) -> Transmat_4x4`
  - Utilities:
    - `summary() -> str` — brief info for logging.
    - `copy(deep: bool = True) -> GenericSkeleton`

Notes and Conventions
- Root joint pose: In our unified model (`unified_model.py`) the pelvis/root can have pose that rotates/translates the entire skeleton; here it is represented by `T_local[root]` relative to world. FK applies it as the first transform.
- Actuated joints vs joints: We treat all joints as the canonical set (`joint_names`). There is no distinction for “actuated” in this representation.
- Link names: Not modeled. Human skeletons typically use joint names; link/segment naming is optional and can be attached as edge metadata (`segment`), but most pipelines won’t need it.

Validation Rules
- Graph must be a rooted DAG with exactly one root `base_joint_name`.
- Every node reachable from the root; no isolated subgraphs.
- Each `T_local` must be a valid homogeneous transform; last row `[0,0,0,1]` within `1e-6`.

Implementation Plan
1) Data model and utilities
   - Implement `Transmat_4x4` type alias, `JointNode` attrs with validators, and transform helpers (e.g., `as_float32`, `is_homogeneous`).
2) Core skeleton and graph management
   - Implement `GenericSkeleton` with constructors, validation, and topology helpers. Do not cache names; expose read‑only properties.
3) Forward kinematics
   - Implement FK over topo order; return a dict of `(4,4)` transforms. Add precise shape/type checks and informative errors.
4) Minimal I/O helpers (optional)
   - Serialize/deserialize to a simple JSON/YAML (names + 4×4 lists + edges + root). Defer if not immediately needed.
5) Tests
   - Replace MJCF tests with NetworkX tests: build a tiny skeleton (pelvis → thigh → shin) with 1m Z offsets; verify `get_global_transforms()` positions and topology functions.
   - Deterministic, no external I/O.
6) Refactor and cleanup
   - Replace `src/smplx_toolbox/core/skeleton.py` MJCF code with the new implementation.
   - Update `src/smplx_toolbox/core/__init__.py` exports if needed.
   - Remove MuJoCo/URDF deps from `pyproject.toml` (and mypy ignore for `mujoco.*`).
7) Docs
   - Add/adjust README section: “GenericSkeleton (Joint Graph)”.
   - Mention that human workflows typically use joint names; link names are optional and not modeled by default.

Acceptance Criteria
- `mypy src` passes (strict flags as configured).
- `ruff check` and `ruff format` clean.
- Unit tests for FK/topology pass; no dependency on MuJoCo/URDF.
- Clear, minimal public API: read‑only name properties, clean setters/getters for transforms.

Open Questions / Future Extensions
- Per‑joint DoF constraints and limits: out of scope for the graph itself; can be attached later as metadata or separate constraints module.
- Batch FK for time sequences (NumPy/torch): keep this class minimal; provide separate batched utilities if needed.
- Import/export bridges (SMPL‑X, HumanML3D rest poses) to construct a `GenericSkeleton` directly from those assets.

Work Breakdown (Sequenced)
1) Implement `JointNode`, transform validators, and helpers.
2) Implement `GenericSkeleton` (constructors, graph validation, properties).
3) Implement FK (`get_global_transforms`, single‑joint query, setters/getters).
4) Write unit tests (`tests/test_generic_skeleton.py`) for the new design; remove MJCF fixture.
5) Replace `skeleton.py` implementation; run `pixi run -e dev qa`.
6) Remove MuJoCo/URDF deps; update docs.

Notes for Contributors
- Follow `.magic-context/general/python-coding-guide.md` and repo style (attrs, typing, 4‑space indents, width 88, double quotes, sorted imports).
- Use `pixi run -e dev mypy src`, `ruff check .`, and `pytest -q -m "not slow"` for local validation.
- Temporary exploration scripts should go under `tmp/`.
