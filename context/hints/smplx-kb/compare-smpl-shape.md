# SMPL Shape/Pose Space Compatibility

This note compares `shapedirs` (shape blend shapes) and `posedirs` (pose
correctives) across SMPL, SMPL‑H, SMPL‑H‑mano, and SMPL‑X, and clarifies vertex
topology relationships (subset/equality) between models.

## Files Compared (workspace‑relative)
- SMPL: `data/body_models/smpl/SMPL_NEUTRAL.pkl`
- SMPL-H: `data/body_models/smplh/SMPLH_MALE.pkl`
- SMPL-H-mano: `data/body_models/mano_v1_2/models/SMPLH_male.pkl`
- SMPL-H (NPZ variant): `data/body_models/smplh/male/model.npz`
- SMPL-X (PKL): `data/body_models/smplx/SMPLX_MALE.pkl`
- SMPL-X (NPZ): `data/body_models/smplx/SMPLX_MALE.npz`

## Array Shapes (observed)
- SMPL: `v_template=(6890,3)`, `shapedirs=(6890,3,10)`, `posedirs=(6890,3,207)`
- SMPL-H: `v_template=(6890,3)`, `shapedirs=(6890,3,10)`, `posedirs=(6890,3,459)`
- SMPL-H-mano: `v_template=(6890,3)`, `shapedirs=(6890,3,10)`, `posedirs=(6890,3,459)`
- SMPL-H (NPZ): `v_template=(6890,3)`, `shapedirs=(6890,3,16)`, `posedirs=(6890,3,459)`
- SMPL-X (PKL/NPZ): `v_template=(10475,3)`, `shapedirs=(10475,3,400)`,
  `posedirs=(10475,3,486)`

Notes
- SMPL and SMPL‑H share the same vertex topology (6890 verts, 13776 faces).
- SMPL‑H (NPZ) extends shape space to 16 components; PKL has 10.
- SMPL‑X uses a different topology (10475 verts, 20908 faces) and splits shape
  space internally as 300 body shape + 100 expression components (total 400).

## Vertex Topology Relationships (subset/equality)
- SMPL ↔ SMPL‑H / SMPL‑H‑mano
  - Relationship: identical topology and indexing (1:1 match). B.vertices is
    equal to A.vertices (same ordering), not merely a subset.
  - Evidence: SMPL implementation notes “SMPL and SMPL‑H share the same
    topology”; observed identical `(V,F)` in the files above.

- SMPL/SMPL‑H ↔ SMPL‑X
  - Relationship: different topology. SMPL/SMPL‑H vertices are NOT a strict
    subset of SMPL‑X vertex indices, nor are indices aligned.
  - Practical view: The SMPL‑X body region corresponds geometrically to an SMPL
    body, but vertices are re‑indexed and the head/hands are fused; direct
    index‑level transfer of per‑vertex data (e.g., `shapedirs`, `posedirs`) is
    not valid without an explicit cross‑model vertex correspondence.

- MANO (hand‑only) ↔ SMPL‑H/SMPL‑X
  - Relationship: independent hand topology (778 verts per hand). MANO meshes
    are not a subset of SMPL/SMPL‑H/SMPL‑X body meshes; SMPL‑H/SMPL‑X integrate
    MANO kinematics but with re‑targeted/fused vertices.

## Shapedirs Compatibility
- Within same topology (SMPL ↔ SMPL‑H/SMPL‑H‑mano)
  - `shapedirs` share `(V=6890,3,*)`. You can:
    - Use SMPL betas directly with SMPL‑H PKL (10D). For SMPL‑H NPZ (16D), pad
      SMPL’s 10D betas with zeros in the last 6 dims.
    - Converting 16D SMPL‑H betas to SMPL: drop the last 6 dims.
  - Numerical values are trained per model release, but semantics and vertex
    alignment match; direct reuse is structurally compatible.

- Across different topology (to/from SMPL‑X)
  - `shapedirs` are per‑vertex; since `(V)` differs (6890 vs 10475), you cannot
    copy or index‑subset `shapedirs` between SMPL/SMPL‑H and SMPL‑X.
  - Transfer strategy: compute `v_shaped` in each model’s native space from its
    own betas (and expression for SMPL‑X). If you must transfer per‑vertex
    displacements, build an explicit vertex correspondence (e.g., closest point
    mapping between templates) and resample.

- SMPL‑X shape vs expression components
  - The 400 components comprise ~300 body‑shape and ~100 expression directions.
    For “SMPL‑like” body shapes, use the body shape subspace; expression should
    be zeroed when comparing with SMPL/SMPL‑H.

## Posedirs Compatibility
- Dimensions recap
  - SMPL: 23 body joints × 9 = 207
  - SMPL‑H: 51 joints × 9 = 459 (body + 30 hand joints)
  - SMPL‑X: 54 joints × 9 = 486 (SMPL‑H + jaw + eyes)

- Within same topology (SMPL ↔ SMPL‑H/SMPL‑H‑mano)
  - The first 207 pose components in SMPL‑H correspond to SMPL’s 23 body joints
    and share the same vertex topology (6890). Thus, body‑only posedirs are
    structurally compatible; SMPL‑H adds hand‑joint correctives beyond that.

- Across different topology (to/from SMPL‑X)
  - `posedirs` are per‑vertex; due to different `(V)`, direct reuse is not
    valid. Joint ordering for the body part matches conceptually (SMPL‑X builds
    on SMPL‑H), but the vertex basis differs; use the native model to compute
    pose correctives or resample via a vertex correspondence.

## Practical Guidance
- Same topology (SMPL ↔ SMPL‑H variants)
  - Shapes: interchange betas by padding/truncating to the target dimension.
  - Poses: body posedirs align; hands exist only in SMPL‑H.

- Different topology (any ↔ SMPL‑X or MANO)
  - Do not transfer `shapedirs/posedirs` by slicing; compute in the target
    model, or build a vertex correspondence and resample per‑vertex data.
  - When comparing shapes, zero SMPL‑X expression to isolate body shape.

