# How To Convert SMPL-H (or SMPL) Parameters / Meshes to SMPL-X

This hint summarizes the official SMPL-X conversion approach (see the original `transfer_model` reference code) for upgrading legacy SMPL / SMPL-H assets (poses, shapes, motions) into SMPL-X while preserving body shape and pose and initializing hands/face reasonably.

## When to Use
Use this when you have:
- Existing motion capture or datasets in SMPL or SMPL+H (e.g. AMASS) and want SMPL-X meshes / parameters.
- Downstream pipelines (rendering, learning) that now expect SMPL-X topology (articulated hands + expressive face).

## Core Idea
1. Build (or load) precomputed vertex correspondences from source topology (SMPL / SMPL-H) to target (SMPL-X) via barycentric mapping on the source posed mesh.
2. Use the correspondences to synthesize a pseudo target-topology mesh for each source frame.
3. Fit the SMPL-X parametric model (pose, shape, expression, translation) to that synthetic mesh using a staged optimization:
   - Edge length term (robust to global translation) to initialize articulation per-joint.
   - (Optional) translation-only refinement.
   - Full vertex-to-vertex optimization over all free parameters.
4. Export resulting SMPL-X parameters and/or meshes.

## Prerequisites
- Access to licensed SMPL-H / SMPL-X model files (neutral or gendered) and the deformation transfer (correspondence) data produced offline.
- PyTorch environment matching the reference implementation requirements.
- Deformation / correspondence matrix (pickle) mapping source vertices to target template (see below).

## Deformation (Correspondence) Matrix
The reference code loads a pickled structure containing a sparse (or dense) matrix `mtx` (or `matrix`). For pure vertex positions (no normals):
```
with open(path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')
M = data['mtx']  # shape: [N_target_verts, 2*N_source_verts] or [N_target_verts, N_source_verts]
if hasattr(M, 'todense'): M = M.todense()
# If normals concatenated, slice first half:
num_source = M.shape[1] // 2
M = M[:, :num_source]
```
Each target vertex v^X_i is a linear combination of source vertices (barycentric embedding over the source mesh triangles or one-hot for identical regions).

In code (see `utils/def_transfer.py`):
```
def_vertices = torch.einsum('mn,bni->bmi', [def_matrix, source_vertices])
```
Produces a batch of synthesized SMPL-X topology meshes from source posed meshes.

## Optimization Pipeline (Reference `transfer_model.run_fitting`)
High-level steps per batch/frame:
1. Initialize learnable parameter tensors:
   - translation (transl)
   - global_orient (1×3 axis-angle)
   - body_pose (J_body×3 axis-angle)
   - (hands) left_hand_pose / right_hand_pose (if target is SMPL-X, sized NUM_HAND_JOINTS)
   - (face) jaw_pose, leye_pose, reye_pose, expression (if SMPL-X)
   - betas (shape coefficients)
2. Stage 1: Edge-based fitting
   - Build edge list from current target template faces restricted to valid vertex mask.
   - Loss: sum over edges of squared difference of edge vectors between estimated and synthesized (`def_vertices`).
   - Optionally optimize per-joint (loop joints, optimize a local axis-angle `part` inserted into the body pose array) for stability.
3. Stage 2: (If present) translation-only vertex loss to align centroids.
4. Stage 3: Full vertex-based loss over all parameters (pose, shape, expression, translation).
5. Decode axis-angle to rotation matrices via `batch_rodrigues` before forwarding the body model.
6. Save resulting parameters and faces.

### Pseudocode Sketch
```
# Load source frame vertices_smplh (B,V_src,3) and faces_src
# Load deformation transfer matrix M (N_tgt, V_src)
def_vertices = einsum('mn,bni->bmi', M, vertices_smplh)

# Build SMPL-X model (target)
params = init_zero_params(model)  # create tensors with requires_grad

# 1. Edge stage
for each body/hand pose joint j:
    optimize local axis-angle part_j with edge_loss( model(params_with_part_j), def_vertices )

# 2. Translation stage (optional)
optimize transl with vertex_l2( model(params), def_vertices )

# 3. Full stage
optimize all params with vertex_l2( model(params), def_vertices )

# Output: pose, betas, expression, transl
```

## Masking Invalid Vertices
Some SMPL-X vertices (eyes, inner mouth) have no valid source correspondence. Maintain a boolean validity mask; restrict edge and vertex losses to masked indices / edges whose both endpoints are valid.

## Key Losses
- Edge loss: encourages preservation of local differential geometry before absolute alignment.
- Vertex loss: final fine alignment in Euclidean space.

## Why Edge-Then-Vertex?
Optimizing edges first reduces sensitivity to global translation / small drift and provides a good pose initialization before refining absolute positions and shape/expression.

## Handling Shape & Expression
- Shape (betas) typically optimized only in the final stage (vertex loss) because edge differences carry less direct shape signal.
- Expression coefficients initialized at zero; unless you have face correspondences with high confidence, expect neutral expressions—some pipelines freeze expression or apply a lightweight regularizer.

## SMPL vs SMPL-H vs SMPL-X
- SMPL ↔ SMPL-H: identical torso/body topology; hands require additional joints & blend shapes when going to SMPL-H.
- SMPL-H ↔ SMPL-X: SMPL-X extends with facial and refined hand rig. For SMPL-H→SMPL-X you can reuse body & hand correspondences directly; only face needs barycentric mapping or is initialized neutral.

## Practical Tips
- Always operate in consistent scale (models are meters).
- Use float32 tensors on GPU for speed; convert correspondence matrix to torch tensor once and cache.
- If optimizing large sequences (e.g., motion capture), warm-start each frame from previous frame’s result to accelerate convergence.
- Regularization: you may add pose prior or shape prior losses (not shown in minimal reference) to stabilize.
- Batch size often 1 (frame-wise); vectorization possible if frames share topology.

## Minimal Example Snippets
Load deformation transfer matrix:
```
from transfer_model.utils.def_transfer import read_deformation_transfer
M = read_deformation_transfer('path/to/smplh2smplx.pkl', device=device)
```
Apply to posed SMPL-H mesh vertices:
```
def_vertices = torch.einsum('mn,bni->bmi', M, smplh_vertices)
```
Edge optimization closure pattern (simplified):
```
optimizer = torch.optim.LBFGS([part_param])

def closure():
    optimizer.zero_grad()
    out = smplx_model(return_full_pose=True, get_skin=True, **param_dict)
    loss = edge_loss(out['vertices'], def_vertices)
    loss.backward()
    return loss
optimizer.step(closure)
```
Final decoding of axis-angle to rotation matrices uses `batch_rodrigues` before forward pass.

## External References
- SMPL: https://smpl.is.tue.mpg.de/
- SMPL+H: https://mano.is.tue.mpg.de/ (includes body + hands, see Embodied Hands paper)
- SMPL-X: https://smpl-x.is.tue.mpg.de/
- SMPL-X Paper (CVPR 2019): Pavlakos et al. “Expressive Body Capture: 3D Hands, Face, and Body from a Single Image.”
- AMASS dataset: https://amass.is.tue.mpg.de/

## Common Pitfalls
- Misaligned gender: ensure consistent gender-specific model sets (male/female/neutral) between source and target for best shape transfer.
- Forgetting to mask invalid vertices causing artifacts (eyes collapsing or facial jitter).
- Directly copying pose parameters: joint index ordering / hierarchy differences invalidate naive transfer.
- Large frame-to-frame variation: consider temporal smoothing or warm-start.

## Summary Checklist
- [ ] Load source SMPL/SMPL-H vertices.
- [ ] Load deformation transfer matrix.
- [ ] Generate target-topology mesh via linear combination.
- [ ] Initialize SMPL-X parameters (zeros).
- [ ] Edge-based per-joint pose optimization.
- [ ] (Optional) translation-only refinement.
- [ ] Full vertex fitting over pose, shape, expression.
- [ ] Save SMPL-X parameters & meshes.

---
Source insight derived from reference code: `transfer_model.py`, `utils/def_transfer.py`, and `docs/transfer.md`.
