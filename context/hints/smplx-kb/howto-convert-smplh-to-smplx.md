# Guide: Converting Between SMPL, SMPL-H, and SMPL-X

Two major workflows:
1. SMPL → SMPL-H (add articulated hands; same body topology).
2. SMPL-H → SMPL-X (add hands+face / extended joints; requires optimization & deformation transfer when starting from pure SMPL, but SMPL-H shares hands already).

---
## 1. SMPL → SMPL-H

This section explains upgrading legacy SMPL pose/shape data to SMPL-H (i.e. adding articulated hands) while preserving existing body motion and shape. No barycentric remeshing is needed because SMPL and SMPL-H share identical body vertex topology; SMPL-H augments the kinematic tree and blend weights with MANO-style hands.

### When to Use
You have SMPL parameters and need:
- Hand joint angles for grasp / gesture / interaction tasks.
- Unified body+hand meshes for rendering or training models expecting SMPL-H.

### Core Differences
- Body mesh topology: identical.
- Additional joints: hand articulation (≈15 per hand).
- Added parameters: `left_hand_pose`, `right_hand_pose` (axis-angle per hand joint).
- Shape space: same length betas (commonly 10) — copy directly.

### Straight Parameter Lift (Baseline)
1. Copy `betas`, `global_orient`, `body_pose`, `transl` from SMPL → SMPL-H.
2. Set `left_hand_pose`, `right_hand_pose` to zeros (neutral open hands).
3. Forward SMPL-H to get vertices & joints.

### Optional Hand Pose Initialization
- Heuristic slight finger curl (2–5°) for realism.
- Mirror one hand to the other if only one side provided (rare).
- Sample from a MANO / learned prior for activity-specific initialization.
- Optimize to 2D/3D keypoints if available.

### Step-by-Step
```python
from smplx import create
smplh = create(model_path, model_type='smplh', gender='neutral', use_pca=False, batch_size=B)

# Assume smpl_theta flattened (B, 24*3); betas (B,10); transl (B,3)
global_orient = smpl_theta[:, :3].reshape(B, 1, 3)
body_pose = smpl_theta[:, 3: 3 + smplh.NUM_BODY_JOINTS*3].reshape(B, smplh.NUM_BODY_JOINTS, 3)

param_dict = dict(
    betas=betas,
    global_orient=global_orient,
    body_pose=body_pose,
    left_hand_pose=torch.zeros(B, smplh.NUM_HAND_JOINTS, 3, device=device),
    right_hand_pose=torch.zeros(B, smplh.NUM_HAND_JOINTS, 3, device=device),
    transl=transl,
)
out = smplh(return_full_pose=True, get_skin=True, **param_dict)
vertices_h = out['vertices']
```

### Relaxed Hand Curl Helper
```python
def relaxed_hand_pose(batch, num_joints, device):
    pose = torch.zeros(batch, num_joints, 3, device=device)
    pose[..., 0] = 0.05  # small flex (radians)
    return pose
param_dict['left_hand_pose'] = relaxed_hand_pose(B, smplh.NUM_HAND_JOINTS, device)
param_dict['right_hand_pose']= relaxed_hand_pose(B, smplh.NUM_HAND_JOINTS, device)
```

### Optional Hand Optimization
```python
opt_params = [param_dict['left_hand_pose'], param_dict['right_hand_pose']]
optimizer = torch.optim.Adam(opt_params, lr=1e-2)
for it in range(200):
    out = smplh(return_full_pose=True, get_skin=True, **param_dict)
    pred = out['joints'][:, hand_joint_indices]
    loss = (pred - gt_hand_keypoints).pow(2).mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

### Practical Tips & Pitfalls
- Axis-angle values are radians; keep initial magnitudes small.
- Ensure consistent gender-specific model files with betas.
- Don’t reindex body joints—ordering matches.
- Pad betas with zeros if target expects more than source.

### Checklist (SMPL→SMPL-H)
- [ ] Load SMPL params.
- [ ] Instantiate SMPL-H model.
- [ ] Copy shared parameters.
- [ ] Initialize hand poses (zero / heuristic / prior).
- [ ] (Optional) optimize hands.
- [ ] Forward & export.

---
## 2. SMPL-H → SMPL-X

This section restores the original SMPL-H (or SMPL) to SMPL-X conversion guidance (hands + expressive face). When starting from SMPL-H you already have articulated hands, so main additions are facial joints & expression coefficients; when starting from pure SMPL you also need hand articulation.

### Overview
SMPL-X extends SMPL-H with face (jaw, eyes, expression blendshapes) and refined hand rig. Direct parameter copying is invalid because joint placements differ and SMPL-X mesh topology differs in head & hands regions. Thus a fitting (optimization) procedure is applied using a deformation (correspondence) transfer to build a pseudo target-topology mesh, then minimizing geometric discrepancies.

### Core Idea
1. Load or precompute vertex correspondences (barycentric mapping) from source (SMPL-H/SMPL) to target SMPL-X.
2. Synthesize a SMPL-X topology mesh per frame via linear combination of source posed vertices.
3. Stage optimization of SMPL-X parameters:
   - Edge length term for pose initialization (per-joint or holistic).
   - (Optional) translation-only refinement.
   - Full vertex loss over pose, shape, expression, translation.
4. Export SMPL-X parameters / meshes.

### Deformation (Correspondence) Matrix
Pickle file with `mtx` / `matrix` mapping source vertices to target template. If normals concatenated, slice first half. Example:
```python
with open(path,'rb') as f: data = pickle.load(f, encoding='latin1')
M = data['mtx']
if hasattr(M,'todense'): M = M.todense()
if M.shape[1] == 2*source_V: M = M[:, :source_V]
def_vertices = torch.einsum('mn,bni->bmi', [torch.tensor(M, device=device, dtype=torch.float32), source_vertices])
```

### Optimization Pipeline (`transfer_model.run_fitting` Inspired)
1. Initialize tensors: `transl`, `global_orient`, `body_pose`, `betas`, plus `left_hand_pose`, `right_hand_pose`, and (for SMPL-X) `jaw_pose`, `leye_pose`, `reye_pose`, `expression`.
2. Stage 1 (Edge): optimize pose (optionally per joint) using edge length loss on valid-correspondence edges.
3. Stage 2 (Translation): align centroids via vertex loss over translation.
4. Stage 3 (Full): optimize all parameters with vertex-to-vertex L2.
5. Decode axis-angle with `batch_rodrigues` before forward.

### Pseudocode Sketch
```python
# Given def_vertices (B, Vx, 3) from correspondence
params = init_zero_params(smplx_model)
edge_stage(params, def_vertices)        # per-joint LBFGS or Adam
transl_stage(params, def_vertices)      # optional
full_stage(params, def_vertices)        # optimize all
out = smplx_model(return_full_pose=True, get_skin=True, **decoded_params)
```

### Masking Invalid Vertices
Discard eye / inner-mouth vertices lacking reliable correspondence; restrict losses accordingly (edges kept only if both endpoints valid).

### Losses
- Edge loss: pose initialization robust to translation.
- Vertex loss: final alignment (optionally add priors for stability).

### Shape & Expression Handling
- Optimize `betas` mainly in vertex stage.
- Initialize `expression` to zeros (neutral) unless reliable face correspondences exist.

### Practical Tips
- Keep correspondence matrix on GPU.
- Use per-joint edge optimization for stability when starting from poor initialization.
- Warm-start across frames for sequences.
- Add regularizers (pose prior, shape prior) for noisy inputs.

### Common Pitfalls
- Directly copying pose vectors (joint semantics differ) → distorted limbs.
- Ignoring invalid vertex mask → facial artifacts.
- Forgetting to decode axis-angle before model forward.
- Mismatch in gender model → shape inconsistencies.

### Checklist (SMPL-H→SMPL-X)
- [ ] Load source vertices / faces.
- [ ] Load deformation transfer matrix.
- [ ] Build SMPL-X topology mesh (`def_vertices`).
- [ ] Initialize SMPL-X parameter tensors.
- [ ] Edge-based per-joint pose optimization.
- [ ] (Optional) translation refinement.
- [ ] Full vertex optimization over pose/shape/expression.
- [ ] Export parameters & mesh.

---
## References
- SMPL: https://smpl.is.tue.mpg.de/
- MANO / SMPL-H: https://mano.is.tue.mpg.de/
- SMPL-X: https://smpl-x.is.tue.mpg.de/
- SMPL-X Paper: Pavlakos et al., CVPR 2019.
- Embodied Hands (SMPL-H) Paper: Romero et al., SIGGRAPH Asia 2017.
- AMASS Dataset: https://amass.is.tue.mpg.de/

---
Source insights derived from reference code: `transfer_model.py`, `utils/def_transfer.py`, `docs/transfer.md` (original SMPL-X repo concepts) plus added SMPL→SMPL-H guidance.
