# How to Extract FlowMDM HumanML3D Root Transform

## Why this hint exists
Task 2.4 asks whether FlowMDM already computes global translation and orientation for HumanML3D generations, and how to access them. The answer is **yes**—the reconstruction happens inside the HumanML3D decoding path via `recover_root_rot_pos`. This note shows where that logic lives and how to reuse it when you need explicit root transforms.

## Where FlowMDM computes the root pose
- `context/refcode/FlowMDM/runners/generate.py`: `feats_to_xyz()` loads HumanML3D samples, denormalises them, and calls `recover_from_ric()`.
- `context/refcode/FlowMDM/data_loaders/humanml/scripts/motion_process.py`: `recover_from_ric()` invokes `recover_root_rot_pos()` (lines 371–402) to integrate the planar root trajectory and yaw quaternion before joints are assembled.

Key takeaways from `recover_root_rot_pos()`:
- Input features are already normalised; the first channels store root yaw velocity, planar velocity, and pelvis height.
- The function integrates yaw velocity to a quaternion `(w, x, y, z)`; only the Y component is non-zero because HumanML3D models yaw-only global rotation.
- Root translation is accumulated in X/Z, while the Y component (height) is copied from the pelvis channel. The result `r_pos` is a per-frame `[x, y, z]` vector.
- Downstream functions (`recover_from_ric`, `recover_from_rot`) discard the quaternion/translation after applying them to the joints, so they are *not* saved in `results.npy`.

## How to extract the data yourself
1. Run generation using `generate-ex.py` (or the original script) but intercept the tensor **before** it goes into `feats_to_xyz()`. The tensor shape is `[batch, 263, 1, frames]` for HumanML3D.
2. Denormalise using the same statistics as the original pipeline:
   ```python
   import numpy as np
   from pathlib import Path

   flowmdm_root = Path('context/refcode/FlowMDM')
   mean = np.load(flowmdm_root / 'dataset/HML_Mean_Gen.npy')
   std = np.load(flowmdm_root / 'dataset/HML_Std_Gen.npy')
   sample = sample.cpu().permute(0, 2, 3, 1)  # [B, 1, T, 263]
   sample = (sample * std + mean).float().squeeze(1)  # [B, T, 263]
   ```
3. Call `recover_root_rot_pos()` to retrieve the root quaternion and translation:
   ```python
   from context.refcode.FlowMDM.data_loaders.humanml.scripts.motion_process import recover_root_rot_pos

   r_rot_quat, r_pos = recover_root_rot_pos(sample)
   # Shapes: [B, T, 4] and [B, T, 3]
   ```
4. Convert the quaternion to the representation you need (axis–angle, rotation matrix, etc.). Example using SciPy:
   ```python
   from scipy.spatial.transform import Rotation as R

   # FlowMDM stores quaternions as (w, x, y, z)
   rot = R.from_quat(r_rot_quat.numpy()[0, :, [1, 2, 3, 0]])  # reorder to (x, y, z, w)
   global_orient_axis_angle = rot.as_rotvec()  # [T, 3]
   ```
5. Combine with translation if you want 4×4 homogeneous transforms:
   ```python
   import numpy as np

   transforms = []
   for aa, transl in zip(global_orient_axis_angle, r_pos.numpy()[0]):
       R_mat = R.from_rotvec(aa).as_matrix()
       T = np.eye(4, dtype=np.float32)
       T[:3, :3] = R_mat
       T[:3, 3] = transl
       transforms.append(T)
   ```

### Tips
- If you already have `results.npy`, you can recover translations (`joints[:, 0, :]`) but you **cannot** reconstruct the yaw quaternion because it was consumed during joint recovery. Capture the denormalised features instead when you run the generator.
- For debugging, print a few frames of `r_rot_quat`; values other than the first two entries should stay near zero, confirming yaw-only global rotation.
- When adapting this for batches >1, remember that FlowMDM stacks prompts in the batch dimension, so index into `r_rot_quat[batch_index]`.

## HumanML3D Normalization Stats (HML_Mean_Gen.npy / HML_Std_Gen.npy)

- Files: `context/refcode/FlowMDM/dataset/HML_Mean_Gen.npy`, `context/refcode/FlowMDM/dataset/HML_Std_Gen.npy`
- Origin: symlinks to HumanML3D’s `Mean.npy` and `Std.npy` in `context/refcode/HumanML3D/HumanML3D/`
- Shape/dtype: `(263,)` `float32` each
- Meaning: per‑channel dataset mean and standard deviation for the 263‑dim HumanML3D feature vector. Used to z‑normalize during training and to de‑normalize model outputs during sampling.
- Where used: `context/refcode/FlowMDM/runners/generate.py:93` loads these files to denormalize before converting features to XYZ joints.

Denormalization rule (applied per channel, per frame):

```python
denorm = norm * std + mean  # broadcasted over time dimension
```

Keep the stats paired with the checkpoint. Wrong stats → wrong global scale/trajectory.

## 263‑D Feature Layout (channel map)

After de‑normalization, the HumanML3D feature vector at each frame has the following layout (indices in brackets):

- Root state (4):
  - [0] yaw angular velocity (Δθ around Y, radians per frame)
  - [1:3] planar linear velocity (Vx, Vz) in the root frame
  - [3] root height Y (pelvis height)
- RIC local joint positions (63):
  - [4 : 4 + 21×3) → joints 1..21, 3D each
- Joint rotations, continuous 6D (126):
  - [4 + 21×3 : 4 + 21×3 + 21×6) → joints 1..21, 6D each
- Local joint velocities (66):
  - next 22×3 channels → joints 0..21, 3D each
- Foot contacts (4):
  - last 4 channels → binary contacts for pair of L/R foot joints

This ordering matches how `motion_process.py` concatenates features during dataset creation and how `recover_from_ric()` expects them at decode time.

## Recover Root Transform Using The Stats

Recommended: use the built‑in function after de‑normalization.

```python
from context.refcode.FlowMDM.data_loaders.humanml.scripts.motion_process import recover_root_rot_pos

# sample_denorm: [B, T, 263] from the steps above
r_quat, r_pos = recover_root_rot_pos(sample_denorm)
# r_quat: [B, T, 4] in (w, x, y, z) with yaw-only rotation
# r_pos : [B, T, 3] accumulated translation in world frame (x, y, z)
```

What `recover_root_rot_pos()` does internally:
- Integrates yaw angle by cumulative sum of channel [0] (Δθ) to get θ(t)
- Builds quaternion `(w=cos θ, x=0, y=sin θ, z=0)` (yaw‑only)
- Integrates planar velocities from channels [1:3] over time, rotated into world frame by the inverse yaw
- Sets translation Y from channel [3]

## Manual Root Decode (optional)

If you want to replicate the math without calling the helper:

```python
import torch
from context.refcode.FlowMDM.data_loaders.humanml.scripts.quaternion import qinv, qrot

# X: [B, T, 263] denormalized features
rot_vel = X[..., 0]                                 # [B, T]
planar_v = X[..., 1:3]                              # [B, T, 2] (Vx, Vz)
height_y = X[..., 3]                                # [B, T]

# integrate yaw angle θ(t) from per-frame Δθ
theta = torch.zeros_like(rot_vel)
theta[..., 1:] = rot_vel[..., :-1]
theta = torch.cumsum(theta, dim=-1)                 # [B, T]

# yaw quaternion (w, x, y, z) with yaw-only rotation
r_quat = torch.zeros(X.shape[:-1] + (4,), device=X.device)
r_quat[..., 0] = torch.cos(theta)
r_quat[..., 2] = torch.sin(theta)

# accumulate translation in world frame
r_pos = torch.zeros(X.shape[:-1] + (3,), device=X.device)
r_pos[..., 1:, 0] = planar_v[..., :-1, 0]
r_pos[..., 1:, 2] = planar_v[..., :-1, 1]
r_pos = qrot(qinv(r_quat), r_pos)                   # rotate velocities into world
r_pos = torch.cumsum(r_pos, dim=-2)                 # integrate over time
r_pos[..., 1] = height_y                            # set pelvis height
```

Notes:
- Channel indices refer to the de‑normalized feature vector. Always apply `denorm = norm * std + mean` first using the 263‑D stats.
- The quaternion convention is `(w, x, y, z)` and the global rotation is yaw‑only by construction of HumanML3D features.

## References
- `context/refcode/FlowMDM/runners/generate.py`
- `context/refcode/FlowMDM/data_loaders/humanml/scripts/motion_process.py`
- `context/refcode/FlowMDM/data_loaders/humanml/utils/paramUtil.py`
