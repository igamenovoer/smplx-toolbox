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

## References
- `context/refcode/FlowMDM/runners/generate.py`
- `context/refcode/FlowMDM/data_loaders/humanml/scripts/motion_process.py`
- `context/refcode/FlowMDM/data_loaders/humanml/utils/paramUtil.py`
