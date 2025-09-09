# SMPL Skeleton Keypoint Compatibility

This report aligns the official SMPL-X 55-joint skeleton (first 55 entries of `JOINT_NAMES`) against SMPL-H and SMPL.

- Indices are zero-based within each model’s official joint-name list.
- If a keypoint does not exist in a model, it is marked as `x`.
- Sources: `context/refcode/smplx/smplx/joint_names.py`.

## Models
- `SMPL-X (55)`: `JOINT_NAMES[:55]`; files: `data/body_models/smplx/SMPLX_MALE.pkl` (or `data/body_models/smplx/SMPLX_MALE.npz`).
- `SMPL-H (52 core)`: matched against `SMPLH_JOINT_NAMES` (length 73; first 52 are core skeleton); files: `data/body_models/smplh/SMPLH_MALE.pkl` (or `data/body_models/smplh/male/model.npz`).
- `SMPL-H-mano (52 core)`: matched against `SMPLH_JOINT_NAMES`; file: `data/body_models/mano_v1_2/models/SMPLH_male.pkl`. Note: lacks MANO hand PCA components; joint naming and order match SMPL-H for skeleton indices.
- `SMPL (24)`: matched against `SMPL_JOINT_NAMES`; file: `data/body_models/smpl/SMPL_NEUTRAL.pkl`.

## Index Map
| Keypoint | SMPL-X idx | SMPL-H idx | SMPL-H-mano idx | SMPL idx |
|---|---:|---:|---:|---:|
| pelvis | 0 | 0 | 0 | 0 |
| left_hip | 1 | 1 | 1 | 1 |
| right_hip | 2 | 2 | 2 | 2 |
| spine1 | 3 | 3 | 3 | 3 |
| left_knee | 4 | 4 | 4 | 4 |
| right_knee | 5 | 5 | 5 | 5 |
| spine2 | 6 | 6 | 6 | 6 |
| left_ankle | 7 | 7 | 7 | 7 |
| right_ankle | 8 | 8 | 8 | 8 |
| spine3 | 9 | 9 | 9 | 9 |
| left_foot | 10 | 10 | 10 | 10 |
| right_foot | 11 | 11 | 11 | 11 |
| neck | 12 | 12 | 12 | 12 |
| left_collar | 13 | 13 | 13 | 13 |
| right_collar | 14 | 14 | 14 | 14 |
| head | 15 | 15 | 15 | 15 |
| left_shoulder | 16 | 16 | 16 | 16 |
| right_shoulder | 17 | 17 | 17 | 17 |
| left_elbow | 18 | 18 | 18 | 18 |
| right_elbow | 19 | 19 | 19 | 19 |
| left_wrist | 20 | 20 | 20 | 20 |
| right_wrist | 21 | 21 | 21 | 21 |
| jaw | 22 | x | x | x |
| left_eye_smplhf | 23 | x | x | x |
| right_eye_smplhf | 24 | x | x | x |
| left_index1 | 25 | 22 | 22 | x |
| left_index2 | 26 | 23 | 23 | x |
| left_index3 | 27 | 24 | 24 | x |
| left_middle1 | 28 | 25 | 25 | x |
| left_middle2 | 29 | 26 | 26 | x |
| left_middle3 | 30 | 27 | 27 | x |
| left_pinky1 | 31 | 28 | 28 | x |
| left_pinky2 | 32 | 29 | 29 | x |
| left_pinky3 | 33 | 30 | 30 | x |
| left_ring1 | 34 | 31 | 31 | x |
| left_ring2 | 35 | 32 | 32 | x |
| left_ring3 | 36 | 33 | 33 | x |
| left_thumb1 | 37 | 34 | 34 | x |
| left_thumb2 | 38 | 35 | 35 | x |
| left_thumb3 | 39 | 36 | 36 | x |
| right_index1 | 40 | 37 | 37 | x |
| right_index2 | 41 | 38 | 38 | x |
| right_index3 | 42 | 39 | 39 | x |
| right_middle1 | 43 | 40 | 40 | x |
| right_middle2 | 44 | 41 | 41 | x |
| right_middle3 | 45 | 42 | 42 | x |
| right_pinky1 | 46 | 43 | 43 | x |
| right_pinky2 | 47 | 44 | 44 | x |
| right_pinky3 | 48 | 45 | 45 | x |
| right_ring1 | 49 | 46 | 46 | x |
| right_ring2 | 50 | 47 | 47 | x |
| right_ring3 | 51 | 48 | 48 | x |
| right_thumb1 | 52 | 49 | 49 | x |
| right_thumb2 | 53 | 50 | 50 | x |
| right_thumb3 | 54 | 51 | 51 | x |

## Note on MANO
- MANO is hand-only. Its 15 joints per hand correspond to the finger chains used in SMPL-H/SMPL-X (e.g., `thumb1..3`, `index1..3`, etc.). For non-hand keypoints (body, jaw, eyes), MANO does not apply.
