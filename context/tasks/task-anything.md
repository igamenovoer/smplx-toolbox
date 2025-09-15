# Revise `NamedPose` class

we need to revise the `NamedPose` class (src/smplx_toolbox/core/containers.py) with the following changes.

## Allow to store information about the root joint (pelvis)

the current `NamedPose` class is designed to only store intrinsic pose information (i.e., relative rotations of joints), and it does not store any information about the root joint (pelvis). This is inconvenient in some scenarios, as user will always find some other way to store the pelvis information (e.g., as a separate tensor).

besides, we need to make the joint indices consistent with SMPL definitions, it defines pelvis as the root joint (index 0), while in our current `NamedPose` design, pelvis is excluded from the pose representation, so all other joints are shifted by -1 index. For indices definition, see `context/hints/smplx-kb/compare-smpl-skeleton.md`.

### the current design of `NamedPose` is as follows:
- it has a python property pelvis, which always returns a zero tensor of shape (B, 3)
- in its getters (e.g., get_joint_pose()), it refuses to return the pelvis joint (raise error), to force users to handle it separately

### change it into:
- rename the `.packed_pose` member variable into `.intrinsic_pose`, to clarify that it only stores intrinsic pose, semantic is the same as before
- add a new member variable `.root_pose` of shape (B, 3), to represent the global orientation of the pelvis
- in property `.pelvis`, return the actual pelvis rotation stored in `.root_pose`
- allow getters (e.g., get_joint_pose()) to return the pelvis joint as well, if requested
- `.root_pose` can be None, in which case the pelvis is assumed to be zero (same as before)

### member function behavior changes:
- pose getters by name, will now return pelvis if requested
- pose getters by index, will follow SMPL indexing, see `context/hints/smplx-kb/compare-smpl-skeleton.md` for index mapping, basically 0 for pelvis, and all other joints shifted by +1 index (index k is for .intrinsic_pose[k-1] except for k=0)
- `to_dict(pelvis_pose:Tensor|None=None)` will include pelvis anyway, no `with_pelvis` argument, if pelvis_pose is given, then use it as the pelvis rotation, otherwise use self.pelvis

### add these functions:
- `to_full_pose(pelvis_pose:Tensor|None = None) -> Tensor`, which returns a packed pose tensor of shape (B, N, 3), where N is the number of joints including pelvis if with_root is True (use torch.cat() to prepend pelvis to intrinsic_pose). If pelvis_pose (shape=(B,3)) is given, then use it as the pelvis rotation, otherwise use self.pelvis. In this way, we allow user to store pelvis rotation separately, but still can get a full pose tensor when needed.