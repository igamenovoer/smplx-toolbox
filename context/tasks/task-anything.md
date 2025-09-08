# specify pose with explicit keypoint

the pose vector is quite confusing for end user, because user will want to know which pose controls which key point but such information is missing in current design. We need to revise this significantly, and you need to consult smplx source code in `context/refcode` as well as #context7 for info. Here is what we are going to do:

- create a `attrs` structure to hold all the keypoint names and their tensor value in the pose vector, goes like this:

```python
@defined
class PoseByKeypoints:
    root: torch.Tensor | None  # (B,3)
    jaw: torch.Tensor | None  # (B,3)
    left_eye: torch.Tensor | None # (B,3)
    right_eye: torch.Tensor | None # (B,3)
    left_eyeball: torch.Tensor | None # (B,3)
    right_eyeball: torch.Tensor | None # (B,3)
    ...
```

such a structure will be used to specify the complete pose of smplx model. For use with subset of keypoints (smplh and smpl), just convert the structure to the corresponding one with less keypoints.