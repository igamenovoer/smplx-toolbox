Saved Command: Implement keypoint-based fitting losses (design + references)

- Read `ROADMAP.md` to confirm current focus on keypoint-based fitting.
- Create a design doc at `tasks/features/keypoint-match/task-keypoint-match.md` describing class architecture (no code yet).
- Plan code to live under `src/smplx_toolbox/optimization/` in separate files:
  - `keypoint_match_builder.py`, `pose_prior_vposer_builder.py`, `shape_prior_builder.py`, `angle_prior_builder.py`, plus shared base utilities.
- Define a PyTorch-style `KeypointMatchLossBuilder` with
  - `by_target_positions(dict[keypoint_name, position], dict[keypoint_name, weight] | None) -> torch.Tensor`-style behavior via a returned loss module.
  - Initialize builders with the unified model: `src/smplx_toolbox/core/unified_model.py`.
  - Produce differentiable, backprop-enabled losses that users can optimize.
- Keep VPoser as a separate loss builder; do not entangle with keypoint data term.
- Consult examples in `context/refcode/smplify-x` for data term, priors, and robustifier patterns.
- Optionally consult `context/hints/smplx-kb/howto-fit-smpl-family-models.md` for 2D vs 3D fitting guidance.

