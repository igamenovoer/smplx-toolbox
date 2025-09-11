# Implement helper class for keypoint-based smpl fitting

## Purpose
we are going to write a helper class that can be used to:
- build loss terms for keypoint matching, and internally keep track of all the loss terms
- apply vposer prior, l2 regularization on pose, etc.
- run the optimization loop using simple interface

the purpose of this class is to reduce the amount of boilerplate code needed to run keypoint-based fitting.

## requirements
- this is ONLY for 3d keypoint fitting, should work on SMPL, SMPL-H, SMPL-X models
- the `helper` class should be initialized with a smpl model
- for vposer prior, user can provide the checkpoint path or a pre-loaded vposer model
- user can set keypoint target positions, by specifying the keypoint names and target positions, and the fitting weight, and unset them if needed.
- user can set l2 reguilarization on pose, and can unset it if needed
- user can choose to enable/disable vposer prior, and can set the weight
- the run the optimization loop, the user first call `step_iter=init_fitting(smpl_params: UnifiedSmplInputs)` to get an iterator, then call `next(step_iter)` to run one step of optimization, the optimization status will be returned as a `attrs` structure (see `.magic-context/instructions/attrs-usage-guide.md`), and user uses standard python iterator interface to run the optimization loop and detect when the optimization is done. `smpl_params` is the initial parameters to start the optimization, user can set `requires_grad=True` on the parameters they want to optimize, and also the initial values are given in this input.
- user can also get all the loss terms and their weights, if user wants to build their own optimization loop externally.

## Reference
- demo of fitting: `tests/fitting/*.py`