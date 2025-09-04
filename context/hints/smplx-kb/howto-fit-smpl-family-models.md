# How to Fit SMPL, SMPL-H, and SMPL-X Models

This document explains the optimization-based approach to fitting SMPL family models to 2D and 3D keypoints. The analysis is based on the source code provided in the `context/refcode/smplify-x` and `context/refcode/human_body_prior` submodules.

## 1. Fitting to 2D Keypoints (Analysis of `context/refcode/smplify-x`)

The goal is to find the body model parameters (shape, pose, camera translation) that best reproduce a set of 2D keypoint detections from a single image. This is an "analysis-by-synthesis" process where we iteratively adjust model parameters to minimize a projection error.

### Core Components

1.  **Body Model (e.g., `smplx.SMPLX`)**: A differentiable function that maps low-dimensional parameters to a 3D body mesh and joints.
    - **Inputs**: `betas` (shape), `body_pose`, `global_orient` (pose), `transl`.
    - **Outputs**: 3D vertices and 3D joint locations.

2.  **Camera Model (`smplifyx.camera.PerspectiveCamera`)**: A weak perspective camera that projects 3D points into the 2D image plane. Its translation is a key variable to be optimized.

3.  **Loss Function (`smplifyx.fitting.SMPLifyLoss`)**: The objective function to be minimized. It is a weighted sum of a data term and several regularization terms (priors).

4.  **Optimizer (`torch.optim.LBFGS`)**: An iterative algorithm that adjusts the model parameters to find a minimum of the loss function.

### The Optimization Process

The fitting is framed as minimizing a loss function `L` with respect to the model parameters `Θ = {β, θ, t, R}` where `β` are shape betas, `θ` is the pose, `t` is camera translation, and `R` is global orientation.

`L(Θ) = w_data * L_data + w_pose * L_pose + w_shape * L_shape + ...`

#### A. Data Term

The data term measures the discrepancy between the 2D keypoints provided (`J_2D_gt`) and the model's 3D joints projected into 2D (`J_2D_proj`).

- **Projection**: `J_3D = Model(β, θ, R)` -> `J_2D_proj = Camera(J_3D, t)`
- **Loss**: A robust L2 distance is used to be less sensitive to outlier detections. Joint confidences are used to weight the error for each joint.

```python
# From: context/refcode/smplify-x/smplifyx/fitting.py (SMPLifyLoss.forward)
projected_joints = camera(body_model_output.joints)
joint_diff = self.robustifier(gt_joints - projected_joints)
joint_loss = torch.sum(weights ** 2 * joint_diff) * self.data_weight ** 2
```

#### B. Prior Terms (Regularizers)

Priors are critical for ensuring that the optimized parameters result in a plausible human shape and pose, preventing degenerate solutions.

- **Pose Prior**: Penalizes unlikely joint rotations. This is often a `VPoser` model, which provides a low-dimensional latent space for valid human poses, or a Gaussian Mixture Model on axis-angle rotations.
- **Shape Prior**: A simple L2 penalty on the `betas` vector to keep the body shape within a realistic distribution. `shape_loss = ||β||²`.
- **Angle Prior**: Heuristics to prevent joints like elbows and knees from bending in unnatural directions.
- **Collision Prior**: A penalty for self-interpenetration, ensuring body parts do not pass through each other. This is computationally more expensive and is often applied in a final optimization stage.

#### C. Staged Optimization

To make the optimization more stable, it is performed in stages, typically by adjusting the weights of the loss terms or optimizing only a subset of parameters at a time. A common strategy is:
1.  **Initialize Camera**: Roughly estimate camera depth based on the scale of the 2D skeleton.
2.  **Fit Global Orientation & Pose**: Optimize for the overall pose while keeping the shape fixed.
3.  **Full Fit**: Optimize all parameters jointly, including shape (`betas`), and potentially enable the collision loss.

---

## 2. How to Fit to 3D Keypoints

Fitting to 3D keypoints (e.g., from mocap) is a simpler and more direct application of the same principles. The key difference is the **removal of the camera projection** from the data term. The optimization happens directly in 3D space. The implementation in the `context/refcode/human_body_prior` submodule provides an excellent reference for this task.

### Core Idea

The objective is to minimize the direct 3D distance between the model's joints and the target 3D keypoints, while still being regularized by pose and shape priors.

`L(Θ) = w_data * L_data_3D + w_pose * L_pose + w_shape * L_shape`

Here, the parameters to optimize are `Θ = {β, θ, t, R}`, where `t` and `R` are now the model's global translation and orientation in 3D space.

### Implementation Algorithm

1.  **Define the Data Term**: The data loss is the direct 3D distance (e.g., MSE or L1 loss) between the model's output joints (`J_3D_model`) and the target 3D joints (`J_3D_target`).

    ```python
    # From: context/refcode/human_body_prior/src/human_body_prior/models/ik_engine.py (ik_fit)
    res = source_kpts_model(free_vars) # Forward pass of the body model
    opt_objs['data'] = data_loss(res['source_kpts'], static_vars['target_kpts'])
    ```

2.  **Remove the Camera**: The camera model is no longer needed. The body model's `transl` and `global_orient` parameters are optimized directly.

3.  **Define Parameters to Optimize**: The set of learnable parameters (`free_vars`) should include:
    - `transl`: The model's root translation in 3D space.
    - `global_orient`: The model's root orientation.
    - `betas`: The shape parameters.
    - `poZ_body` (if using VPoser) or `body_pose`: The articulated body pose.
    - Hand and face parameters for SMPL-H/X.

4.  **Construct the Optimization Closure**: Create a function that performs the following steps:
    a. **Forward Pass**: Run the body model with the current set of parameters to get the model's 3D joint locations, `J_3D_model`.
    b. **Calculate Data Loss**: Compute `L_data = || J_3D_model - J_3D_target ||²`.
    c. **Calculate Prior Losses**: Compute `L_pose`, `L_shape`, and any other relevant priors.
    d. **Combine Losses**: Compute the total weighted loss: `L_total = w_data * L_data + ...`.
    e. **Return `L_total`**.

5.  **Run the Optimizer**: Use an optimizer like LBFGS to minimize the loss by iteratively calling the closure and updating the parameters.

    ```python
    # From: context/refcode/human_body_prior/src/human_body_prior/models/ik_engine.py (IK_Engine.forward)
    # 1. Initialize free_vars (betas, trans, poZ_body, root_orient) as torch.nn.Parameter
    free_vars = {k: torch.nn.Parameter(v.detach(), requires_grad=True) ... }

    # 2. Create the optimizer
    optimizer = torch.optim.LBFGS(list(free_vars.values()), ...)

    # 3. Create the closure function
    closure = ik_fit(optimizer, source_kpts_model, ...)

    # 4. Run the optimization loop, potentially in stages with different weights
    for wts in self.stepwise_weights:
        optimizer.step(lambda: closure(wts, free_vars))
    ```

This 3D fitting approach is more direct and often more robust than 2D fitting, as it avoids the ambiguities introduced by camera projection. The use of strong priors remains essential for achieving realistic results.

---

## 3. References

The concepts and code snippets described in this guide are based on the source code found in the local submodules and their corresponding public repositories:

-   **`context/refcode/smplify-x`**: The 2D fitting methodology.
    -   **Public Repository**: [https://github.com/vchoutas/smplify-x](https://github.com/vchoutas/smplify-x) (Note: The `smplify-x` code is part of the `smplx` repository).

-   **`context/refcode/human_body_prior`**: The variational pose prior (VPoser) and the 3D inverse kinematics (IK) engine.
    -   **Public Repository**: [https://github.com/nghorbani/human_body_prior](https://github.com/nghorbani/human_body_prior)