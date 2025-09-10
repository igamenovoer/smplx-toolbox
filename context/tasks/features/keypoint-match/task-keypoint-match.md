Title: Keypoint-Based Fitting – Loss Builder Design (No Code Yet)

Context
- Scope: Loss-building utilities for keypoint-based fitting under `src/smplx_toolbox/optimization/`.
- Model: Built around `UnifiedSmplModel` from `src/smplx_toolbox/core/unified_model.py`.
- References: `context/refcode/smplify-x` (2D fitting pipeline, priors, robustifier) and VPoser from `context/refcode/human_body_prior`.
- Goal: Design a small set of PyTorch-style "loss builder" classes that construct differentiable losses users can optimize with their chosen optimizers. No implementation in this task.

Design Principles
- Composable losses: Each builder focuses on a related family of losses (keypoint matching, pose prior, shape prior, etc.).
- Backprop-enabled: Builders return callables/modules that compute scalar losses from model outputs and optional extras (e.g., camera).
- Unified joint space: Default operations use the unified 55-joint scheme via `UnifiedSmplModel` and its `UnifiedSmplOutput`.
- Targets: Primary API uses joint names only; an advanced batch API accepts full-skeleton targets by index in the model's native joint order (SMPL:24/SMPL-H:52/SMPL-X:55).
- Robustness: Support robust penalties (e.g., GMoF/Huber) and masking of missing/unreliable joints.
- Separation of concerns: VPoser/priors are separate builders; keypoint data term is independent of priors.
 - User-controlled composition: Users compose and weight the resulting `SmplLossTerm`s themselves (e.g., `total = w_data * L_data + w_pose * L_pose + ...`). No framework-level composer is provided.

Key Types (conceptual)
- `UnifiedSmplModel`: model adapter (provided).
- `UnifiedSmplOutput`: forward result with `joints`, `vertices`, `full_pose`, etc. (provided).
- `SmplLossTerm`: a `torch.nn.Module`-like object exposing `forward(...) -> torch.Tensor` that returns a scalar loss. (Interface only in this doc.)

Package Layout (planned; subject to refinement)
- `src/smplx_toolbox/optimization/`
  - `builders_base.py` – Abstract base helpers shared by all builders.
  - `keypoint_match_builder.py` – Data term for 3D keypoint matching in model/unified space.
  - `projected_keypoint_match_builder.py` – 2D keypoint matching via camera projection (requires camera at init).
  - `pose_prior_vposer_builder.py` – VPoser-based body pose prior.
  - `shape_prior_builder.py` – Shape/beta L2 prior and optional constraints.
  - `angle_prior_builder.py` – Knee/elbow bending prior (heuristic).

Base Interfaces
- `class BaseLossBuilder:`
  - Purpose: Common utilities (device/dtype sync, name→index mapping, broadcasting weights, robustifier creation, joint selection).
  - Init:
    - `model: UnifiedSmplModel`
  - Utilities (non-exhaustive):
    - `_names_to_indices(names: list[str]) -> list[int]` (internal helper)
    - `_prepare_weights(n: int, weights: float | Tensor | dict[str,float|Tensor] | None) -> Tensor` (broadcast to `(B, n, 1)` or `(n,)`)
    - `_robust(x: Tensor, kind: Literal["l2","huber","gmof"], rho: float | None) -> Tensor`
    - `_ensure_target(tgt) -> Tensor` (to model device/dtype)
    - `_select_joints(joints, names: list[str]) -> Tensor` (via `UnifiedSmplModel.select_joints`)
  - Reduction: Follow PyTorch convention and supply `reduction` per individual loss (e.g., `"none"|"mean"|"sum"`), not at builder init. Accept both Python `None` and string `"none"` with equivalent meaning.

Keypoint Matching (3D Data Term)
- `class KeypointMatchLossBuilder(BaseLossBuilder):`
  - Purpose: Build losses that pull unified/model joints to specified 3D targets.
  - 3D only. For 2D supervision, use `ProjectedKeypointMatchLossBuilder`.
  - Public builders:
    - `by_target_positions(`
      `targets: dict[str, Tensor],`  
      `weights: dict[str, float|Tensor] | float | Tensor | None = None,`
      `robust: Literal["l2","huber","gmof"] = "gmof",`
      `rho: float = 100.0,`
      `confidence: dict[str, float|Tensor] | float | Tensor | None = None,`  
      `missing: Literal["ignore","zero","nan"] = "ignore",`
      `reduction: Literal["none","mean","sum"] | None = "mean",`
      `) -> SmplLossTerm`
      - Targets must be provided by joint name: `{ "left_shoulder": (B,3), ... }`.
      - Behavior:
        - Compare directly in model/world space: `||J_pred - J_tgt||` on unified joints selected by name.
        - Apply `confidence` and `weights` multiplicatively per joint; ignore or zero-out missing joints based on `missing`.
      - Returns a `SmplLossTerm` with signature:
        - `forward(output: UnifiedSmplOutput) -> Tensor`
    - `by_target_positions_packed(`
      `target_positions: Tensor,`
      `weights: Tensor,`
      `robust: Literal["l2","huber","gmof"] = "gmof",`
      `rho: float = 100.0,`
      `missing: Literal["ignore","zero","nan"] = "ignore",`
      `reduction: Literal["none","mean","sum"] | None = "mean",`
      `) -> SmplLossTerm`
      - Advanced API: provide all keypoints at once by index using the model's native joint order (not unified). The user must know the ordering of the underlying SMPL family model.
      - Shapes:
        - `target_positions`: `(B, J_raw, 3)` for 3D.
        - `weights`: `(J_raw,)` or `(B, J_raw)`; broadcastable to `(B, J_raw, 1)`. Set `weights[i]=0` to omit the i-th joint.
      - Behavior:
        - Uses `output.extras["joints_raw"]` for comparison in the model's native joint order.
        - Requires full skeleton (cannot be partial); will validate `J_raw` against the wrapped model type (24/52/55).
      - Returns a `SmplLossTerm` with signature:
        - `forward(output: UnifiedSmplOutput) -> Tensor`

Projected Keypoint Matching (2D Data Term)
- `class ProjectedKeypointMatchLossBuilder(BaseLossBuilder):`
  - Purpose: Build losses that pull model joints to 2D targets in image space via a differentiable camera projection using Kornia conventions.
  - Camera parameters (Kornia-style):
    - Accept any of the following at initialization; precedence top→bottom:
      - `camera_model`: a Kornia camera module with projection capability, e.g., `kornia.geometry.camera.PinholeCamera` or `kornia.sensors.camera.CameraModel`. Must expose a callable to project `(B, J, 3_c)` camera-frame points to `(B, J, 2)` pixels or a `project`-like method we can invoke.
      - Intrinsics + extrinsics:
        - `K`: intrinsic matrix `(B, 3, 3)` or `(3, 3)` with `[fx, 0, cx; 0, fy, cy; 0, 0, 1]`.
        - `Tcw`: world-to-camera transform `(B, 4, 4)` or `(4, 4)`; if only rotation/translation are provided, we assemble `Tcw` internally.
        - Optionally `distortion` parameters (reserved for future; we can later route through Kornia distortion utilities).
      - Camera type: `camera_type: Literal["pinhole","orthographic"] = "pinhole"` to select between perspective projection (default) and orthographic helpers in Kornia.
    - Implementation sketch:
      - Compute camera-frame points from world joints: `X_cam = (Tcw @ to_homogeneous(X_world))[..., :3]` or via Kornia transforms.
      - Project to pixels:
        - Pinhole: `uv = kornia.geometry.camera.cam2pixel(X_cam, K)`.
        - Orthographic: `uv = kornia.geometry.camera.project_points_orthographic(X_cam, K_like)` (or an equivalent helper; we will map inputs accordingly).
      - If a `camera_model` is provided, we call its projection method instead of using raw `K/Tcw`.
    - Parameter optimization: If `K`, `Tcw`, or `camera_model` hold `torch.nn.Parameter`s, gradients flow into them, enabling joint optimization of camera and body.
  - Public builders:
    - `by_target_positions(`
      `targets: dict[str, Tensor],`
      `weights: dict[str, float|Tensor] | float | Tensor | None = None,`
      `confidence: dict[str, float|Tensor] | float | Tensor | None = None,`
      `robust: Literal["l2","huber","gmof"] = "gmof",`
      `rho: float = 100.0,`
      `missing: Literal["ignore","zero","nan"] = "ignore",`
      `reduction: Literal["none","mean","sum"] | None = "mean",`
      `) -> SmplLossTerm`
      - Targets must be provided by joint name: `{ "left_shoulder": (B,2), ... }` (pixel units).
      - Returns `SmplLossTerm` with `forward(output: UnifiedSmplOutput) -> Tensor` (camera captured from init and used via Kornia projection).
    - `by_target_positions_packed(`
      `target_positions: Tensor,`
      `weights: Tensor,`
      `robust: Literal["l2","huber","gmof"] = "gmof",`
      `rho: float = 100.0,`
      `missing: Literal["ignore","zero","nan"] = "ignore",`
      `reduction: Literal["none","mean","sum"] | None = "mean",`
      `) -> SmplLossTerm`
      - Advanced API using model's native joint order; shapes: `target_positions: (B, J_raw, 2)` (pixels), `weights: (J_raw,) | (B, J_raw)`.
      - Compares 2D targets against the Kornia-projected `output.extras["joints_raw"]`.
  - Notes on Kornia usage:
    - Intrinsics/extrinsics follow typical forms used by Kornia: `K (fx, fy, cx, cy)` within a `(3,3)` matrix; `Tcw` is world→camera homogeneous transform.
    - We rely on `kornia.geometry.camera.cam2pixel` for pinhole projection when using `K/Tcw`. Distortion models (e.g., affine/Kannala-Brandt) can be integrated later via Kornia camera modules.
    - Input and output are in pixel units; choose robustifier `rho` accordingly (pixels).
  - Notes:
    - To optimize camera parameters, pass a camera module with trainable parameters (captured at init). The loss will propagate into the camera.

Pose Prior – VPoser
- `class VPoserPriorLossBuilder(BaseLossBuilder)`
  - Purpose: VPoser-based body pose priors (excluding hands/face).
  - Single source of truth: see `context/tasks/features/keypoint-match/task-vposer-prior.md` for the complete interface and semantics.
  - Interfaces (summarized; refer to the document above):
    - `by_pose_latent(latent, weight)` → weighted L2 on latent.
    - `by_pose(pose, w_pose_fit, w_latent_l2)` → self-reconstruction MSE + latent L2.
  - Notes: Stays separate from keypoint data terms; users compose manually.

Shape Prior
- `class ShapePriorLossBuilder(BaseLossBuilder):`
  - `l2_on_betas(weight: float|Tensor=1.0) -> SmplLossTerm` penalizing `||β||^2`.
  - Optionally: bounds or Gaussian prior parameters in future.

Angle Prior (Heuristic)
- `class AnglePriorLossBuilder(BaseLossBuilder):`
  - `knees_elbows_bending(weight: float|Tensor=1.0, strategy: Literal["smplify","sign"] = "smplify") -> SmplLossTerm`
  - Mimics SMPLify-X’s bending prior on body AA subvector.

Combining Losses
- Users sum individual `SmplLossTerm`s directly, e.g., `total = w1 * loss1(out) + w2 * loss2(out)`.

Data Shapes and Conventions
- Batch-first for everything `(B, ...)`.
- KeypointMatchLossBuilder (3D):
  - Unified joints `(B, 55, 3)`; selection by joint name.
  - Packed API uses native joint space `(B, J_raw, 3)` where `J_raw ∈ {24, 52, 55}`.
  - 3D targets `(B, n, 3)`.
- ProjectedKeypointMatchLossBuilder (2D):
  - Unified selection by name; targets `(B, n, 2)`.
  - Packed API targets `(B, J_raw, 2)` using native joint order.
- Weights/confidences broadcast to `(B, n, 1)`; scalar weights allowed.
- Names resolved against `UnifiedSmplModel.get_joint_names(unified=True)`.

Error Handling & Edge Cases
- Missing joints: selectable policy – ignore (mask out), zero, or treat as NaN and mask.
- Nonexistent names: warn via model’s warn_fn if provided; exclude from loss.
- PCA hands: irrelevant to loss builders; work on unified joints independent of hand parameterization.
- Device/Dtype: targets and weights are moved to model’s device/dtype at build time; rechecked at `forward` if needed.
- Packed API validation: error if `target_positions.shape[1] != J_raw` for the current model type.

Robust Loss Options (PyTorch)
- Built-in robust losses we can leverage instead of custom implementations:
  - `torch.nn.SmoothL1Loss` / `torch.nn.functional.smooth_l1_loss` (aka “Huber” style prior to dedicated class).
  - `torch.nn.HuberLoss` / `torch.nn.functional.huber_loss` (explicit Huber loss; configurable `delta` parameter).
- Mapping from our `robust` argument:
  - `"l2"` → `torch.nn.MSELoss` or `(residual**2)`.
  - `"huber"` → `torch.nn.HuberLoss(delta=rho)` or `F.huber_loss(..., delta=rho)` where `rho` matches the transition point.
  - `"gmof"` (Geman–McClure/GMoF) → not in PyTorch core; we will provide a small internal implementation (mirroring SMPLify-X’s `GMoF`), parameterized by `rho`.
- Docs (PyTorch stable):
  - HuberLoss: https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html
  - SmoothL1Loss: https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html

Example usage inside a term
```python
import torch.nn.functional as F

def robustify(residual, kind: str, rho: float):
    if kind == "l2":
        return (residual ** 2)
    elif kind == "huber":
        # reduction handled later; return elementwise
        return F.huber_loss(residual, torch.zeros_like(residual), delta=rho, reduction="none")
    elif kind == "gmof":
        # Geman–McClure: r^2 / (r^2 + rho^2)
        r2 = (residual ** 2)
        return r2 / (r2 + (rho ** 2))
    else:
        raise ValueError(f"Unknown robust kind: {kind}")
```

Example Usage (illustrative, not executable here)
- 3D keypoint fit data term only:
  - `km = KeypointMatchLossBuilder(model)`
  - `loss_kpts = km.by_target_positions({"left_shoulder": L_sh, "right_shoulder": R_sh}, weights={"left_shoulder": 2.0})`
  - `total = loss_kpts(output)`
- 2D fitting with camera and priors:
  - `proj = ProjectedKeypointMatchLossBuilder(model, camera)`
  - `data2d = proj.by_target_positions({"left_shoulder": L2D, "right_shoulder": R2D}, confidence={"left_shoulder": 1.0, "right_shoulder": 0.8})`
  - `vp = VPoserPriorLossBuilder.from_vposer(model, vposer)`
  - `pose_prior = vp.by_pose_latent(latent, weight=1.0)`
  - `shape_prior = ShapePriorLossBuilder(model).l2_on_betas(0.001)`
  - `total = data2d(output) + pose_prior(output) + shape_prior(output)`
- Packed full-skeleton by index (advanced, 3D):
  - `targets3d = torch.zeros(B, J_raw, 3)`  # user fills in native-ordered 3D keypoints
  - `w = torch.ones(J_raw); w[5] = 0.0`  # omit joint 5
  - `data_packed = km.by_target_positions_packed(targets3d, w)`
  - `total = data_packed(output)`
  - `vp = VPoserPriorLossBuilder.from_vposer(model, vposer)`
  - `pose_prior = vp.by_pose_latent(latent, weight=1.0)`
  - `shape_prior = ShapePriorLossBuilder(model).l2_on_betas(0.001)`
  - `total = data2d(output) + pose_prior(output) + shape_prior(output)`

Optimization Flow (external to builders)
- Users create trainable parameters (e.g., body pose latent, betas, transl, global_orient) and run optimization (LBFGS/Adam).
- A typical closure calls the model to get `UnifiedSmplOutput`, then evaluates individual loss terms, sums, and backprops.
- Staged schedules (camera init → pose → full) can be implemented by adjusting per-term weights across iterations.

Test Plan (high level)
- Unit-test name→index resolution and joint selection against `UnifiedSmplModel.get_joint_names`.
- Unit-test weight broadcasting and masking behavior with synthetic targets.
- Golden tests comparing 2D error to SMPLify-X behavior on a tiny synthetic projection.
- Deterministic seeds; no external I/O; guard VPoser with optional skip.

Future Extensions
- Multi-view 2D supervision (sum over views); per-view camera.
- Temporal priors (velocity/acceleration) for sequences.
- Collision/interpenetration builder analogous to SMPLify-X.
- Detector-specific keypoint mappers (COCO/OpenPose/H36M) as optional adapters.
