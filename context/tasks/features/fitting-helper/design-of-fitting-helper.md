Title: Design — SMPL Keypoint Fitting Helper (3D)

Context
- Goal: Reduce boilerplate for 3D keypoint–based SMPL/SMPL‑H/SMPL‑X fitting by providing a small helper that builds losses, manages priors, and drives an optimization loop via a simple iterator interface.
- Scope: 3D keypoints only (world/model space); 2D projection and camera fitting are explicitly out of scope for this helper.
- Requirements source: context/tasks/task-anything.md (this document is the canonical design for the implementation phase).

Class Overview
- Name: SmplKeypointFittingHelper
- Location: src/smplx_toolbox/fitting/helper.py
- Style: Strongly typed, factory pattern, internal members prefixed with m_, public read-only properties + explicit setters (per project coding guide).
- Models: Works with UnifiedSmplModel for SMPL, SMPL‑H, SMPL‑X.
- Priors: Optional VPoser prior (either by checkpoint path or preloaded VPoserModel), optional L2 pose regularization; easily extensible to add more priors later (e.g., angle/shape) without changing the external API.

Key Responsibilities
- Maintain a registry of data terms and priors (loss terms) with their weights.
- Keep track of 3D keypoint targets by name (unified joint space), with per‑joint weights and optional confidences.
- Provide a generator/iterator API to run optimization step‑by‑step and report structured status each step.
- Control VPoser prior and pose L2 via weights (0 disables); adjust weights at runtime.
- Expose the underlying loss terms for advanced users who want a custom optimization loop.
- Allow registering and managing arbitrary custom loss terms by name.

External Dependencies (existing code)
- smplx_toolbox.core.UnifiedSmplModel / UnifiedSmplInputs / UnifiedSmplOutput
- smplx_toolbox.optimization.KeypointMatchLossBuilder
- smplx_toolbox.optimization.VPoserPriorLossBuilder
- smplx_toolbox.optimization.AnglePriorLossBuilder
- smplx_toolbox.optimization.robustifiers (gmof/huber/l2 via builders_base)
- smplx_toolbox.vposer.load_vposer (when loading by checkpoint)

Public API (with NumPy-style docstrings)

```python
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Literal

import torch
from torch import Tensor, nn

from smplx_toolbox.core import UnifiedSmplModel, UnifiedSmplInputs, UnifiedSmplOutput
from smplx_toolbox.optimization.keypoint_match_builder import SmplLossTerm


class SmplKeypointFittingHelper:
    """Helper for 3D keypoint-based fitting on SMPL/SMPL-H/SMPL-X.

    This class reduces boilerplate by managing keypoint targets, priors
    (e.g., VPoser, L2), and driving an optimization loop via a Python
    iterator that yields per-step status.
    """

    @classmethod
    def from_model(cls, model: UnifiedSmplModel) -> "SmplKeypointFittingHelper":
        """Create a helper bound to a unified SMPL family model.

        Parameters
        ----------
        model : UnifiedSmplModel
            The unified adapter wrapping an SMPL/SMPL-H/SMPL-X model.

        Returns
        -------
        SmplKeypointFittingHelper
            A helper instance bound to ``model``.
        """
        ...

    def set_keypoint_targets(
        self,
        targets: dict[str, Tensor] | Tensor,
        *,
        weights: dict[str, float | Tensor] | float | Tensor | None = None,
        robust: Literal["l2", "huber", "gmof"] = "gmof",
        rho: float = 100.0,
        confidence: dict[str, float | Tensor] | float | Tensor | None = None,
    ) -> None:
        """Set or overwrite 3D keypoint targets by joint name.

        Builds/rebuilds the internal data term via
        ``KeypointMatchLossBuilder.by_target_positions`` using the provided
        robustifier settings and (optional) per-joint weights and confidences.

        Parameters
        ----------
        targets : dict[str, Tensor] or torch.Tensor
            - Mapping from unified joint name to target 3D positions of shape
              ``(B, 3)``; or
            - A packed tensor of shape ``(B, K, 3)`` following the same
              joint order and semantics as :attr:`NamedPose.intrinsic_pose`
              for the current model type (pelvis excluded).
        weights : dict[str, float or Tensor] or float or Tensor or None, optional
            Per-joint or scalar weights.
            - If ``targets`` is a dict, keys must be a subset of ``targets``.
            - If ``targets`` is a packed tensor ``(B, K, 3)``, accepts
              ``(B, K)``, ``(K,)``, scalar, or ``(B, K, 1)``.
        robust : {"l2", "huber", "gmof"}, optional
            Robust penalty for the data term. Defaults to ``"gmof"``.
        rho : float, optional
            Scale parameter for the robustifier. Defaults to ``100.0``.
        confidence : dict[str, float or Tensor] or float or Tensor or None, optional
            Additional confidence multipliers (same broadcasting rules as
            ``weights``). When provided, the effective weight is
            ``weights * confidence``.
        """
        ...

    def set_pose_l2_reg(self, weight: float | Tensor) -> None:
        """Enable global L2 regularization on optimized pose parameters.

        Applies to whichever pose tensors the user marked with
        ``requires_grad=True`` (e.g., ``root_orient``, ``pose_body``, and
        optionally left/right hand poses).

        Parameters
        ----------
        weight : float or Tensor
            Global scalar weight multiplied by the sum of squared elements in
            the optimized pose tensors.
            If ``0`` (or a zero tensor), pose L2 is disabled.
        """
        ...

    # --- VPoser control (revised naming) ---
    def vposer_init(self, *, ckpt_path: str | Path | None = None, vposer_model: nn.Module | None = None) -> None:
        """Initialize VPoser by loading from checkpoint or attaching a model.

        Parameters
        ----------
        ckpt_path : str or Path, optional
            Filesystem path to ``vposer-v2.ckpt``. Mutually exclusive with
            ``vposer_model``.
        vposer_model : nn.Module, optional
            A pre-loaded VPoser model. Mutually exclusive with ``ckpt_path``.

        Raises
        ------
        ValueError
            If neither ``ckpt_path`` nor ``vposer_model`` is provided.
        """
        ...

    def vposer_set_weight(self, *, w_pose_fit: float | Tensor = 0.0, w_latent_l2: float | Tensor = 0.0) -> None:
        """Set VPoser prior weights; zero disables the prior.

        Parameters
        ----------
        w_pose_fit : float or Tensor, optional
            Weight for the self-reconstruction MSE term. ``0`` disables this term.
            Defaults to ``0.0`` (disabled).
        w_latent_l2 : float or Tensor, optional
            Weight for the latent L2 term. ``0`` disables this term.
            Defaults to ``0.0`` (disabled).
        Notes
        -----
        The VPoser loss is active only if a VPoser model has been loaded via
        :meth:`vposer_init` and at least one of ``w_pose_fit`` or ``w_latent_l2``
        is non-zero.
        """
        ...

    def vposer_is_enabled(self) -> bool:
        """Return whether the VPoser prior is currently active.

        Returns
        -------
        bool
            True if a VPoser model is loaded and at least one of the weights
            ``w_pose_fit`` or ``w_latent_l2`` is non-zero; otherwise False.
        """
        ...

    def get_loss_terms(self) -> list[tuple[str, nn.Module, float]]:
        """Return active loss terms and their weights for external loops.

        Returns
        -------
        list of (str, nn.Module, float)
            Triplets ``(name, term_module, weight)`` for each active term.
        """
        ...

    # --- Custom loss registry ---
    def set_custom_loss(self, name: str, loss_term: SmplLossTerm | None, *, weight: float = 1.0) -> None:
        """Set or remove a named custom loss term with an outer weight.

        This allows adding arbitrary losses built via existing builders (e.g.,
        keypoint/pose/angle prior builders) or user-defined terms deriving from
        :class:`SmplLossTerm` and operating on ``UnifiedSmplOutput``.

        Parameters
        ----------
        name : str
            Unique key for the term. Setting an existing name overwrites it.
        loss_term : SmplLossTerm or None
            The loss module to register. If ``None``, the named term is
            removed and its weight is discarded.
        weight : float, optional
            An outer scalar multiplier applied to the term's value at runtime.
            Defaults to ``1.0``. This multiplies any internal weighting the
            term may already apply.
        Notes
        -----
        The helper aligns the term's device/dtype to the model during
        registration.
        """
        ...

    def get_custom_loss(self, name: str) -> tuple[SmplLossTerm, float] | None:
        """Return ``(term, weight)`` for a registered name, or ``None``."""
        ...

    def get_custom_losses(self) -> dict[str, tuple[SmplLossTerm, float]]:
        """Return a mapping of all registered custom loss terms to their weights."""
        ...

    def init_fitting(
        self,
        initial: UnifiedSmplInputs,
        *,
        optimizer: Literal["adam", "lbfgs"] = "adam",
        lr: float = 0.05,
        num_iter_per_step: int = 10,
    ) -> Iterator["FitStepStatus"]:
        """Prepare the optimizer and return a step iterator.

        Scans ``initial`` for tensors with ``requires_grad=True`` to collect
        trainable parameters, initializes the optimizer, and returns an
        iterator. Each "next" performs one forward/backward/step and yields a
        :class:`FitStepStatus`.

        Parameters
        ----------
        initial : UnifiedSmplInputs
            Initial pose/shape/translation parameters. Set
            ``requires_grad=True`` on tensors you want to optimize.
        optimizer : {"adam", "lbfgs"}, optional
            Optimizer type. Defaults to ``"adam"``.
        lr : float, optional
            Learning rate for the optimizer. Defaults to ``0.05``.
        num_iter_per_step : int, optional
            Number of internal optimization iterations to perform per
            ``next()`` call before yielding a status. Defaults to ``10``.
            Increasing this reduces the number of status objects constructed
            and can lower Python overhead during long runs.
        Notes
        -----
        The iterator does not stop automatically. Users fully control the
        number of steps by how many times they call ``next()``.

        Returns
        -------
        Iterator[FitStepStatus]
            Yields per-step status with total and per-term losses.
        """
        ...

    # No terminate/last_output/last_params helpers; the iterator status contains
    # both the current UnifiedSmplInputs and UnifiedSmplOutput on each step.
```

Returned Status Type (attrs)
- File: src/smplx_toolbox/fitting/types.py
-
```python
from attrs import define, field
from typing import Any

@define(kw_only=True)
class FitStepStatus:
    step: int
    loss_total: float
    loss_terms: dict[str, float]
    params: UnifiedSmplInputs
    output: UnifiedSmplOutput
    grad_norm: float | None = None
    extras: dict[str, Any] = field(factory=dict)
```

Internal State & Members (read‑only properties)

```python
m_model: UnifiedSmplModel  # property model
m_targets: dict[str, Tensor]
m_weights: dict[str, Tensor | float] | float | Tensor | None
m_confidence: dict[str, Tensor | float] | float | Tensor | None
m_data_term: nn.Module | None
m_pose_l2_weight: Tensor | None
m_vposer: nn.Module | None
m_vposer_term: nn.Module | None
m_vposer_weights: tuple[Tensor, Tensor] | None
m_custom_terms: dict[str, tuple[SmplLossTerm, float]]
m_optimizer: torch.optim.Optimizer | None
m_trainable_params: list[torch.nn.Parameter]
m_step: int
m_vposer_enabled: bool
```

Data Term Build Strategy
- Build a single keypoint data term covering all configured targets, with
  merged weights and confidences.
  - If targets were provided by name (dict), build via
    ``KeypointMatchLossBuilder.by_target_positions`` directly.
  - If targets were provided as a packed tensor ``(B, K, 3)`` in
    :attr:`NamedPose.intrinsic_pose` order, convert to a name→tensor mapping
    using the model's intrinsic joint order (pelvis excluded) and then use
    ``by_target_positions``.
- Robust kind and rho are provided via ``set_keypoint_targets`` and baked into
  the created term; to change robustifier settings, call
  ``set_keypoint_targets`` again.
- Overwriting semantics: each call to ``set_keypoint_targets`` replaces the previous targets and rebuilds the data term. To clear all targets, call ``set_keypoint_targets(targets={})``.
- If m_targets is empty, m_data_term = None.

Loss Assembly (each step)
- Compute total loss as a weighted sum of all active terms:
  - data: weight implicitly 1.0 (the builder already incorporates per‑joint weights)
  - vposer: included only if a VPoser model is loaded and any VPoser weight
    is non-zero; the term internally applies ``(w_pose_fit, w_latent_l2)``.
  - pose_l2: m_pose_l2_weight * sum(param**2) over all optimized pose parameters present in current initial (root_orient, pose_body, and any hand poses the user enabled)
  - custom: for each (name, (term, w)) in m_custom_terms, compute
    ``loss_name = w * term(output)`` and add to total; record per-term value
    under that name.
- The helper does not impose other priors by default; extensibility hooks allow new priors later.

Parameter Selection & Optimizer
- Users provide an initial UnifiedSmplInputs with tensors (e.g., root_orient, pose_body, left_hand_pose, right_hand_pose). The helper scans those tensors for requires_grad=True and collects them as trainable parameters.
  - Default optimizer is Adam(lr); LBFGS support is planned with a simple closure.
  - No automatic stopping and no per-step gradient clipping knob; users control loop length externally.

Device & Dtype
- All configured tensors/terms (including custom terms) are moved/aligned to the model’s device/dtype at registration/build time.
- The helper never changes the device of the wrapped model; users move the model beforehand as needed (UnifiedSmplModel wraps a smplx model).

Errors & Validation
- set_keypoint_targets: accepts either name→tensor mapping or packed tensor.
  Validates shapes ``(B,3)`` for named targets and ``(B,K,3)`` for packed
  targets (K must match the model's intrinsic joint count excluding pelvis).
  Enforces consistent batch size, validates optional ``weights``/``confidence``
  broadcastability, and warns/ignores unknown joint names via the model’s
  warn function when possible.
- vposer_init: requires either ckpt_path or vposer_model; raises ValueError if both are None.
- init_fitting: raises ValueError if no trainable parameters were found (requires_grad=False on all provided tensors).

Extensibility Points
- Arbitrary terms: Use ``set_custom_loss(name, term, weight=...)`` to register
  any :class:`SmplLossTerm` with an outer weight. Use
  ``set_custom_loss(name, None)`` to remove it. Query via
  ``get_custom_loss(name)`` (returns ``(term, weight)``) or
  ``get_custom_losses()`` (mapping to ``(term, weight)``).

Example Usage (illustrative, not code to ship)
```python
model = UnifiedSmplModel.from_smpl_model(base)
helper = SmplKeypointFittingHelper.from_model(model)

# Configure targets
targets = {
  "left_wrist": torch.randn(B, 3, device=model.device, dtype=model.dtype),
  "right_foot": torch.randn(B, 3, device=model.device, dtype=model.dtype),
}
helper.set_keypoint_targets(targets, weights={"left_wrist": 2.0})

# Enable priors
helper.set_pose_l2_reg(weight=1e-2)
helper.vposer_init(ckpt_path="data/vposer/vposer-v2.ckpt")
helper.vposer_set_weight(w_pose_fit=1.0, w_latent_l2=0.1)

# Prepare trainable parameters
root = torch.zeros(B, 3, device=model.device, dtype=model.dtype, requires_grad=True)
pose = torch.zeros(B, 63, device=model.device, dtype=model.dtype, requires_grad=True)
init = UnifiedSmplInputs(root_orient=root, pose_body=pose)

it = helper.init_fitting(init, optimizer="adam", lr=0.05)
for i, status in zip(range(200), it):  # user controls loop length
    print(status.step, status.loss_total, status.loss_terms)
out = status.output
```

```python
# Add a custom angle prior term (example)
from smplx_toolbox.optimization.angle_prior_builder import AnglePriorLossBuilder

angle_term = AnglePriorLossBuilder.from_model(model).build()
helper.set_custom_loss("angle_prior", angle_term, weight=0.2)

# Later, remove it:
helper.set_custom_loss("angle_prior", None)
```

```python
# Alternatively, configure targets via packed tensor matching NamedPose order
K = len(model.get_joint_names(unified=True)) - 1  # pelvis excluded
packed_targets = torch.randn(B, K, 3, device=model.device, dtype=model.dtype)
packed_weights = torch.ones(B, K, device=model.device, dtype=model.dtype) * 0.5
helper.set_keypoint_targets(packed_targets, weights=packed_weights, robust="gmof", rho=50.0)
```

Non‑Goals
- No 2D keypoints or projection/camera optimization.
- No multi‑frame or multi‑view scheduling/aggregation (single frame per helper session).

Implementation Plan (high‑level)
1) Create module src/smplx_toolbox/fitting/helper.py with the SmplKeypointFittingHelper class skeleton and strongly typed API.
2) Implement configuration methods (targets with overwrite semantics — including robust args —, pose L2 via ``set_pose_l2_reg(weight)`` where 0 disables, VPoser via ``vposer_set_weight`` where 0 disables) with validation and internal term rebuilds; add custom loss registry (``set_custom_loss``, ``get_custom_loss``, ``get_custom_losses``).
3) Implement init_fitting and iterator/generator that performs: forward → loss assembly → backward → optimizer step → status yield.
4) Add FitStepStatus attrs type in src/smplx_toolbox/fitting/types.py.
5) Add minimal unit tests that mock targets and run a few steps, ensuring iterator yields statuses and terms are computed; reuse patterns from unittests/fitting/test_keypoint_match.py.
6) Provide concise docs and usage examples in docs/fitting/helper.md.
