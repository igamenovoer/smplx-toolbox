"""Helper for 3D keypoint-based SMPL/SMPL-X fitting.

Implements the design in context/tasks/features/fitting-helper/design-of-fitting-helper.md
with APIs to set keypoint targets, priors, VPoser, DOF toggles, and an
iterator-driven optimization loop.
"""

from __future__ import annotations

from typing import Any, Iterator, Literal

import torch
import torch.nn as nn
from torch import Tensor

from smplx_toolbox.core import UnifiedSmplModel, UnifiedSmplInputs, UnifiedSmplOutput
from smplx_toolbox.optimization import KeypointMatchLossBuilder, SmplLossTerm
from smplx_toolbox.optimization.pose_prior_vposer_builder import (
    VPoserPriorLossBuilder,
)
from smplx_toolbox.vposer.model import VPoserModel

from .types import FitStepStatus


class SmplKeypointFittingHelper:
    """End‑to‑end 3D keypoint fitting helper for SMPL/SMPL‑H/SMPL‑X.

    This utility class streamlines keypoint‑based optimization by managing
    target specification, robust data terms, priors (VPoser and L2), degrees‑of‑
    freedom (DOF) toggles, and a simple iterator‑driven optimization loop.

    Attributes
    ----------
    m_model : UnifiedSmplModel | None
        Bound unified model adapter. Set via ``from_model``.
    m_keypts_targets : dict[str, Tensor]
        Mapping ``joint_name -> (B, 3)`` target positions in model units.
    m_keypts_weights : dict[str, Tensor | float] | float | Tensor | None
        Optional per‑joint weights (mapping) or per‑sample scalar/tensor weight.
    m_keypts_confidence : dict[str, Tensor | float] | float | Tensor | None
        Optional confidences in the same forms as weights.
    m_keypts_data_term : nn.Module | None
        Compiled data‑term module produced by ``KeypointMatchLossBuilder``.
    m_pose_l2_weight : Tensor | None
        L2 weight for pose regularization. ``None`` disables pose L2.
    m_shape_l2_weight : Tensor | None
        L2 weight for shape regularization. ``None`` disables shape L2.
    m_vposer : VPoserModel | None
        VPoser model instance when VPoser prior is used.
    m_vposer_term : nn.Module | None
        Reserved for a prebuilt VPoser term (currently rebuilt per step).
    m_vposer_weights : tuple[Tensor, Tensor] | None
        Tuple ``(w_pose_fit, w_latent_l2)`` for VPoser prior weights.
    m_vposer_enabled : bool
        Convenience flag indicating whether VPoser prior contributes.
    m_custom_terms : dict[str, tuple[SmplLossTerm, float]]
        Registry of named custom loss terms and their scalar weights.
    m_enable_global_orient : bool
        DOF toggle for root orientation optimization.
    m_enable_global_translation : bool
        DOF toggle for global translation optimization.
    m_enable_shape_deform : bool
        DOF toggle for shape (betas) optimization.
    m_optimizer : torch.optim.Optimizer | None
        The optimizer instance created by ``init_fitting``.
    m_trainable_params : list[torch.nn.Parameter]
        Flat list of trainable tensors handed to the optimizer.
    m_step : int
        0‑based outer step counter, incremented on each ``yield``.

    Notes
    -----
    - Call ``from_model`` to bind a model, then configure targets/priors/DOFs,
      then call ``init_fitting`` and iterate the returned generator.
    - The helper updates the same ``nn.Parameter`` objects supplied in
      ``UnifiedSmplInputs`` in place (see ``init_fitting`` docstring).
    """

    def __init__(self) -> None:
        # Bound model
        self.m_model: UnifiedSmplModel | None = None

        # Keypoint targets and data term
        self.m_keypts_targets: dict[str, Tensor] = {}
        self.m_keypts_weights: dict[str, Tensor | float] | float | Tensor | None = None
        self.m_keypts_confidence: dict[str, Tensor | float] | float | Tensor | None = None
        self.m_keypts_data_term: nn.Module | None = None

        # Regularization weights
        self.m_pose_l2_weight: Tensor | None = None
        self.m_shape_l2_weight: Tensor | None = None

        # VPoser prior
        self.m_vposer: VPoserModel | None = None
        self.m_vposer_term: nn.Module | None = None
        self.m_vposer_weights: tuple[Tensor, Tensor] | None = None  # (w_pose_fit, w_latent_l2)
        self.m_vposer_enabled: bool = False

        # Custom terms
        self.m_custom_terms: dict[str, tuple[SmplLossTerm, float]] = {}

        # DOF toggles
        self.m_enable_global_orient: bool = True
        self.m_enable_global_translation: bool = True
        self.m_enable_shape_deform: bool = True

        # Optimizer state
        self.m_optimizer: torch.optim.Optimizer | None = None
        self.m_trainable_params: list[torch.nn.Parameter] = []
        self.m_step: int = 0

    # --- Factory ---
    @classmethod
    def from_model(cls, model: UnifiedSmplModel) -> "SmplKeypointFittingHelper":
        """Create a helper bound to a specific unified model.

        Parameters
        ----------
        model : UnifiedSmplModel
            The unified SMPL family model to optimize.

        Returns
        -------
        SmplKeypointFittingHelper
            A new helper instance bound to ``model``.
        """
        inst = cls()
        inst.m_model = model
        return inst

    # --- Configuration: Data term ---
    def set_keypoint_targets(
        self,
        targets: dict[str, Tensor],
        *,
        weights: dict[str, float | Tensor] | float | Tensor | None = None,
        robust: Literal["l2", "huber", "gmof"] = "gmof",
        rho: float = 100.0,
        confidence: dict[str, float | Tensor] | float | Tensor | None = None,
    ) -> None:
        """Set or overwrite 3D keypoint targets by joint name.

        Parameters
        ----------
        targets : dict[str, Tensor]
            Mapping ``name -> (B, 3)`` of target positions. All entries must
            share the same batch size and are moved to the model's device/dtype.
        weights : dict[str, float | Tensor] | float | Tensor | None, optional
            Optional weights for the data term. Accepts:
            - per‑joint mapping ``name -> scalar|tensor``;
            - per‑sample tensor ``(B,)`` or ``(B, 1)``;
            - scalar. ``None`` means uniform weighting.
        robust : {"l2", "huber", "gmof"}, optional
            Robustifier type for the data term. Defaults to ``"gmof"``.
        rho : float, optional
            Scale parameter for the robustifier. Defaults to ``100.0``.
        confidence : dict[str, float | Tensor] | float | Tensor | None, optional
            Optional confidences with the same accepted forms as ``weights``.

        Notes
        -----
        Passing an empty ``targets`` dict clears the current data term.
        """
        self._ensure_model()

        # Clear if empty
        if not targets:
            self.m_keypts_targets = {}
            self.m_keypts_weights = None
            self.m_keypts_confidence = None
            self.m_keypts_data_term = None
            return

        # Validate shapes and move to device/dtype
        model = self.model
        dev, dt = model.device, model.dtype
        fixed_targets: dict[str, Tensor] = {}
        B_ref: int | None = None
        for name, t in targets.items():
            if not isinstance(t, torch.Tensor):
                raise TypeError("targets values must be torch.Tensor")
            t2 = t.to(device=dev, dtype=dt)
            if t2.dim() != 2 or t2.shape[-1] != 3:
                raise ValueError("each target tensor must have shape (B, 3)")
            if B_ref is None:
                B_ref = int(t2.shape[0])
            elif int(t2.shape[0]) != B_ref:
                raise ValueError("all target tensors must share the same batch size B")
            fixed_targets[name] = t2
        assert B_ref is not None
        B = B_ref

        # Normalize weights and confidence
        weights_prepared = self._prepare_per_sample_or_mapping(weights, B, list(fixed_targets.keys()))
        conf_prepared = self._prepare_per_sample_or_mapping(confidence, B, list(fixed_targets.keys()))

        # Build data term
        km = KeypointMatchLossBuilder.from_model(model)
        self.m_keypts_targets = fixed_targets
        self.m_keypts_weights = weights_prepared["raw"]
        self.m_keypts_confidence = conf_prepared["raw"]
        # Adapt per-sample raw tensors to (B, n) for builder
        names = list(fixed_targets.keys())
        n = len(names)
        w_builder = weights_prepared["builder"]
        c_builder = conf_prepared["builder"]
        if isinstance(w_builder, torch.Tensor):
            if w_builder.dim() == 1 and w_builder.numel() == B:
                w_builder = w_builder.view(B, 1).expand(B, n)
            elif w_builder.dim() == 2 and w_builder.shape == (B, 1):
                w_builder = w_builder.expand(B, n)
        if isinstance(c_builder, torch.Tensor):
            if c_builder.dim() == 1 and c_builder.numel() == B:
                c_builder = c_builder.view(B, 1).expand(B, n)
            elif c_builder.dim() == 2 and c_builder.shape == (B, 1):
                c_builder = c_builder.expand(B, n)
        self.m_keypts_data_term = km.by_target_positions(
            fixed_targets,
            weights=w_builder,
            robust=robust,
            rho=float(rho),
            confidence=c_builder,
            reduction="mean",
        )

    # --- Configuration: DOF toggles ---
    def set_dof_global_orient(self, enable: bool) -> None:
        """Enable or disable optimization of global orientation.

        Parameters
        ----------
        enable : bool
            If ``True``, include ``global_orient`` in trainables; otherwise
            detach and keep constant during optimization.
        """
        self.m_enable_global_orient = bool(enable)

    def set_dof_global_translation(self, enable: bool) -> None:
        """Enable or disable optimization of global translation.

        Parameters
        ----------
        enable : bool
            If ``True``, include ``trans`` in trainables; otherwise detach.
        """
        self.m_enable_global_translation = bool(enable)

    def set_dof_shape_deform(self, enable: bool) -> None:
        """Enable or disable optimization of shape coefficients (betas).

        Parameters
        ----------
        enable : bool
            If ``True``, include ``betas`` in trainables; otherwise detach.
        """
        self.m_enable_shape_deform = bool(enable)

    # --- Configuration: Regularization ---
    def set_reg_pose_l2(self, weight: float | Tensor) -> None:
        """Set L2 regularization weight for pose terms.

        Parameters
        ----------
        weight : float or torch.Tensor
            Scalar weight. ``0`` disables pose L2; stored on model device/dtype.
        """
        self._ensure_model()
        w = weight if isinstance(weight, torch.Tensor) else torch.tensor(weight)
        w = w.to(device=self.model.device, dtype=self.model.dtype)
        if float(w.detach().cpu().item()) == 0.0:
            self.m_pose_l2_weight = None
        else:
            self.m_pose_l2_weight = w

    def set_reg_shape_l2(self, weight: float | Tensor = 1e-2) -> None:
        """Set L2 regularization weight for shape coefficients (betas).

        Parameters
        ----------
        weight : float or torch.Tensor, optional
            Scalar weight. ``0`` disables shape L2. Defaults to ``1e-2``.
        """
        self._ensure_model()
        w = weight if isinstance(weight, torch.Tensor) else torch.tensor(weight)
        w = w.to(device=self.model.device, dtype=self.model.dtype)
        if float(w.detach().cpu().item()) == 0.0:
            self.m_shape_l2_weight = None
        else:
            self.m_shape_l2_weight = w

    # --- VPoser ---
    def vposer_init(self, *, vposer_model: VPoserModel) -> None:
        """Attach a VPoser model for use as a prior.

        Parameters
        ----------
        vposer_model : VPoserModel
            Loaded VPoser model compatible with the current device/dtype.
        """
        self.m_vposer = vposer_model

    def vposer_set_reg(
        self, *, w_pose_fit: float | Tensor = 0.0, w_latent_l2: float | Tensor = 0.0
    ) -> None:
        """Configure VPoser prior weights.

        Parameters
        ----------
        w_pose_fit : float or torch.Tensor, optional
            Weight for VPoser self‑reconstruction error on ``pose_body``.
        w_latent_l2 : float or torch.Tensor, optional
            Weight for L2 magnitude of VPoser latent code.
        """
        self._ensure_model()
        w1 = w_pose_fit if isinstance(w_pose_fit, torch.Tensor) else torch.tensor(w_pose_fit)
        w2 = w_latent_l2 if isinstance(w_latent_l2, torch.Tensor) else torch.tensor(w_latent_l2)
        w1 = w1.to(device=self.model.device, dtype=self.model.dtype)
        w2 = w2.to(device=self.model.device, dtype=self.model.dtype)
        self.m_vposer_weights = (w1, w2)
        self.m_vposer_enabled = bool((float(w1.detach().cpu()) != 0.0) or (float(w2.detach().cpu()) != 0.0)) and (
            self.m_vposer is not None
        )

    def vposer_is_enabled(self) -> bool:
        """Return whether the VPoser prior will contribute to the loss.

        Returns
        -------
        bool
            ``True`` if a VPoser model is attached and at least one of the
            VPoser weights is non‑zero; ``False`` otherwise.
        """
        return self.m_vposer is not None and self.m_vposer_weights is not None and (
            float(self.m_vposer_weights[0].detach().cpu()) != 0.0
            or float(self.m_vposer_weights[1].detach().cpu()) != 0.0
        )

    # --- Custom loss registry ---
    def set_custom_loss(self, name: str, loss_term: SmplLossTerm | None, *, weight: float = 1.0) -> None:
        """Register or remove a custom named loss term.

        Parameters
        ----------
        name : str
            Unique loss name used as a key in diagnostics.
        loss_term : SmplLossTerm or None
            A callable module mapping ``UnifiedSmplOutput -> Tensor``. If
            ``None``, the named term is removed.
        weight : float, optional
            Scalar multiplier applied to the term. Defaults to ``1.0``.
        """
        if loss_term is None:
            self.m_custom_terms.pop(name, None)
            return
        self.m_custom_terms[name] = (loss_term, float(weight))

    def get_custom_loss(self, name: str) -> tuple[SmplLossTerm, float] | None:
        """Retrieve a registered custom loss term.

        Parameters
        ----------
        name : str
            Loss name supplied to ``set_custom_loss``.

        Returns
        -------
        tuple[SmplLossTerm, float] | None
            The ``(term, weight)`` pair if present; otherwise ``None``.
        """
        return self.m_custom_terms.get(name)

    def get_custom_losses(self) -> dict[str, tuple[SmplLossTerm, float]]:
        """Return a shallow copy of all registered custom loss terms.

        Returns
        -------
        dict[str, tuple[SmplLossTerm, float]]
            Mapping from loss name to ``(term, weight)``.
        """
        return dict(self.m_custom_terms)

    def get_loss_terms(self) -> list[tuple[str, nn.Module, float]]:
        """List currently configured loss terms for inspection.

        Returns
        -------
        list[tuple[str, nn.Module, float]]
            A list of ``(name, module, weight)`` for data, VPoser (if enabled),
            and any registered custom losses. The returned modules are ready to
            be called with ``UnifiedSmplOutput``.
        """
        terms: list[tuple[str, nn.Module, float]] = []
        if self.m_keypts_data_term is not None:
            terms.append(("data", self.m_keypts_data_term, 1.0))
        if self.vposer_is_enabled():
            # Placeholder; the actual VPoser term is built dynamically per-step
            vp_builder = VPoserPriorLossBuilder.from_vposer(self.model, self.m_vposer)  # type: ignore[arg-type]
            dummy = nn.Identity()  # type: ignore[assignment]
            terms.append(("vposer", dummy, 1.0))
        for k, (term, w) in self.m_custom_terms.items():
            terms.append((k, term, float(w)))
        return terms

    # --- Optimization ---
    def init_fitting(
        self,
        initial: UnifiedSmplInputs,
        *,
        optimizer: Literal["adam", "lbfgs"] = "adam",
        lr: float = 0.05,
        num_iter_per_step: int = 10,
    ) -> Iterator[FitStepStatus]:
        """Prepare the optimizer and return a step iterator.

        In‑place update behavior
        ------------------------
        The helper does not copy pose/shape tensors. Instead, it collects the
        exact ``nn.Parameter`` objects from ``initial`` (according to the DOF
        toggles) and builds the optimizer over those references. During
        optimization, ``opt.step()`` mutates these parameters in place.

        - Enabled DOFs (e.g., ``named_pose.intrinsic_pose``, ``global_orient``,
          ``trans``, ``betas`` when present and marked trainable) are added to
          the optimizer and thus updated in place.
        - Disabled DOFs are excluded from optimization and are passed to the
          model via ``.detach()`` so they remain constant and accumulate no
          gradients.
        - Each iterator step constructs a fresh ``UnifiedSmplInputs`` wrapper,
          but the fields inside reference the same live tensors, so updates are
          immediately visible in subsequent iterations.

        Practical implications
        ----------------------
        - To resume optimization from the latest state, you can reuse the same
          ``initial`` object and call ``init_fitting`` again.
        - To restart from a pristine state, clone the relevant tensors up front
          and restore them later (e.g., ``param.data.copy_(backup)``), or create
          a new ``UnifiedSmplInputs`` with new ``nn.Parameter`` instances.
        - The yielded ``FitStepStatus.params`` also contains live references to
          the current tensors (not deep copies) for inspection or reuse.

        Parameters
        ----------
        initial : UnifiedSmplInputs
            Input container holding the parameters to optimize. The helper uses
            the contained ``nn.Parameter`` objects directly.
        optimizer : {"adam", "lbfgs"}, optional
            Optimizer type to use. Defaults to ``"adam"``.
        lr : float, optional
            Optimizer learning rate. Defaults to ``0.05``.
        num_iter_per_step : int, optional
            Number of inner optimization steps to run per yielded outer step.
            Use ``1`` for maximal per‑step insight; increase to reduce Python
            overhead in long runs.

        Yields
        ------
        FitStepStatus
            Per‑step snapshot containing total loss, term breakdown, the live
            inputs, and the latest model output.
        """
        self._ensure_model()
        self.m_step = 0

        # Collect trainable params according to DOF toggles
        trainables: list[torch.nn.Parameter] = []

        # NamedPose intrinsic
        npz = initial.named_pose
        if npz is not None and hasattr(npz, "intrinsic_pose") and isinstance(npz.intrinsic_pose, torch.Tensor):
            if isinstance(npz.intrinsic_pose, torch.nn.Parameter) and npz.intrinsic_pose.requires_grad:
                trainables.append(npz.intrinsic_pose)

        # Root orient
        if self.m_enable_global_orient and initial.global_orient is not None and isinstance(initial.global_orient, torch.Tensor):
            if isinstance(initial.global_orient, torch.nn.Parameter) and initial.global_orient.requires_grad:
                trainables.append(initial.global_orient)

        # Translation
        if self.m_enable_global_translation and initial.trans is not None and isinstance(initial.trans, torch.Tensor):
            if isinstance(initial.trans, torch.nn.Parameter) and initial.trans.requires_grad:
                trainables.append(initial.trans)

        # Shape
        if self.m_enable_shape_deform and initial.betas is not None and isinstance(initial.betas, torch.Tensor):
            if isinstance(initial.betas, torch.nn.Parameter) and initial.betas.requires_grad:
                trainables.append(initial.betas)

        if not trainables:
            raise ValueError("No trainable parameters found. Ensure requires_grad=True and DOFs enabled.")

        # Optimizer
        if optimizer == "adam":
            opt: torch.optim.Optimizer = torch.optim.Adam(trainables, lr=float(lr))
        elif optimizer == "lbfgs":
            opt = torch.optim.LBFGS(trainables, lr=float(lr))
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        self.m_optimizer = opt
        self.m_trainable_params = trainables

        def _make_inputs(detach_for_disabled: bool = True) -> UnifiedSmplInputs:
            # Use initial as source but respect DOF gates by detaching disabled tensors
            go = initial.global_orient
            if not self.m_enable_global_orient and go is not None:
                go = go.detach() if isinstance(go, torch.Tensor) else go
            tr = initial.trans
            if not self.m_enable_global_translation and tr is not None:
                tr = tr.detach() if isinstance(tr, torch.Tensor) else tr
            b = initial.betas
            if not self.m_enable_shape_deform and b is not None:
                b = b.detach() if isinstance(b, torch.Tensor) else b
            return UnifiedSmplInputs(
                named_pose=initial.named_pose,
                global_orient=go,
                trans=tr,
                betas=b,
            )

        # Prepare data term if not already
        if self.m_keypts_data_term is None and self.m_keypts_targets:
            # Rebuild with defaults
            km = KeypointMatchLossBuilder.from_model(self.model)
            self.m_keypts_data_term = km.by_target_positions(
                self.m_keypts_targets, weights=self.m_keypts_weights, confidence=self.m_keypts_confidence, reduction="mean"
            )

        # Iterator loop
        while True:
            # Run several inner iters per step
            for _ in range(max(1, int(num_iter_per_step))):
                opt.zero_grad()
                current_inputs = _make_inputs()
                out = self.model.forward(current_inputs)
                loss, terms = self._assemble_loss(out, current_inputs)

                if isinstance(opt, torch.optim.LBFGS):
                    def closure() -> Tensor:  # type: ignore[no-redef]
                        opt.zero_grad()
                        out_c = self.model.forward(_make_inputs())
                        loss_c, _ = self._assemble_loss(out_c, _make_inputs())
                        loss_c.backward()
                        return loss_c

                    loss = opt.step(closure)  # type: ignore[assignment]
                else:
                    loss.backward()
                    opt.step()

            # Diagnostics
            try:
                loss_val = float(loss.detach().item())  # type: ignore[arg-type]
            except Exception:
                loss_val = float(loss.item())  # type: ignore[attr-defined]

            grad_norm = None
            try:
                total = 0.0
                for p in trainables:
                    if p.grad is not None:
                        total += float(p.grad.detach().norm().item())
                grad_norm = total
            except Exception:
                grad_norm = None

            status = FitStepStatus(
                step=self.m_step,
                loss_total=loss_val,
                loss_terms=terms,
                params=_make_inputs(detach_for_disabled=False),
                output=out,
                grad_norm=grad_norm,
            )
            self.m_step += 1
            yield status

    # --- Internals ---
    def _ensure_model(self) -> None:
        """Validate that a model has been bound via ``from_model``.

        Raises
        ------
        ValueError
            If no model is currently bound.
        """
        if self.m_model is None:
            raise ValueError("Helper not initialized. Use from_model(model).")

    @property
    def model(self) -> UnifiedSmplModel:
        """Return the bound unified model.

        Returns
        -------
        UnifiedSmplModel
            The model previously provided to ``from_model``.
        """
        self._ensure_model()
        assert self.m_model is not None
        return self.m_model

    def _prepare_per_sample_or_mapping(
        self,
        w: dict[str, float | Tensor] | float | Tensor | None,
        batch: int,
        names: list[str],
    ) -> dict[str, Any]:
        """Prepare weight/confidence inputs for the data‑term builder.

        Parameters
        ----------
        w : dict[str, float | Tensor] | float | Tensor | None
            Per‑joint mapping, per‑sample vector ``(B,)`` or ``(B, 1)``, scalar,
            or ``None``.
        batch : int
            Batch size ``B`` inferred from targets.
        names : list[str]
            Ordered joint names used to align per‑sample tensors when needed.

        Returns
        -------
        dict
            A dict with keys:
            - ``raw``: value to store in member fields; and
            - ``builder``: value adapted for ``by_target_positions``.

        Raises
        ------
        ValueError
            If an unsupported tensor shape is provided (e.g., per‑joint array).
        """
        if w is None:
            return {"raw": None, "builder": None}

        # If mapping, pass through
        if isinstance(w, dict):
            # Move any tensor scalars to device/dtype
            fixed: dict[str, float | Tensor] = {}
            for k, v in w.items():
                if isinstance(v, torch.Tensor):
                    fixed[k] = v.to(device=self.model.device, dtype=self.model.dtype)
                else:
                    fixed[k] = float(v)
            return {"raw": fixed, "builder": fixed}

        # Tensor/scalar path
        if isinstance(w, torch.Tensor):
            wt = w.to(device=self.model.device, dtype=self.model.dtype)
            # Only per-sample accepted: (B,) or (B,1); scalar allowed via 0-dim
            if wt.dim() == 0:
                # scalar -> let builder broadcast as scalar
                return {"raw": wt, "builder": float(wt.detach().cpu().item())}
            if wt.dim() == 1 and wt.numel() == batch:
                # Expand to (B, n) in builder by passing a lambda is not possible; instead we will
                # replicate per name via dict mapping of equal weights. But builder expects either dict or
                # array (B,n), (n,), scalar, or (B,n,1). We cannot know n here; but the builder will be called
                # with names; to preserve semantics, we will pass back the raw tensor and adapt in set_keypoint_targets
                return {"raw": wt, "builder": wt}
            if wt.dim() == 2 and wt.shape[0] == batch and wt.shape[1] == 1:
                return {"raw": wt, "builder": wt}
            # Reject per-joint tensors here
            raise ValueError("Raw tensor weights/confidence must be shape (B,) or (B,1) or scalar")

        # numeric scalar
        return {"raw": float(w), "builder": float(w)}

    def _assemble_loss(
        self, output: UnifiedSmplOutput, inputs: UnifiedSmplInputs
    ) -> tuple[Tensor, dict[str, float]]:
        """Assemble the total loss from configured terms.

        Parameters
        ----------
        output : UnifiedSmplOutput
            Model output for the current parameters.
        inputs : UnifiedSmplInputs
            The live inputs passed to the forward call (used for priors).

        Returns
        -------
        tuple[Tensor, dict[str, float]]
            A tuple of the total loss tensor and a per‑term float breakdown.
        """
        total: Tensor | None = None
        terms_f: dict[str, float] = {}

        # Data term
        if self.m_keypts_data_term is not None:
            l_data = self.m_keypts_data_term(output)
            total = l_data if total is None else total + l_data
            try:
                terms_f["data"] = float(l_data.detach().item())
            except Exception:
                terms_f["data"] = float(l_data.item())

        # VPoser (built dynamically to reflect current NamedPose)
        if self.vposer_is_enabled() and inputs.named_pose is not None:
            assert self.m_vposer is not None and self.m_vposer_weights is not None
            vp_builder = VPoserPriorLossBuilder.from_vposer(self.model, self.m_vposer)
            pose_body = VPoserModel.convert_named_pose_to_pose_body(inputs.named_pose)
            vp_term = vp_builder.by_pose(pose_body, self.m_vposer_weights[0], self.m_vposer_weights[1])
            l_vposer = vp_term(output)
            total = l_vposer if total is None else total + l_vposer
            try:
                terms_f["vposer"] = float(l_vposer.detach().item())
            except Exception:
                terms_f["vposer"] = float(l_vposer.item())

        # Pose L2 over trainable pose parameters only
        if self.m_pose_l2_weight is not None:
            l2: Tensor = torch.zeros((), device=self.model.device, dtype=self.model.dtype)
            # NamedPose intrinsic pose
            if inputs.named_pose is not None and isinstance(inputs.named_pose.intrinsic_pose, torch.Tensor):
                p = inputs.named_pose.intrinsic_pose
                if p.requires_grad:
                    l2 = l2 + (p**2).sum()
            # global orient
            if inputs.global_orient is not None and isinstance(inputs.global_orient, torch.Tensor) and inputs.global_orient.requires_grad:
                l2 = l2 + (inputs.global_orient**2).sum()
            l_pose = self.m_pose_l2_weight * l2
            total = l_pose if total is None else total + l_pose
            try:
                terms_f["pose_l2"] = float(l_pose.detach().item())
            except Exception:
                terms_f["pose_l2"] = float(l_pose.item())

        # Shape L2 over betas
        if self.m_shape_l2_weight is not None and inputs.betas is not None and isinstance(inputs.betas, torch.Tensor):
            if inputs.betas.requires_grad:
                l_shape = self.m_shape_l2_weight * (inputs.betas**2).sum()
                total = l_shape if total is None else total + l_shape
                try:
                    terms_f["shape_l2"] = float(l_shape.detach().item())
                except Exception:
                    terms_f["shape_l2"] = float(l_shape.item())

        # Custom terms
        for name, (term, weight) in self.m_custom_terms.items():
            val = term(output)
            w = float(weight)
            l_custom = val * w
            total = l_custom if total is None else total + l_custom
            try:
                terms_f[name] = float(l_custom.detach().item())
            except Exception:
                terms_f[name] = float(l_custom.item())

        if total is None:
            total = torch.zeros((), device=self.model.device, dtype=self.model.dtype)
        return total, terms_f
