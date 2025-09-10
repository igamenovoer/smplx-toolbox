"""Keypoint-based 3D matching losses for SMPL/SMPL-X outputs.

This module implements a builder for constructing differentiable data terms
that match model joints to target 3D keypoint positions. It operates either on
the unified 55-joint space (selection by joint names) or on the raw/native
joint space via a packed API.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from smplx_toolbox.core import UnifiedSmplModel, UnifiedSmplOutput

from .builders_base import BaseLossBuilder


class SmplLossTerm(nn.Module):
    """Base class for loss terms operating on ``UnifiedSmplOutput``.

    Subclasses must implement ``forward(output)`` and return a scalar loss by
    default. If constructed with ``reduction='none'`` the term may return a
    per-batch vector.
    """

    def __init__(self) -> None:
        super().__init__()


class KeypointMatchLossBuilder(BaseLossBuilder):
    """Build losses that match joints to 3D target positions.

    Supports two modes:
    - Name-based selection over the unified 55-joint space: convenient and model-agnostic.
    - Packed, native joint order: advanced users provide full-skeleton targets.
    """

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def from_model(cls, model: UnifiedSmplModel) -> KeypointMatchLossBuilder:
        return super().from_model(model)

    # --- Unified (by names)
    def by_target_positions(
        self,
        targets: dict[str, Tensor],
        *,
        weights: dict[str, float | Tensor] | float | Tensor | None = None,
        robust: Literal["l2", "huber", "gmof"] = "gmof",
        rho: float = 100.0,
        confidence: dict[str, float | Tensor] | float | Tensor | None = None,
        missing: Literal["ignore", "zero", "nan"] = "ignore",
        reduction: Literal["none", "mean", "sum"] | None = "mean",
    ) -> SmplLossTerm:
        """Create a 3D keypoint data term using joint names in unified space.

        Parameters
        ----------
        targets : dict[str, Tensor]
            Mapping from joint name to target 3D position with shape ``(B, 3)``.
        weights : dict[str, float|Tensor] | float | Tensor | None, optional
            Per-joint or scalar weights. If a dict, keys must match ``targets``.
        robust : {"l2", "huber", "gmof"}, optional
            Robust loss type. Defaults to ``"gmof"``.
        rho : float, optional
            Scale parameter for robustifiers. Defaults to ``100.0``.
        confidence : dict[str, float|Tensor] | float | Tensor | None, optional
            Additional confidence multipliers (same broadcasting rules as weights).
        missing : {"ignore", "zero", "nan"}, optional
            How to handle NaNs in targets: ignore (mask), zero (zero targets),
            or keep NaNs and mask them out. Defaults to ``"ignore"``.
        reduction : {"none", "mean", "sum"} | None, optional
            Reduction applied to the final loss. ``None`` is treated as
            ``"none"``.
        """
        if not targets:
            raise ValueError("targets must not be empty")

        # Maintain the insertion order of provided names
        names = list(targets.keys())
        # Stack targets to (B, n, 3)
        tgt_list = [self._ensure_target(targets[n]) for n in names]
        target_positions = torch.stack(tgt_list, dim=1)

        # Determine batch size and allocate weights/confidences
        batch = target_positions.shape[0]
        n = target_positions.shape[1]

        w = self._prepare_weights(batch, n, weights, names=names)
        c = (
            self._prepare_weights(batch, n, confidence, names=names)
            if confidence is not None
            else torch.ones_like(w)
        )
        w_total = w * c

        # Name indices for forward-time pred selection
        indices = self._names_to_indices(names)

        class _UnifiedNameTerm(SmplLossTerm):
            def __init__(
                self,
                outer: KeypointMatchLossBuilder,
                indices_in: list[int],
                target_in: Tensor,
                weights_in: Tensor,
                robust_in: Literal["l2", "huber", "gmof"],
                rho_in: float,
                missing_policy: str,
                reduction_in: Literal["none", "mean", "sum"] | None,
            ) -> None:
                super().__init__()
                self.m_outer = outer
                self.m_indices = torch.tensor(indices_in, dtype=torch.long)
                self.m_target = target_in  # (B, n, 3)
                self.m_weights = weights_in  # (B, n)
                self.m_robust = robust_in
                self.m_rho = float(rho_in)
                self.m_missing = missing_policy
                self.m_reduction = reduction_in

            def forward(self, output: UnifiedSmplOutput) -> Tensor:
                # Select predicted joints (B, n, 3)
                pred = output.joints[:, self.m_indices.to(device=output.joints.device)]

                # Move cached targets/weights to match output's device/dtype if needed
                if (
                    self.m_target.device != pred.device
                    or self.m_target.dtype != pred.dtype
                ):
                    tgt = self.m_target.to(device=pred.device, dtype=pred.dtype)
                else:
                    tgt = self.m_target
                if (
                    self.m_weights.device != pred.device
                    or self.m_weights.dtype != pred.dtype
                ):
                    w = self.m_weights.to(device=pred.device, dtype=pred.dtype)
                else:
                    w = self.m_weights

                # Missing policy via mask from target NaNs
                if torch.isnan(tgt).any():
                    mask = (~torch.isnan(tgt).any(dim=-1)).to(pred.dtype)  # (B, n)
                    if self.m_missing == "ignore" or self.m_missing == "nan":
                        w = w * mask
                    elif self.m_missing == "zero":
                        # Replace NaNs with zeros, but keep weights
                        tgt = torch.nan_to_num(tgt, nan=0.0)
                    else:
                        raise ValueError(f"Unknown missing policy: {self.m_missing}")
                # Residuals and robustification
                residual = pred - tgt  # (B, n, 3)
                penal = self.m_outer._robust(residual, self.m_robust, self.m_rho)
                # Sum over XYZ
                penal_xy = penal.sum(dim=-1)  # (B, n)
                # Apply weights
                penal_xy = penal_xy * w
                # Reduce
                return self.m_outer._reduce(penal_xy, self.m_reduction)

        return _UnifiedNameTerm(
            outer=self,
            indices_in=indices,
            target_in=target_positions,
            weights_in=w_total,
            robust_in=robust,
            rho_in=rho,
            missing_policy=missing,
            reduction_in=reduction,
        )

    # --- Packed (native order)
    def by_target_positions_packed(
        self,
        target_positions: Tensor,
        *,
        weights: Tensor | float | None = None,
        robust: Literal["l2", "huber", "gmof"] = "gmof",
        rho: float = 100.0,
        missing: Literal["ignore", "zero", "nan"] = "ignore",
        reduction: Literal["none", "mean", "sum"] | None = "mean",
    ) -> SmplLossTerm:
        """Advanced API: targets in the model's native joint order.

        Parameters
        ----------
        target_positions : torch.Tensor
            Target keypoints of shape ``(B, J_raw, 3)``.
        weights : torch.Tensor | float | None, optional
            Weights of shape ``(B, J_raw)``, ``(J_raw,)``, scalar, or
            ``(B, J_raw, 1)``.
        robust : {"l2", "huber", "gmof"}, optional
            Robust loss type. Defaults to ``"gmof"``.
        rho : float, optional
            Scale parameter for robustifiers. Defaults to ``100.0``.
        missing : {"ignore", "zero", "nan"}, optional
            How to handle NaNs in targets. See unified API.
        reduction : {"none", "mean", "sum"} | None, optional
            Reduction applied to the final loss.
        """
        tgt = self._ensure_target(target_positions)
        if tgt.dim() != 3 or tgt.shape[-1] != 3:
            raise ValueError("target_positions must have shape (B, J_raw, 3)")

        batch, j_raw, _ = tgt.shape

        # Validate J_raw against current model
        expected = {"smpl": 24, "smplh": 52, "smplx": 55}[str(self.model.model_type)]
        if j_raw != expected:
            raise ValueError(
                f"target_positions.shape[1] == {j_raw}, expected {expected}"
            )

        # Prepare weights
        if isinstance(weights, torch.Tensor):
            w = self._prepare_weights(batch, j_raw, weights)
        elif weights is None:
            w = torch.ones((batch, j_raw), device=self.device, dtype=self.dtype)
        else:
            w = self._prepare_weights(batch, j_raw, float(weights))

        class _PackedTerm(SmplLossTerm):
            def __init__(
                self,
                outer: KeypointMatchLossBuilder,
                target_in: Tensor,
                weights_in: Tensor,
                robust_in: Literal["l2", "huber", "gmof"],
                rho_in: float,
                missing_policy: str,
                reduction_in: Literal["none", "mean", "sum"] | None,
            ) -> None:
                super().__init__()
                self.m_outer = outer
                self.m_target = target_in  # (B, J_raw, 3)
                self.m_weights = weights_in  # (B, J_raw)
                self.m_robust = robust_in
                self.m_rho = float(rho_in)
                self.m_missing = missing_policy
                self.m_reduction = reduction_in

            def forward(self, output: UnifiedSmplOutput) -> Tensor:
                # Use raw joints from extras; fall back to unified if missing
                pred_raw = output.extras.get("joints_raw", None)
                pred = output.joints if pred_raw is None else pred_raw

                # Ensure device/dtype alignment
                if (
                    self.m_target.device != pred.device
                    or self.m_target.dtype != pred.dtype
                ):
                    tgt = self.m_target.to(device=pred.device, dtype=pred.dtype)
                else:
                    tgt = self.m_target
                if (
                    self.m_weights.device != pred.device
                    or self.m_weights.dtype != pred.dtype
                ):
                    w = self.m_weights.to(device=pred.device, dtype=pred.dtype)
                else:
                    w = self.m_weights

                if torch.isnan(tgt).any():
                    mask = (~torch.isnan(tgt).any(dim=-1)).to(pred.dtype)
                    if self.m_missing == "ignore" or self.m_missing == "nan":
                        w = w * mask
                    elif self.m_missing == "zero":
                        tgt = torch.nan_to_num(tgt, nan=0.0)
                    else:
                        raise ValueError(f"Unknown missing policy: {self.m_missing}")

                residual = pred - tgt  # (B, J_raw, 3)
                penal = self.m_outer._robust(residual, self.m_robust, self.m_rho)
                penal_xy = penal.sum(dim=-1) * w  # (B, J_raw)
                return self.m_outer._reduce(penal_xy, self.m_reduction)

        return _PackedTerm(
            outer=self,
            target_in=tgt,
            weights_in=w,
            robust_in=robust,
            rho_in=rho,
            missing_policy=missing,
            reduction_in=reduction,
        )
