"""2D keypoint matching losses via camera projection.

This module implements a builder for constructing differentiable 2D keypoint
data terms by projecting 3D joints to the image plane using a user-provided
camera. The camera object must expose a ``project(points: Tensor) -> Tensor``
method that accepts points of shape ``(B, N, 3)`` and returns ``(B, N, 2)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, cast, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor

from smplx_toolbox.core import UnifiedSmplModel, UnifiedSmplOutput

from .builders_base import BaseLossBuilder


@runtime_checkable
class _Projects2D(Protocol):
    def project(self, points: Tensor) -> Tensor:  # (B, N, 3) -> (B, N, 2)
        ...


class ProjectedKeypointMatchLossBuilder(BaseLossBuilder):
    """Build 2D keypoint data terms using a provided camera projector."""

    def __init__(self) -> None:
        super().__init__()
        self.m_camera: _Projects2D | Callable[[Tensor], Tensor] | None = None

    @classmethod
    def from_camera(
        cls, model: UnifiedSmplModel, camera: _Projects2D | Callable[[Tensor], Tensor]
    ) -> ProjectedKeypointMatchLossBuilder:
        instance = cls.from_model(model)
        instance.m_camera = camera
        return instance

    def _project(self, points3d: Tensor) -> Tensor:
        if self.m_camera is None:
            raise ValueError("Camera not set. Use from_camera(model, camera)")
        cam = self.m_camera
        if callable(cam) and not hasattr(cam, "project"):
            return cam(points3d)
        return cast(_Projects2D, cam).project(points3d)

    def by_target_positions(
        self,
        targets: dict[str, Tensor],
        *,
        weights: dict[str, float | Tensor] | float | Tensor | None = None,
        robust: Literal["l2", "huber", "gmof"] = "gmof",
        rho: float = 10.0,
        confidence: dict[str, float | Tensor] | float | Tensor | None = None,
        missing: Literal["ignore", "zero", "nan"] = "ignore",
        reduction: Literal["none", "mean", "sum"] | None = "mean",
    ) -> nn.Module:
        """Create a 2D keypoint data term using joint names in unified space."""
        if not targets:
            raise ValueError("targets must not be empty")

        names = list(targets.keys())
        tgt_list = [self._ensure_target(targets[n]) for n in names]
        target_positions = torch.stack(tgt_list, dim=1)  # (B, n, 2)

        batch, n, two = target_positions.shape
        if two != 2:
            raise ValueError("2D targets must have shape (B, n, 2)")

        w = self._prepare_weights(batch, n, weights, names=names)
        c = (
            self._prepare_weights(batch, n, confidence, names=names)
            if confidence is not None
            else torch.ones_like(w)
        )
        w_total = w * c
        indices = self._names_to_indices(names)

        class _Unified2DTerm(nn.Module):
            def __init__(
                self,
                outer: ProjectedKeypointMatchLossBuilder,
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
                self.m_target = target_in
                self.m_weights = weights_in
                self.m_robust = robust_in
                self.m_rho = float(rho_in)
                self.m_missing = missing_policy
                self.m_reduction = reduction_in

            def forward(self, output: UnifiedSmplOutput) -> Tensor:
                joints = output.joints[:, self.m_indices.to(device=output.joints.device)]
                proj = self.m_outer._project(joints)  # (B, n, 2)

                tgt = self.m_target.to(device=proj.device, dtype=proj.dtype)
                w = self.m_weights.to(device=proj.device, dtype=proj.dtype)

                if torch.isnan(tgt).any():
                    mask = (~torch.isnan(tgt).any(dim=-1)).to(proj.dtype)
                    if self.m_missing in ("ignore", "nan"):
                        w = w * mask
                    elif self.m_missing == "zero":
                        tgt = torch.nan_to_num(tgt, nan=0.0)
                    else:
                        raise ValueError(f"Unknown missing policy: {self.m_missing}")

                residual = proj - tgt  # (B, n, 2)
                penal = self.m_outer._robust(residual, self.m_robust, self.m_rho)
                penal_xy = penal.sum(dim=-1) * w  # (B, n)
                return self.m_outer._reduce(penal_xy, self.m_reduction)

        return _Unified2DTerm(
            outer=self,
            indices_in=indices,
            target_in=target_positions,
            weights_in=w_total,
            robust_in=robust,
            rho_in=rho,
            missing_policy=missing,
            reduction_in=reduction,
        )

    def by_target_positions_packed(
        self,
        target_positions: Tensor,
        *,
        weights: Tensor | float | None = None,
        robust: Literal["l2", "huber", "gmof"] = "gmof",
        rho: float = 10.0,
        missing: Literal["ignore", "zero", "nan"] = "ignore",
        reduction: Literal["none", "mean", "sum"] | None = "mean",
    ) -> nn.Module:
        """Advanced API: packed targets in native joint order (2D)."""
        tgt = self._ensure_target(target_positions)
        if tgt.dim() != 3 or tgt.shape[-1] != 2:
            raise ValueError("target_positions must have shape (B, J_raw, 2)")

        batch, j_raw, _ = tgt.shape
        expected = {"smpl": 24, "smplh": 52, "smplx": 55}[str(self.model.model_type)]
        if j_raw != expected:
            raise ValueError(f"target_positions.shape[1] == {j_raw}, expected {expected}")

        if isinstance(weights, torch.Tensor):
            w = self._prepare_weights(batch, j_raw, weights)
        elif weights is None:
            w = torch.ones((batch, j_raw), device=self.device, dtype=self.dtype)
        else:
            w = self._prepare_weights(batch, j_raw, float(weights))

        class _Packed2DTerm(nn.Module):
            def __init__(
                self,
                outer: ProjectedKeypointMatchLossBuilder,
                target_in: Tensor,
                weights_in: Tensor,
                robust_in: Literal["l2", "huber", "gmof"],
                rho_in: float,
                missing_policy: str,
                reduction_in: Literal["none", "mean", "sum"] | None,
            ) -> None:
                super().__init__()
                self.m_outer = outer
                self.m_target = target_in
                self.m_weights = weights_in
                self.m_robust = robust_in
                self.m_rho = float(rho_in)
                self.m_missing = missing_policy
                self.m_reduction = reduction_in

            def forward(self, output: UnifiedSmplOutput) -> Tensor:
                pred_raw = output.extras.get("joints_raw", None)
                points3d = output.joints if pred_raw is None else pred_raw
                proj = self.m_outer._project(points3d)

                tgt = self.m_target.to(device=proj.device, dtype=proj.dtype)
                w = self.m_weights.to(device=proj.device, dtype=proj.dtype)

                if torch.isnan(tgt).any():
                    mask = (~torch.isnan(tgt).any(dim=-1)).to(proj.dtype)
                    if self.m_missing in ("ignore", "nan"):
                        w = w * mask
                    elif self.m_missing == "zero":
                        tgt = torch.nan_to_num(tgt, nan=0.0)
                    else:
                        raise ValueError(f"Unknown missing policy: {self.m_missing}")

                residual = proj - tgt  # (B, J_raw, 2)
                penal = self.m_outer._robust(residual, self.m_robust, self.m_rho)
                penal_xy = penal.sum(dim=-1) * w  # (B, J_raw)
                return self.m_outer._reduce(penal_xy, self.m_reduction)

        return _Packed2DTerm(
            outer=self,
            target_in=tgt,
            weights_in=w,
            robust_in=robust,
            rho_in=rho,
            missing_policy=missing,
            reduction_in=reduction,
        )
