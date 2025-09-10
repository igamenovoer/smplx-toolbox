"""Base utilities for building SMPL/SMPL-X optimization loss terms.

This module defines a common base class for loss builders that operate on the
unified SMPL family model interface. Builders should accept a configured
``UnifiedSmplModel`` via a factory method and expose helpers for name→index
resolution, device/dtype synchronization, weight broadcasting, and robust
penalties.

Notes
-----
- Follow the project coding guidelines: member variables are prefixed with
  ``m_`` and constructors take no arguments; use factory methods for
  initialization.
- All tensors are expected to be batch-first.
"""

from __future__ import annotations

from typing import Literal, TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor

from smplx_toolbox.core import UnifiedSmplModel

from .robustifiers import gmof

T_Base = TypeVar("T_Base", bound="BaseLossBuilder")


class BaseLossBuilder:
    """Shared helpers for optimization loss builders.

    The base builder stores a reference to the configured unified model and
    provides utilities commonly needed across loss builders.

    Notes
    -----
    Use :meth:`from_model` to create instances.
    """

    def __init__(self) -> None:
        self.m_model: UnifiedSmplModel | None = None

    @classmethod
    def from_model(cls: type[T_Base], model: UnifiedSmplModel) -> T_Base:
        """Create a builder bound to a ``UnifiedSmplModel``.

        Parameters
        ----------
        model : UnifiedSmplModel
            The unified SMPL family model adapter.

        Returns
        -------
        BaseLossBuilder
            A configured builder instance.
        """
        instance = cls()
        instance.m_model = model
        return instance

    # --- Read-only convenience properties
    @property
    def model(self) -> UnifiedSmplModel:
        """Access the bound model (read-only)."""
        if self.m_model is None:
            raise ValueError("Builder is not initialized. Use from_model().")
        return self.m_model

    @property
    def device(self) -> torch.device:
        """Return the model's device for tensor placement."""
        return self.model.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the model's dtype for tensor casting."""
        return self.model.dtype

    # --- Core helpers
    def _names_to_indices(self, names: list[str]) -> list[int]:
        """Map unified joint names to indices.

        Unknown names are ignored with a warning via the model's warn function.
        """
        joint_names = self.model.get_joint_names(unified=True)
        indices: list[int] = []
        for n in names:
            if n in joint_names:
                indices.append(joint_names.index(n))
            else:
                # Defer to model's warn handler
                warn_fn = getattr(self.model, "m_warn_fn", None)
                if callable(warn_fn):
                    warn_fn(f"Joint name '{n}' not found; skipping.")
        return indices

    def _ensure_tensor(self, x: Tensor | float | int) -> Tensor:
        """Convert scalars to tensors and move to model's device/dtype."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        return x.to(device=self.device, dtype=self.dtype)

    def _ensure_target(self, tgt: Tensor) -> Tensor:
        """Move a target tensor to the model's device/dtype."""
        return tgt.to(device=self.device, dtype=self.dtype)

    def _prepare_weights(
        self,
        batch: int,
        n: int,
        weights: float | Tensor | dict[str, float | Tensor] | None,
        names: list[str] | None = None,
    ) -> Tensor:
        """Broadcast weights to shape ``(B, n)``.

        Accepts scalars, per-joint tensors, or dictionaries keyed by joint name.
        Returns a tensor on the model's device/dtype.
        """
        if weights is None:
            w = torch.ones((batch, n), device=self.device, dtype=self.dtype)
            return w

        if isinstance(weights, dict):
            if names is None:
                raise ValueError("names must be provided when weights is a dict")
            per_joint = torch.ones((n,), device=self.device, dtype=self.dtype)
            for i, name in enumerate(names):
                if name in weights:
                    per_joint[i] = self._ensure_tensor(weights[name]).to(self.dtype)
            w = per_joint.expand(batch, -1).clone()
            return w

        if isinstance(weights, torch.Tensor):
            w = self._ensure_target(weights)
            # Accept shapes: (), (n,), (B,n), (B,n,1)
            if w.dim() == 0:
                w = w.view(1, 1).expand(batch, n)
            elif w.dim() == 1:
                if w.numel() != n:
                    raise ValueError(f"weights has {w.numel()} items; expected {n}")
                w = w.view(1, n).expand(batch, -1)
            elif w.dim() == 2:
                if w.shape != (batch, n):
                    raise ValueError(f"weights shape {w.shape} != ({batch},{n})")
            elif w.dim() == 3 and w.shape[2] == 1:
                if w.shape[:2] != (batch, n):
                    raise ValueError(f"weights shape {w.shape} != ({batch},{n},1)")
                w = w.squeeze(-1)
            else:
                raise ValueError("Unsupported weights shape")
            return w

        # Scalar numeric
        w = self._ensure_tensor(weights).view(1, 1).expand(batch, n)
        return w

    def _robust(
        self, residual: Tensor, kind: Literal["l2", "huber", "gmof"], rho: float
    ) -> Tensor:
        """Apply a robust penalty elementwise to ``residual``.

        Parameters
        ----------
        residual : torch.Tensor
            Elementwise residuals of arbitrary shape.
        kind : {"l2", "huber", "gmof"}
            Robust penalty type.
        rho : float
            Scale parameter for ``huber``/``gmof``.
        """
        if kind == "l2":
            return residual**2
        if kind == "huber":
            zeros = torch.zeros_like(residual)
            return F.huber_loss(residual, zeros, delta=float(rho), reduction="none")
        if kind == "gmof":
            return gmof(residual, rho=float(rho))
        raise ValueError(f"Unknown robust kind: {kind}")

    def _reduce(
        self,
        values: Tensor,
        reduction: Literal["none", "mean", "sum"] | None,
        *,
        per_batch_sum: bool = True,
    ) -> Tensor:
        """Reduce a tensor according to ``reduction``.

        If ``per_batch_sum`` is True and ``reduction`` is ``None`` or ``"none"``,
        the method returns a per-batch vector of shape ``(B,)`` (summing over all
        remaining dimensions). Otherwise it returns a scalar.
        """
        if reduction is None or reduction == "none":
            if not per_batch_sum:
                return values
            # Sum all but batch dim
            while values.dim() > 1:
                values = values.sum(dim=-1)
            return values  # (B,)
        if reduction == "mean":
            return values.mean()
        if reduction == "sum":
            return values.sum()
        raise ValueError(f"Unknown reduction: {reduction}")
