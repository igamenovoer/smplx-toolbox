"""Shape prior losses for SMPL/SMPL-X betas."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from smplx_toolbox.core import UnifiedSmplModel, UnifiedSmplOutput

from .builders_base import BaseLossBuilder


class ShapePriorLossBuilder(BaseLossBuilder):
    """Build shape parameter priors (e.g., L2 on betas)."""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def from_model(cls, model: UnifiedSmplModel) -> ShapePriorLossBuilder:
        return super().from_model(model)

    def l2_on_betas(self, weight: float | Tensor = 1.0) -> nn.Module:
        """Penalize squared magnitude of shape parameters ``betas``.

        Notes
        -----
        The loss reads ``betas`` from ``output.extras['betas']``. Ensure the
        forward pass propagated betas through the unified model.
        """
        w = weight if isinstance(weight, torch.Tensor) else torch.tensor(weight, device=self.device, dtype=self.dtype)

        class _BetasL2(nn.Module):
            def __init__(self, outer: ShapePriorLossBuilder, weight_in: Tensor) -> None:
                super().__init__()
                self.m_outer = outer
                self.m_weight: Tensor = weight_in

            def forward(self, output: UnifiedSmplOutput) -> Tensor:
                betas = output.extras.get("betas", None)
                if betas is None:
                    raise ValueError("UnifiedSmplOutput.extras['betas'] not found; cannot apply shape prior")
                assert isinstance(betas, torch.Tensor)
                b = betas.to(device=self.m_weight.device, dtype=self.m_weight.dtype)
                return self.m_weight * (b ** 2).mean()

        return _BetasL2(self, w)
