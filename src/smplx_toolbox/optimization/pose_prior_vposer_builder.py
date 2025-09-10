"""VPoser-based pose priors for SMPL body pose."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from smplx_toolbox.core import UnifiedSmplModel, UnifiedSmplOutput

from .builders_base import BaseLossBuilder


class VPoserPriorLossBuilder(BaseLossBuilder):
    """Build VPoser-based priors for body pose.

    Requires a VPoser-like module providing ``decode(latent) -> Tensor|dict``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.m_vposer: Any | None = None

    @classmethod
    def from_vposer(cls, model: UnifiedSmplModel, vposer: nn.Module) -> VPoserPriorLossBuilder:
        instance = cls.from_model(model)
        instance.m_vposer = vposer
        return instance

    def _decode_body_pose(self, latent: Tensor) -> Tensor:
        if self.m_vposer is None:
            raise ValueError("VPoser not set. Use from_vposer(model, vposer)")
        vposer = self.m_vposer
        assert vposer is not None
        out = vposer.decode(latent)
        if isinstance(out, dict):
            # Common field names
            for key in ("pose_body", "pose_body_tensor", "pose_body_decoded"):
                if key in out:
                    val = out[key]
                    assert isinstance(val, torch.Tensor)
                    return val
            # Otherwise fallback: first tensor value
            for v in out.values():
                if isinstance(v, torch.Tensor):
                    return v
            raise ValueError("VPoser.decode returned dict without tensor values")
        assert isinstance(out, torch.Tensor)
        return out

    def from_latent(self, latent: Tensor, weight: float | Tensor = 1.0) -> nn.Module:
        """Simple L2 prior on VPoser latent."""
        w = weight if isinstance(weight, torch.Tensor) else torch.tensor(weight, device=self.device, dtype=self.dtype)

        class _LatentL2(nn.Module):
            def __init__(self, latent_in: Tensor, weight_in: Tensor) -> None:
                super().__init__()
                self.latent: Tensor
                self.weight: Tensor
                self.register_buffer("latent", latent_in)
                self.register_buffer("weight", weight_in)

            def forward(self, _: UnifiedSmplOutput) -> Tensor:
                return self.weight * (self.latent ** 2).mean()

        return _LatentL2(latent.to(device=self.device, dtype=self.dtype), w)

    def decoded_body_pose(self, weight: float | Tensor = 1.0) -> nn.Module:
        """L2 prior on decoded body pose angles.

        This targets zero-mean body pose in axis-angle space if no likelihood
        evaluation API is available from the provided VPoser.
        """
        w = weight if isinstance(weight, torch.Tensor) else torch.tensor(weight, device=self.device, dtype=self.dtype)

        class _DecodedBodyPoseL2(nn.Module):
            def __init__(self, outer: VPoserPriorLossBuilder, weight_in: Tensor) -> None:
                super().__init__()
                self.m_outer = outer
                self.weight: Tensor
                self.register_buffer("weight", weight_in)

            def forward(self, _: UnifiedSmplOutput) -> Tensor:
                # Expect that the user manages latent externally; this term
                # assumes decoded pose is available via outer.decode(latent)
                raise RuntimeError(
                    "decoded_body_pose() requires a latent tensor at build time; use from_latent()."
                )

        # For now we require from_latent() usage; could extend later to accept a latent here.
        return _DecodedBodyPoseL2(self, w)
