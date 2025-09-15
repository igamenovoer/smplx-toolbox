"""VPoser-based pose priors for SMPL body pose."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from smplx_toolbox.core import UnifiedSmplModel, UnifiedSmplOutput, NamedPose
from smplx_toolbox.vposer.model import VPoserModel

from .builders_base import BaseLossBuilder


class VPoserPriorLossBuilder(BaseLossBuilder):
    """Build VPoser-based priors for body pose.

    Requires a VPoser-like module providing ``decode(latent) -> Tensor|dict``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.m_vposer: VPoserModel | None = None

    @classmethod
    def from_vposer(
        cls, model: UnifiedSmplModel, vposer: VPoserModel
    ) -> VPoserPriorLossBuilder:
        instance = cls.from_model(model)
        instance.m_vposer = vposer
        return instance

    @property
    def vposer(self) -> VPoserModel:
        """Access the underlying VPoser model (read-only)."""
        if self.m_vposer is None:
            raise ValueError("VPoser not set. Use from_vposer(model, vposer)")
        return self.m_vposer

    def _decode_body_pose(self, latent: Tensor) -> Tensor:
        if self.m_vposer is None:
            raise ValueError("VPoser not set. Use from_vposer(model, vposer)")
        vposer = self.m_vposer
        assert vposer is not None
        out = vposer.decode(latent)
        # Our VPoserModel.decode returns a dict with 'pose_body' and 'pose_body_matrot'
        val = out.get("pose_body", None)
        if not isinstance(val, torch.Tensor):
            raise ValueError("VPoser.decode did not return 'pose_body' tensor")
        return val

    def encode_pose_to_latent(self, pose: Tensor) -> Tensor:
        """Encode a body AA pose to a latent (uses mean of the posterior).

        Accepts pose as (B, 63) or (B, 21, 3) axis-angle. Returns (B, Z).
        """
        vposer = self.vposer
        if pose.dim() == 3 and pose.shape[-2:] == (21, 3):
            pose_flat = pose.reshape(pose.shape[0], -1)
        elif pose.dim() == 2 and pose.shape[1] == 63:
            pose_flat = pose
        else:
            raise ValueError("pose must be (B,63) or (B,21,3) axis-angle body pose")
        # Move to VPoser's device/dtype before encoding
        vp_param = next(vposer.parameters(), None)
        if vp_param is not None:
            pose_flat = pose_flat.to(device=vp_param.device, dtype=vp_param.dtype)
        q_z = vposer.encode(pose_flat)
        z = cast(Tensor, q_z.mean if hasattr(q_z, "mean") else q_z)
        return z

    def decode_latent_to_pose(self, latent: Tensor) -> Tensor:
        """Decode a latent to body AA pose as (B, 21, 3)."""
        return self._decode_body_pose(latent)

    def by_pose_latent(self, latent: Tensor, weight: float | Tensor = 1.0) -> nn.Module:
        """Case 1: Given a latent ``z``, return weighted L2 prior on ``z``."""
        w = (
            weight
            if isinstance(weight, torch.Tensor)
            else torch.tensor(weight, device=self.device, dtype=self.dtype)
        )

        class _LatentL2(nn.Module):
            def __init__(self, latent_in: Tensor, weight_in: Tensor) -> None:
                super().__init__()
                self.latent: Tensor
                self.weight: Tensor
                self.register_buffer("latent", latent_in)
                self.register_buffer("weight", weight_in)

            def forward(self, _: UnifiedSmplOutput) -> Tensor:
                return self.weight * (self.latent**2).mean()

        return _LatentL2(latent.to(device=self.device, dtype=self.dtype), w)

    # Backward-compat alias
    def from_latent(
        self, latent: Tensor, weight: float | Tensor = 1.0
    ) -> nn.Module:  # pragma: no cover - compat
        return self.by_pose_latent(latent, weight)

    def by_pose(
        self, pose: Tensor, w_pose_fit: float | Tensor, w_latent_l2: float | Tensor
    ) -> nn.Module:
        """Case 2: Given a body pose in AA, encourage it to lie on the VPoser manifold.

        Computes: MSE(pose_in, pose_out) * w_pose_fit + L2(latent_in) * w_latent_l2
        where ``latent_in = VPoser.encode(pose_in).mean`` and ``pose_out = VPoser.decode(latent_in)['pose_body']``.
        Accepts pose shapes (B, 63) or (B, 21, 3) (body only).
        """
        if self.m_vposer is None:
            raise ValueError("VPoser not set. Use from_vposer(model, vposer)")

        w1 = (
            w_pose_fit
            if isinstance(w_pose_fit, torch.Tensor)
            else torch.tensor(w_pose_fit, device=self.device, dtype=self.dtype)
        )
        w2 = (
            w_latent_l2
            if isinstance(w_latent_l2, torch.Tensor)
            else torch.tensor(w_latent_l2, device=self.device, dtype=self.dtype)
        )

        pose_ref = pose

        class _PoseFit(nn.Module):
            def __init__(
                self,
                outer: VPoserPriorLossBuilder,
                pose_in: Tensor,
                w_pose: Tensor,
                w_latent: Tensor,
            ) -> None:
                super().__init__()
                self.m_outer = outer
                self.m_pose_in = pose_in
                self.w_pose: Tensor
                self.w_latent: Tensor
                self.register_buffer("w_pose", w_pose)
                self.register_buffer("w_latent", w_latent)

            def forward(self, _: UnifiedSmplOutput) -> Tensor:
                p = self.m_pose_in
                if p.dim() == 3 and p.shape[-2:] == (21, 3):
                    p_flat = p.reshape(p.shape[0], -1)
                elif p.dim() == 2 and p.shape[1] == 63:
                    p_flat = p
                else:
                    raise ValueError(
                        "pose must be (B,63) or (B,21,3) axis-angle body pose"
                    )

                vposer = self.m_outer.m_vposer
                assert vposer is not None
                # Move to the VPoser device/dtype
                vp_param = next(vposer.parameters(), None)
                if vp_param is not None:
                    p_flat = p_flat.to(device=vp_param.device, dtype=vp_param.dtype)
                q_z = vposer.encode(p_flat)
                z = cast(Tensor, q_z.mean if hasattr(q_z, "mean") else q_z)

                dec = self.m_outer._decode_body_pose(z)
                p_out = dec.reshape(dec.shape[0], -1)

                l_pose = F.mse_loss(p_flat, p_out, reduction="mean")
                l_latent = (z**2).mean()
                return self.w_pose * l_pose + self.w_latent * l_latent

        return _PoseFit(
            self,
            pose_ref,
            w1.to(self.device, self.dtype),
            w2.to(self.device, self.dtype),
        )

    def by_named_pose(
        self, npz: NamedPose, w_pose_fit: float | Tensor, w_latent_l2: float | Tensor
    ) -> nn.Module:
        """Convenience: Build prior from a NamedPose by extracting body pose.

        Uses our VPoserModel mapping utilities to convert the NamedPose to a
        63‑DoF body axis‑angle and then delegates to :meth:`by_pose`.
        """
        pose_body = VPoserModel.convert_named_pose_to_pose_body(npz)
        return self.by_pose(pose_body, w_pose_fit, w_latent_l2)
