"""Minimal VPoser runtime for SMPL family workflows.

This module provides a small, dependency‑free subset of the original
``human_body_prior`` VPoser implementation that is sufficient for:

- Loading the provided VPoser v2 checkpoint
- Encoding a 21‑joint body pose (axis‑angle) into a latent Normal
- Decoding a latent vector back to a 21‑joint body pose

It mirrors the v2 architecture so that the checkpoint weights can be loaded
without requiring the upstream package to be installed.

Notes
-----
- The model operates on body pose only (no root, no hands/face): 21 joints,
  3 DoF per joint (axis‑angle), i.e. 63 DoF in total.
- The returned dictionary from :meth:`VPoserModel.decode` includes both
  axis‑angle and rotation‑matrix representations.

Input/Output Mapping for Body Pose (21 joints)
----------------------------------------------
VPoser expects the body joint axis‑angle in a fixed 21‑joint order (pelvis/root
excluded). We adopt the exact body ordering already used throughout this
toolbox to construct 63‑DoF body poses (see
canonical 21‑joint body ordering used across the toolbox:

    [left_hip, right_hip, spine1, left_knee, right_knee, spine2,
     left_ankle, right_ankle, spine3, left_foot, right_foot,
     neck, left_collar, right_collar, head,
     left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist]

Why this is correct and how to verify:
- Upstream VPoser (context/refcode/human_body_prior/models/vposer_model.py) sets
  `self.num_joints = 21` and `decode()` returns a dict with `pose_body` of shape
  `(B, 21, 3)`. That confirms the 21‑joint body target in axis‑angle.
- Our unified model/containers already define the canonical body AA order used
  to create a `(B, 63)` vector (see the body_joints list used to assemble
  `pose_body` in `UnifiedSmplInputs.from_keypoint_pose`). Aligning VPoser’s
  input ordering with this list ensures consistency across the toolbox.
- Practically: encode any `(B, 63)` body AA built with that order, decode back,
  and verify joints align joint‑wise (except small reconstruction error from the
  VAE). See the helpers below.

Use :meth:`convert_named_pose_to_pose_body` to build a `(B, 63)` tensor from a
`NamedPose`, and :meth:`convert_pose_body_to_named_pose` to go back. This helps
inspect exactly what VPoser consumes/produces without memorizing indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

from smplx_toolbox.core.constants import CoreBodyJoint, ModelType
from smplx_toolbox.core.containers import NamedPose


class BatchFlatten(nn.Module):
    """Flatten all non‑batch dimensions.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape ``(B, ...)``.

    Returns
    -------
    torch.Tensor
        Flattened tensor of shape ``(B, -1)``.
    """

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        b = x.shape[0]
        return x.view(b, -1)


class ContinousRotReprDecoder(nn.Module):
    """Decode a 6D rotation representation to 3x3 rotation matrices.

    This follows the approach from Zhou et al.,
    "On the Continuity of Rotation Representations in Neural Networks".

    Parameters
    ----------
    x : torch.Tensor
        A tensor of shape ``(N, 6)`` representing rotations.

    Returns
    -------
    torch.Tensor
        Rotation matrices of shape ``(N, 3, 3)``.
    """

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        x = x.view(-1, 3, 2)
        b1 = F.normalize(x[:, :, 0], dim=1)
        dot = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(x[:, :, 1] - dot * b1, dim=1)
        b3 = torch.cross(b1, b2, dim=1)
        R = torch.stack([b1, b2, b3], dim=-1)  # (N, 3, 3)
        return R


class NormalDistDecoder(nn.Module):
    """Map features to a diagonal Gaussian (Normal) distribution.

    The module learns the mean and (softplus‑activated) standard deviation in
    latent space from an input feature vector.

    Parameters
    ----------
    num_feat_in : int
        Number of input features.
    latent_dim : int
        Size of the latent vector.
    """

    def __init__(self, num_feat_in: int, latent_dim: int) -> None:
        super().__init__()
        self.mu = nn.Linear(num_feat_in, latent_dim)
        self.logvar = nn.Linear(num_feat_in, latent_dim)

    def forward(self, x: Tensor) -> Normal:
        """Compute the output :class:`torch.distributions.Normal`.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape ``(B, num_feat_in)``.

        Returns
        -------
        torch.distributions.Normal
            A diagonal Normal distribution over the latent space
            with ``loc = mu(x)``, ``scale = softplus(logvar(x))``.
        """
        return Normal(self.mu(x), F.softplus(self.logvar(x)))


def _matrot_to_axis_angle(R: Tensor, eps: float = 1e-6) -> Tensor:
    """Convert rotation matrices to axis‑angle vectors.

    Parameters
    ----------
    R : torch.Tensor
        Rotation matrices of shape ``(N, 3, 3)``.
    eps : float, optional
        Small clamping term to keep values inside valid ranges; default ``1e-6``.

    Returns
    -------
    torch.Tensor
        Axis‑angle vectors of shape ``(N, 3)``.
    """
    # Clamp trace to valid range for arccos
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    theta = torch.arccos(cos_theta)

    # Compute axis from skew-symmetric part
    rx = R[:, 2, 1] - R[:, 1, 2]
    ry = R[:, 0, 2] - R[:, 2, 0]
    rz = R[:, 1, 0] - R[:, 0, 1]
    axis = torch.stack([rx, ry, rz], dim=1)

    # For small angles, fall back to first-order approximation
    sin_theta = torch.sin(theta)
    small = sin_theta.abs() < 1e-4
    # General case
    k = torch.where(small, torch.ones_like(sin_theta), 1.0 / (2.0 * sin_theta))
    axis = axis * k.unsqueeze(1)
    aa = axis * theta.unsqueeze(1)
    # For near-zero angles, use logarithm map approximation: aa ~ vee(R - I)
    if small.any():
        RmI = R[small] - torch.eye(3, device=R.device, dtype=R.dtype)
        rx_s = RmI[:, 2, 1] - RmI[:, 1, 2]
        ry_s = RmI[:, 0, 2] - RmI[:, 2, 0]
        rz_s = RmI[:, 1, 0] - RmI[:, 0, 1]
        aa_small = torch.stack([rx_s, ry_s, rz_s], dim=1) * 0.5
        aa = aa.clone()
        aa[small] = aa_small
    return aa


@dataclass
class VPoserConfig:
    num_neurons: int
    latent_dim: int
    num_joints: int = 21


class VPoserModel(nn.Module):
    """VPoser‑compatible body pose prior (v2 architecture).

    The model receives 21 body joint rotations in axis‑angle (63 DoF) and
    provides a low‑dimensional latent representation along with a decoder back
    to the same pose space.

    Parameters
    ----------
    cfg : VPoserConfig
        Architecture hyper‑parameters (neurons, latent size, joint count).
    """

    def __init__(self, cfg: VPoserConfig) -> None:
        super().__init__()
        nfeat = cfg.num_joints * 3
        self.m_cfg = cfg

        self.encoder_backbone: nn.Sequential = nn.Sequential(
            BatchFlatten(),
            nn.BatchNorm1d(nfeat),
            nn.Linear(nfeat, cfg.num_neurons),
            nn.LeakyReLU(),
            nn.BatchNorm1d(cfg.num_neurons),
            nn.Dropout(0.1),
            nn.Linear(cfg.num_neurons, cfg.num_neurons),
            nn.Linear(cfg.num_neurons, cfg.num_neurons),
        )
        self.encoder_head: NormalDistDecoder = NormalDistDecoder(
            cfg.num_neurons, cfg.latent_dim
        )

        self.decoder_net: nn.Sequential = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.num_neurons, cfg.num_neurons),
            nn.LeakyReLU(),
            nn.Linear(cfg.num_neurons, cfg.num_joints * 6),
            ContinousRotReprDecoder(),
        )

    @property
    def latent_dim(self) -> int:
        """Latent space dimensionality (D)."""
        return self.m_cfg.latent_dim

    @property
    def num_joints(self) -> int:
        """Number of body joints used by VPoser (default 21)."""
        return self.m_cfg.num_joints

    def encode(self, pose_body: Tensor) -> Normal:
        """Encode body pose to a latent Normal distribution.

        Parameters
        ----------
        pose_body : torch.Tensor
            Body pose in axis‑angle. Shape ``(B, 21, 3)`` or flattened
            ``(B, 63)``.

        Returns
        -------
        torch.distributions.Normal
            A diagonal Normal distribution in the latent space.
        """
        if pose_body.dim() == 3:
            b = pose_body.shape[0]
            pose_body = pose_body.reshape(b, -1)
        feats = self.encoder_backbone(pose_body)
        return self.encoder_head.forward(feats)

    def decode(self, z: Tensor) -> dict[str, Tensor]:
        """Decode latent vectors to body pose.

        Parameters
        ----------
        z : torch.Tensor
            Latent vectors of shape ``(B, D)``.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary with keys:
            - ``pose_body``: axis‑angle pose of shape ``(B, 21, 3)``
            - ``pose_body_matrot``: rotation matrices flattened as
              ``(B, 21, 9)``
        """
        bs = z.shape[0]
        mats = self.decoder_net(z)  # (B*21, 3, 3)
        # The original returns flat batch of all joints. Reconstruct per-batch.
        # Our decoder receives (B, latent), final Linear outputs (B, J*6);
        # ContinousRotReprDecoder will reshape internally to (B*J,3,2)->(B*J,3,3).
        mats = mats.view(bs, self.num_joints, 3, 3)
        aa = _matrot_to_axis_angle(mats.view(-1, 3, 3)).view(bs, self.num_joints, 3)
        return {
            "pose_body": aa,
            "pose_body_matrot": mats.reshape(bs, self.num_joints, 9),
        }

    def forward(self, pose_body: Tensor) -> dict[str, Any]:
        """Encode then decode a body pose in a single call.

        Parameters
        ----------
        pose_body : torch.Tensor
            Body pose in AA of shape ``(B, 21, 3)`` or ``(B, 63)``.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the decoded pose and VAE outputs:
            - ``pose_body`` / ``pose_body_matrot`` from :meth:`decode`
            - ``poZ_body_mean`` / ``poZ_body_std`` (latent statistics)
            - ``q_z`` (:class:`torch.distributions.Normal`) the latent dist
        """
        q_z = self.encode(pose_body)
        z = q_z.rsample()
        out = self.decode(z)
        extras: dict[str, Any] = {
            "poZ_body_mean": q_z.mean,
            "poZ_body_std": q_z.scale,
            "q_z": q_z,
        }
        out.update(extras)
        return out

    # ------------------------------------------------------------------
    # Convenience utilities for NamedPose interop
    # ------------------------------------------------------------------
    @staticmethod
    def convert_named_pose_to_pose_body(npz: NamedPose) -> Tensor:
        """Build a (B, 63) body axis‑angle from ``NamedPose``.

        Parameters
        ----------
        npz : NamedPose
            Packed pose accessor; only body joints are used.

        Returns
        -------
        torch.Tensor
            Body pose axis‑angle of shape ``(B, 63)`` suitable for :meth:`encode`.
        """
        body_joints = [
            CoreBodyJoint.LEFT_HIP,
            CoreBodyJoint.RIGHT_HIP,
            CoreBodyJoint.SPINE1,
            CoreBodyJoint.LEFT_KNEE,
            CoreBodyJoint.RIGHT_KNEE,
            CoreBodyJoint.SPINE2,
            CoreBodyJoint.LEFT_ANKLE,
            CoreBodyJoint.RIGHT_ANKLE,
            CoreBodyJoint.SPINE3,
            CoreBodyJoint.LEFT_FOOT,
            CoreBodyJoint.RIGHT_FOOT,
            CoreBodyJoint.NECK,
            CoreBodyJoint.LEFT_COLLAR,
            CoreBodyJoint.RIGHT_COLLAR,
            CoreBodyJoint.HEAD,
            CoreBodyJoint.LEFT_SHOULDER,
            CoreBodyJoint.RIGHT_SHOULDER,
            CoreBodyJoint.LEFT_ELBOW,
            CoreBodyJoint.RIGHT_ELBOW,
            CoreBodyJoint.LEFT_WRIST,
            CoreBodyJoint.RIGHT_WRIST,
        ]
        # Extract from named pose; if any name missing, use zeros
        B = npz.packed_pose.shape[0]
        device = npz.packed_pose.device
        dtype = npz.packed_pose.dtype
        parts: list[Tensor] = []
        for e in body_joints:
            g = npz.get_joint_pose(e.value)
            if g is None:
                parts.append(torch.zeros((B, 1, 3), device=device, dtype=dtype))
            else:
                parts.append(g)
        body = torch.cat(parts, dim=1)  # (B,21,3)
        return body.reshape(B, 63)

    @staticmethod
    def convert_pose_body_to_named_pose(pose_body: Tensor) -> NamedPose:
        """Convert a body AA pose to a ``NamedPose`` instance (SMPL namespace).

        Parameters
        ----------
        pose_body : torch.Tensor
            Body pose as axis‑angle with shape ``(B, 63)`` or ``(B, 21, 3)``.

        Returns
        -------
        NamedPose
            A new instance with body joint fields populated in SMPL namespace;
            root (pelvis) remains zero.
        """
        if pose_body.dim() == 2 and pose_body.shape[1] == 63:
            B = pose_body.shape[0]
            body = pose_body.view(B, 21, 3)
        elif pose_body.dim() == 3 and pose_body.shape[1:] == (21, 3):
            body = pose_body
            B = body.shape[0]
        else:
            raise ValueError("pose_body must be (B,63) or (B,21,3)")
        names = [
            CoreBodyJoint.LEFT_HIP.value,
            CoreBodyJoint.RIGHT_HIP.value,
            CoreBodyJoint.SPINE1.value,
            CoreBodyJoint.LEFT_KNEE.value,
            CoreBodyJoint.RIGHT_KNEE.value,
            CoreBodyJoint.SPINE2.value,
            CoreBodyJoint.LEFT_ANKLE.value,
            CoreBodyJoint.RIGHT_ANKLE.value,
            CoreBodyJoint.SPINE3.value,
            CoreBodyJoint.LEFT_FOOT.value,
            CoreBodyJoint.RIGHT_FOOT.value,
            CoreBodyJoint.NECK.value,
            CoreBodyJoint.LEFT_COLLAR.value,
            CoreBodyJoint.RIGHT_COLLAR.value,
            CoreBodyJoint.HEAD.value,
            CoreBodyJoint.LEFT_SHOULDER.value,
            CoreBodyJoint.RIGHT_SHOULDER.value,
            CoreBodyJoint.LEFT_ELBOW.value,
            CoreBodyJoint.RIGHT_ELBOW.value,
            CoreBodyJoint.LEFT_WRIST.value,
            CoreBodyJoint.RIGHT_WRIST.value,
        ]
        npz = NamedPose(model_type=ModelType.SMPL, batch_size=B)
        for i, name in enumerate(names):
            npz.set_joint_pose_value(name, body[:, i])
        return npz
