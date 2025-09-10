"""Heuristic angle priors (knees/elbows) akin to SMPLify-X."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from smplx_toolbox.core import UnifiedSmplModel, UnifiedSmplOutput

from .builders_base import BaseLossBuilder


class AnglePriorLossBuilder(BaseLossBuilder):
    """Build simple bending priors on body joints.

    This implementation uses a heuristic on axis-angle components of specific
    body joints to discourage unnatural hyperextension. It is not a faithful
    reproduction of SMPLify-X but serves as a lightweight alternative.
    """

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def from_model(cls, model: UnifiedSmplModel) -> AnglePriorLossBuilder:
        return super().from_model(model)

    def knees_elbows_bending(
        self,
        weight: float | Tensor = 1.0,
        strategy: Literal["smplify", "sign"] = "smplify",
    ) -> nn.Module:
        """Encourage natural bending at knees and elbows.

        Parameters
        ----------
        weight : float | Tensor, optional
            Scalar weight for the prior. Defaults to 1.0.
        strategy : {"smplify", "sign"}, optional
            Heuristic strategy. "smplify" applies a quadratic penalty on the
            selected component. "sign" penalizes negative values (hyperextension)
            via ReLU.
        """
        w = weight if isinstance(weight, torch.Tensor) else torch.tensor(weight, device=self.device, dtype=self.dtype)

        # Body joint order used to build pose_body in containers
        body_joints = [
            "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
            "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
            "neck", "left_collar", "right_collar", "head",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
        ]

        # Indices within body AA (63) for target joints
        def joint_aa_slice(name: str) -> slice:
            idx = body_joints.index(name)
            start = 3 * idx
            return slice(start, start + 3)

        knees = [joint_aa_slice("left_knee"), joint_aa_slice("right_knee")]
        elbows = [joint_aa_slice("left_elbow"), joint_aa_slice("right_elbow")]

        class _BendingPrior(nn.Module):
            def __init__(self, w_in: Tensor, strat: str) -> None:
                super().__init__()
                self.weight: Tensor
                self.register_buffer("weight", w_in)
                self.m_strategy = strat

            def forward(self, output: UnifiedSmplOutput) -> Tensor:
                # Extract body AA from full pose: skip global orient (3)
                pose = output.full_pose
                if pose.shape[1] < 3 + 63:
                    raise ValueError("full_pose does not contain 63 body AA components")
                body = pose[:, 3 : 3 + 63]

                # Select components: use Y-axis (index 1) by default as heuristic
                comps = []
                for sl in knees + elbows:
                    aa = body[:, sl]
                    comps.append(aa[:, 1])  # (B,)
                vals = torch.stack(comps, dim=1)  # (B, 4)

                if self.m_strategy == "sign":
                    # Penalize negative values (hyperextension)
                    penal = torch.relu(-vals)
                    loss = (penal ** 2).mean()
                elif self.m_strategy == "smplify":
                    # Quadratic penalty encouraging a modest positive bend
                    target = 0.2  # radians, mild flexion
                    penal = (vals - target) ** 2
                    loss = penal.mean()
                else:
                    raise ValueError(f"Unknown strategy: {self.m_strategy}")

                return self.weight * loss

        return _BendingPrior(w, strategy)
