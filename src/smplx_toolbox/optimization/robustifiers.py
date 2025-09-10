"""Robust penalty functions for keypoint fitting and priors.

This module provides a small set of robustifiers used in SMPL/SMPL-X model
fitting. It includes a reimplementation of the Geman–McClure objective (GMoF)
as used by SMPLify-X.
"""

from __future__ import annotations

from typing import Final

import torch.nn as nn
from torch import Tensor


class GMoF(nn.Module):
    """Geman–McClure robustifier (a.k.a. GMoF) as a module.

    Given residuals ``r``, the robust penalty is defined as::

        rho(r) = rho^2 * (r^2 / (r^2 + rho^2))

    where ``rho > 0`` controls the transition from quadratic to saturated
    behavior. Small residuals behave approximately like L2; large residuals
    are downweighted and asymptotically bounded by ``rho^2``.

    Parameters
    ----------
    rho : float, optional
        Scale parameter of the robustifier. The units should match the units
        of the residual ``r``. Defaults to ``1.0``.

    Notes
    -----
    - This formulation mirrors the implementation used in SMPLify-X.
    - The module is differentiable and can be used inside optimization
      objectives directly.
    """

    def __init__(self, rho: float = 1.0) -> None:
        super().__init__()
        self.rho: Final[float] = float(rho)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"rho = {self.rho}"

    def forward(self, residual: Tensor) -> Tensor:
        """Apply the GMoF penalty elementwise.

        Parameters
        ----------
        residual : torch.Tensor
            Residual tensor of arbitrary shape.

        Returns
        -------
        torch.Tensor
            Robustified penalty with the same shape as ``residual``.
        """
        squared_res = residual**2
        denom = squared_res + (self.rho**2)
        dist = squared_res / denom
        return (self.rho**2) * dist


def gmof(residual: Tensor, rho: float = 1.0) -> Tensor:
    """Functional Geman–McClure robustifier.

    Parameters
    ----------
    residual : torch.Tensor
        Residual tensor of arbitrary shape.
    rho : float, optional
        Scale parameter; see :class:`GMoF`. Defaults to ``1.0``.

    Returns
    -------
    torch.Tensor
        Robustified penalty with the same shape as ``residual``.
    """
    module = GMoF(rho=rho)
    return module.forward(residual)
