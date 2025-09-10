from __future__ import annotations

import math

import torch
import pytest

from smplx_toolbox.optimization import GMoF, gmof


@pytest.mark.unit
def test_gmof_basic_behavior() -> None:
    r = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    rho = 1.0

    # Closed-form per definition
    expected = (rho**2) * (r**2 / (r**2 + rho**2))
    m = GMoF(rho=rho)
    out = m(r)
    assert torch.allclose(out, expected)

    # Functional API parity
    out_func = gmof(r, rho=rho)
    assert torch.allclose(out_func, expected)


@pytest.mark.unit
def test_gmof_limits() -> None:
    # Small residuals ~ quadratic
    r_small = torch.tensor([1e-6, -1e-6])
    rho = 0.5
    out_small = gmof(r_small, rho=rho)
    # Approx equals r^2 when r^2 << rho^2
    assert torch.allclose(out_small, r_small**2, rtol=1e-4, atol=1e-8)

    # Large residuals saturate to rho^2
    r_large = torch.tensor([1000.0, -1000.0])
    out_large = gmof(r_large, rho=rho)
    target = torch.full_like(r_large, rho**2)
    # Asymptotic bound; with large residuals, very close to rho^2
    assert torch.allclose(out_large, target, rtol=1e-6, atol=1e-8)


@pytest.mark.unit
def test_gmof_gradients() -> None:
    # Check that gradients flow and are finite
    r = torch.tensor([0.3, -0.7, 1.2], requires_grad=True)
    rho = 1.3
    loss = gmof(r, rho=rho).sum()
    loss.backward()
    assert r.grad is not None
    assert torch.isfinite(r.grad).all()
