from __future__ import annotations

from typing import Any

from attrs import define, field

from smplx_toolbox.core import UnifiedSmplInputs, UnifiedSmplOutput


@define(kw_only=True)
class FitStepStatus:
    """Per‑step status snapshot produced by SmplKeypointFittingHelper.

    Captures the scalar loss, an optional breakdown of loss terms, references
    to the current live inputs (parameters), and the latest model output. The
    contained ``params`` fields are not deep‑copied: they reference the same
    tensors that the optimizer updates in place.

    Attributes
    ----------
    step : int
        0‑based outer step index (increments once per yielded step).
    loss_total : float
        Total scalar loss value for this step after the last inner iteration.
    loss_terms : dict[str, float]
        Optional per‑term loss breakdown (e.g., {"data": ..., "vposer": ...}).
    params : UnifiedSmplInputs
        Live references to the current parameters used for the forward pass.
        These are not copies; reading them reflects the latest in‑place updates.
    output : UnifiedSmplOutput
        The model output computed at this step.
    grad_norm : float | None
        Optional diagnostic: aggregated gradient norm over trainable tensors.
    extras : dict[str, Any]
        Freeform metadata for callers to stash additional information.
    """

    step: int  # 0-based outer step index
    loss_total: float  # total scalar loss at this step
    loss_terms: dict[str, float]  # per-term breakdown (if available)
    params: UnifiedSmplInputs  # live references to current inputs/parameters
    output: UnifiedSmplOutput  # latest model output
    grad_norm: float | None = None  # optional gradient norm diagnostic
    extras: dict[str, Any] = field(factory=dict)  # user-defined metadata
