from __future__ import annotations

import sys
import torch


def select_device() -> torch.device:
    """Select the best available torch device based on OS.

    Logic:
    - macOS: prefer Metal Performance Shaders (MPS) if available; else CPU.
    - Windows/Linux: prefer CUDA if available; else CPU.

    Returns
    -------
    torch.device
        The selected device object ("mps", "cuda", or "cpu").
    """
    is_macos = sys.platform == "darwin"
    if is_macos:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    # Windows/Linux
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

