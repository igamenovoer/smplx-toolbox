"""Load VPoser v2 checkpoints into the local lightweight runtime.

This utility mirrors the layer naming used by the original VPoser so that
weights from the provided ``.ckpt`` file can be loaded into
``smplx_toolbox.vposer.VPoserModel`` without installing upstream packages.
"""

from __future__ import annotations

from typing import Any

import torch

from .model import VPoserConfig, VPoserModel


def _infer_config_from_state_dict(sd: dict[str, torch.Tensor]) -> VPoserConfig:
    """Infer architecture config from a VPoser state dict.

    Supports both the original Lightning key layout and the remapped keys used
    by this toolbox.

    Parameters
    ----------
    sd : dict[str, torch.Tensor]
        State dictionary with parameter tensors.

    Returns
    -------
    VPoserConfig
        Configuration containing ``num_neurons``, ``latent_dim`` and
        ``num_joints``.
    """
    # Supports both original ('encoder_net.*') and remapped ('encoder_backbone.*' + 'encoder_head.*') keys
    if "encoder_backbone.2.weight" in sd and "encoder_head.mu.weight" in sd:
        num_neurons = int(sd["encoder_backbone.2.weight"].shape[0])
        latent_dim = int(sd["encoder_head.mu.weight"].shape[0])
    else:
        num_neurons = int(sd["encoder_net.2.weight"].shape[0])
        latent_dim = int(sd["encoder_net.8.mu.weight"].shape[0])
    return VPoserConfig(num_neurons=num_neurons, latent_dim=latent_dim, num_joints=21)


def _strip_prefix(sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """Return a copy of ``sd`` without a common key prefix.

    Parameters
    ----------
    sd : dict[str, torch.Tensor]
        Source state dict to process.
    prefix : str
        The prefix to remove (e.g., ``"vp_model."``).

    Returns
    -------
    dict[str, torch.Tensor]
        A new dict whose keys have ``prefix`` removed when present.
    """
    plen = len(prefix)
    return {k[plen:]: v for k, v in sd.items() if k.startswith(prefix)}


def _remap_encoder_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap encoder keys to match the local module split.

    The original model stores the encoder as a single Sequential under
    ``encoder_net``. Locally we split it into ``encoder_backbone`` and
    ``encoder_head``. This function renames keys accordingly while keeping
    decoder keys intact.

    Mapping
    -------
    - ``encoder_net.0..7`` → ``encoder_backbone.0..7``
    - ``encoder_net.8.<sub>`` → ``encoder_head.<sub>`` (e.g., ``mu.weight``)

    Parameters
    ----------
    sd : dict[str, torch.Tensor]
        State dict to remap.

    Returns
    -------
    dict[str, torch.Tensor]
        A new state dict with updated key names.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("encoder_net."):
            rest = k[len("encoder_net.") :]
            if rest.startswith("8."):
                out["encoder_head." + rest[2:]] = v
            else:
                out["encoder_backbone." + rest] = v
        else:
            out[k] = v
    return out


def load_vposer(
    ckpt_path: str,
    *,
    map_location: str | torch.device = "cpu",
) -> VPoserModel:
    """Load a VPoser v2 checkpoint into a :class:`VPoserModel`.

    Parameters
    ----------
    ckpt_path : str
        Filesystem path to the ``.ckpt`` file.
    map_location : str | torch.device, optional
        Device location to load the tensors (default: ``"cpu"``).

    Returns
    -------
    VPoserModel
        A model with weights loaded and set to evaluation mode.

    Raises
    ------
    ValueError
        If the checkpoint structure is invalid.
    RuntimeError
        If unexpected keys are encountered when loading weights.
    """
    ckpt: Any = torch.load(ckpt_path, map_location=map_location)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError("Invalid VPoser checkpoint format: missing state_dict")
    full_sd: dict[str, torch.Tensor] = ckpt["state_dict"]
    # Strip lightning-style 'vp_model.' prefix to match our module
    sd = _strip_prefix(full_sd, "vp_model.")
    sd = _remap_encoder_keys(sd)
    cfg = _infer_config_from_state_dict(sd)
    model = VPoserModel(cfg)
    incompatible = model.load_state_dict(sd, strict=False)
    # Access attributes to avoid relying on return tuple typing
    unexpected = getattr(incompatible, "unexpected_keys", [])
    missing = getattr(incompatible, "missing_keys", [])
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading VPoser: {sorted(unexpected)}")
    # Missing can include buffers if any; we expect weights to load
    model.eval()
    return model
