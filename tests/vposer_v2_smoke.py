"""Plain script: load VPoser v2 using upstream source to sanity-check the ckpt.

This is NOT part of the pytest suite. It demonstrates how to:
- Add local `human_body_prior` sources to `sys.path`
- Instantiate the original VPoser model
- Load `data/vposer/vposer-v2.ckpt` weights
- Run a simple decode/encode smoke pass

Paste this into a Jupyter notebook cell if desired.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import torch


# 1) Prepend local human_body_prior src to sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(root_dir)  # project root
hbp_src = os.path.join(root_dir, "context", "refcode", "human_body_prior", "src")
if hbp_src not in sys.path:
    sys.path.insert(0, hbp_src)


# 2) Import the upstream VPoser class
from human_body_prior.models.vposer_model import VPoser  # type: ignore


@dataclass
class _ModelParams:
    num_neurons: int
    latentD: int


@dataclass
class _ModelPS:
    model_params: _ModelParams


# 3) Load checkpoint and infer architecture
ckpt_path = os.path.join(root_dir, "data", "vposer", "vposer-v2.ckpt")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

ckpt: dict[str, Any] = torch.load(ckpt_path, map_location="cpu")
sd: dict[str, torch.Tensor] = ckpt["state_dict"]

num_neurons = int(sd["vp_model.encoder_net.2.weight"].shape[0])
latentD = int(sd["vp_model.encoder_net.8.mu.weight"].shape[0])

# 4) Build VPoser and load weights
vp = VPoser(_ModelPS(_ModelParams(num_neurons=num_neurons, latentD=latentD)))
vp.eval()

weights = {
    k.replace("vp_model.", "", 1): v for k, v in sd.items() if k.startswith("vp_model.")
}
missing, unexpected = vp.load_state_dict(weights, strict=False)
if unexpected:
    raise RuntimeError(f"Unexpected keys: {unexpected}")
if missing:
    raise RuntimeError(f"Missing keys: {missing}")

# 5) Smoke: decode / encode
with torch.no_grad():
    z = torch.zeros(1, latentD)
    dec = vp.decode(z)
    print("Decoded pose_body shape:", dec["pose_body"].shape)  # (1, 21, 3)

    pose_aa = torch.zeros(1, 21, 3)
    qz = vp.encode(pose_aa.reshape(1, -1))
    print("Latent mean/std shapes:", qz.mean.shape, qz.scale.shape)
