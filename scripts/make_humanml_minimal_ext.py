"""Create a minimal HumanML3D extended bundle (results_ext.npy) from results.npy.

This is a stopgap when FlowMDM hasn't emitted results_ext.npy yet.
It derives joints_world and transl from results.npy and fills the rest with
reasonable defaults so downstream converters can run.

Usage:
  pixi run -e dev python scripts/make_humanml_minimal_ext.py \
    --input_dir tmp/flowmdm-out/humanml3d
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def build_minimal_ext(input_dir: Path) -> Path:
    res = input_dir / "results.npy"
    if not res.exists():
        raise FileNotFoundError(f"results.npy not found under {input_dir}")
    D: dict[str, Any] = np.load(res, allow_pickle=True).item()
    motion = np.asarray(D["motion"])  # (B,22,3,T)
    text = list(D.get("text", []))
    lengths = D.get("lengths", None)
    if lengths is not None:
        try:
            lengths = int(np.asarray(lengths).sum())
        except Exception:
            lengths = int(np.asarray(motion)[0].shape[-1])
    else:
        lengths = int(np.asarray(motion)[0].shape[-1])

    jw = motion[0].transpose(2, 0, 1).astype(np.float32)  # (T,22,3)
    T = int(jw.shape[0])
    cont6d = np.zeros((T, 22, 6), dtype=np.float32)
    r_quat = np.zeros((T, 4), dtype=np.float32)
    r_quat[:, 0] = 1.0  # identity
    r_pos = jw[:, 0, :].astype(np.float32)  # pelvis world pos
    foot = np.zeros((T, 4), dtype=np.float32)
    feats = np.zeros((T, 263), dtype=np.float32)

    sample = {
        "feats_denorm": feats,
        "r_quat": r_quat,
        "r_pos": r_pos,
        "cont6d": cont6d,
        "joints_world": jw,
        "foot_contacts": foot,
        "length": lengths,
        "text": " /// ".join(text) if text else "",
        "fps": 20,
    }

    out = input_dir / "results_ext.npy"
    obj = {"samples": [sample], "meta": {"fps": 20, "notes": "Minimal bundle built from results.npy"}}
    np.save(out, np.array(obj, dtype=object), allow_pickle=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build minimal HumanML3D extended bundle from results.npy")
    ap.add_argument("--input_dir", type=str, default="tmp/flowmdm-out/humanml3d")
    args = ap.parse_args()
    p = Path(args.input_dir).resolve()
    p.mkdir(parents=True, exist_ok=True)
    out = build_minimal_ext(p)
    print(f"[ok] Wrote {out}")


if __name__ == "__main__":
    main()

