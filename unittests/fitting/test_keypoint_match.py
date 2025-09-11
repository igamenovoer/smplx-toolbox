from __future__ import annotations

import os
import pickle
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest
import torch
import smplx

from smplx_toolbox.core.constants import CoreBodyJoint
from smplx_toolbox.core.unified_model import UnifiedSmplInputs, UnifiedSmplModel
from smplx_toolbox.optimization import KeypointMatchLossBuilder, VPoserPriorLossBuilder
from smplx_toolbox.vposer import load_vposer
from smplx_toolbox.utils import select_device


TMP_ROOT = Path("tmp/unittests/fitting")
MODEL_ROOT = Path("data/body_models")
VPOSER_CKPT = Path("data/vposer/vposer-v2.ckpt")

class Params:
    """Adjustable parameters for keypoint matching unit tests."""

    # Optimization knobs
    NOISE_SCALE: float = 0.3
    STEPS: int = 10
    LR: float = 0.05

    # VPoser prior weights
    VPOSER_POSE_FIT: float = 0.5
    VPOSER_LATENT_L2: float = 0.1

    # Round-trip latent tolerance (mean absolute error)
    LATENT_RTT_TOL: float = 1.0

    # Joint subsets
    BASIC_SUBSET: list[str] = [
        CoreBodyJoint.LEFT_WRIST.value,
        CoreBodyJoint.RIGHT_FOOT.value,
    ]
    RICH_SUBSET: list[str] = [
        CoreBodyJoint.LEFT_WRIST.value,
        CoreBodyJoint.RIGHT_WRIST.value,
        CoreBodyJoint.LEFT_FOOT.value,
        CoreBodyJoint.RIGHT_FOOT.value,
        CoreBodyJoint.LEFT_ELBOW.value,
        CoreBodyJoint.RIGHT_ELBOW.value,
        CoreBodyJoint.LEFT_HIP.value,
        CoreBodyJoint.RIGHT_HIP.value,
    ]


def _build_smplx() -> UnifiedSmplModel:
    if not MODEL_ROOT.exists():
        pytest.skip("data/body_models missing")
    try:
        base = smplx.create(
            str(MODEL_ROOT), model_type="smplx", gender="neutral", use_pca=False, batch_size=1, ext="pkl"
        )
    except AssertionError as e:
        pytest.skip(f"SMPL-X resources missing: {e}")
    device = select_device()
    base = base.to(device)
    return UnifiedSmplModel.from_smpl_model(base)


@pytest.mark.unit
def test_vposer_latent_round_trip() -> None:
    if not VPOSER_CKPT.exists():
        pytest.skip("VPoser checkpoint not found")
    device = select_device()
    vposer = load_vposer(str(VPOSER_CKPT), map_location=device)
    vposer.to(device=device).eval()

    # Sample latent and round-trip via decode->encode(mean)
    B = 2
    D = int(vposer.latent_dim)
    torch.manual_seed(0)
    z0 = torch.randn((B, D), device=device)
    pose = vposer.decode(z0)["pose_body"]  # (B,21,3)
    q = vposer.encode(pose.view(B, -1))
    z1 = q.mean

    # Expect approximate consistency (loose tolerance due to VAE)
    diff = (z0 - z1).abs().mean().item()
    assert diff < Params.LATENT_RTT_TOL, f"Round-trip latent mismatch too large: {diff}"


def _optimize_simple_kpts(
    model: UnifiedSmplModel,
    *,
    use_vposer: bool,
    out_path: Path,
    subset: list[str] | None = None,
    title: str | None = None,
) -> None:
    # Setup targets from neutral + noise for a tiny subset of joints
    neutral = model.forward(UnifiedSmplInputs())
    if subset is None:
        subset = [CoreBodyJoint.LEFT_WRIST.value, CoreBodyJoint.RIGHT_FOOT.value]
    with torch.no_grad():
        sel = model.select_joints(neutral.joints, names=subset)  # (B,N,3)
        targets = sel + float(Params.NOISE_SCALE) * torch.randn_like(sel)
    targets_map: dict[str, torch.Tensor] = {subset[i]: targets[:, i] for i in range(len(subset))}

    # Trainable pose
    device, dtype = model.device, model.dtype
    root_orient = torch.nn.Parameter(torch.zeros((1, 3), device=device, dtype=dtype))
    pose_body = torch.nn.Parameter(torch.zeros((1, 63), device=device, dtype=dtype))
    params = [root_orient, pose_body]
    opt = torch.optim.Adam(params, lr=Params.LR)

    # Data term
    km = KeypointMatchLossBuilder.from_model(model)
    term_data = km.by_target_positions(targets_map, robust="gmof", rho=100.0, reduction="mean")

    # Optional VPoser term
    term_vposer = None
    if use_vposer:
        if not VPOSER_CKPT.exists():
            pytest.skip("VPoser checkpoint not found")
        vposer = load_vposer(str(VPOSER_CKPT), map_location=device)
        vposer.to(device=device).eval()
        vp = VPoserPriorLossBuilder.from_vposer(model, vposer)
        term_vposer = vp.by_pose(pose_body, w_pose_fit=1.0, w_latent_l2=0.1)

    # Few iterations to keep unit test quick
    for _ in range(Params.STEPS):
        opt.zero_grad()
        out = model.forward(UnifiedSmplInputs(root_orient=root_orient, pose_body=pose_body))
        loss = term_data(out)
        if term_vposer is not None:
            loss = loss + term_vposer(out)
        loss.backward()
        opt.step()

    # Save artifact for later visualization
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact: dict[str, Any] = {
        "title": title or ("Keypoint Match" + (" + VPoser" if use_vposer else " (L2)")),
        "model": {
            "model_root": str(MODEL_ROOT),
            "model_type": "smplx",
            "gender": "neutral",
            "ext": "pkl",
        },
        "subset_names": subset,
        "targets": {k: v[0].detach().cpu().numpy() for k, v in targets_map.items()},
        "initial": {
            "root_orient": torch.zeros_like(root_orient).cpu().numpy(),
            "pose_body": torch.zeros_like(pose_body).cpu().numpy(),
        },
        "optimized": {
            "root_orient": root_orient.detach().cpu().numpy(),
            "pose_body": pose_body.detach().cpu().numpy(),
        },
        "use_vposer": bool(use_vposer),
    }
    with out_path.open("wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)


@pytest.mark.unit
def test_keypoint_match_saves_artifact() -> None:
    model = _build_smplx()
    out_path = TMP_ROOT / "smoke_test_keypoint_match.pkl"
    # Minimal subset without VPoser
    _optimize_simple_kpts(
        model,
        use_vposer=False,
        out_path=out_path,
        subset=Params.BASIC_SUBSET,
                          title="UnitTest: 3D Keypoint Match (L2)")
    assert out_path.exists(), "Artifact .pkl not written"
    with out_path.open("rb") as f:
        data = pickle.load(f)
    assert "targets" in data and "optimized" in data and "model" in data


@pytest.mark.unit
def test_keypoint_match_with_vposer_saves_artifact() -> None:
    model = _build_smplx()
    out_path = TMP_ROOT / "smoke_test_keypoint_match_vposer.pkl"
    _optimize_simple_kpts(
        model,
        use_vposer=True,
        out_path=out_path,
        subset=Params.RICH_SUBSET,
        title="UnitTest: 3D Keypoint Match + VPoser",
    )
    assert out_path.exists(), "Artifact .pkl not written"
    with out_path.open("rb") as f:
        data = pickle.load(f)
    assert data.get("use_vposer") is True
