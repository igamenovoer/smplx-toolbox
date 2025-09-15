from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest
import torch
import smplx

from smplx_toolbox.core import NamedPose
from smplx_toolbox.core.constants import CoreBodyJoint, ModelType
from smplx_toolbox.core.unified_model import UnifiedSmplInputs, UnifiedSmplModel
from smplx_toolbox.fitting import SmplKeypointFittingHelper
from smplx_toolbox.utils import select_device
from smplx_toolbox.vposer.model import VPoserModel


MODEL_ROOT = Path("data/body_models")
VPOSER_CKPT = Path("data/vposer/vposer-v2.ckpt")


class Params:
    NOISE_SCALE: float = 0.3
    STEPS: int = 15
    LR: float = 0.05
    L2_WEIGHT: float = 1e-2
    SHAPE_L2_WEIGHT: float = 1e-2
    ROBUST_KIND: str = "gmof"
    ROBUST_RHO: float = 100.0

    ENABLE_GLOBAL_ORIENT: bool = True
    ENABLE_GLOBAL_TRANSLATION: bool = True
    ENABLE_SHAPE_DEFORMATION: bool = True

    VPOSER_POSE_FIT: float = 0.5
    VPOSER_LATENT_L2: float = 0.1


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


def _make_targets(model: UnifiedSmplModel, names: list[str]) -> dict[str, torch.Tensor]:
    neutral = model.forward(UnifiedSmplInputs())
    with torch.no_grad():
        sel = model.select_joints(neutral.joints, names=names)
        tgt = sel + float(Params.NOISE_SCALE) * torch.randn_like(sel)
    return {names[i]: tgt[:, i] for i in range(len(names))}


@pytest.mark.unit
def test_helper_keypoint_match_basic() -> None:
    model = _build_smplx()
    device, dtype = model.device, model.dtype

    names = [CoreBodyJoint.LEFT_WRIST.value, CoreBodyJoint.RIGHT_FOOT.value]
    # Unique seed per run
    seed = uuid4().int & 0xFFFFFFFF
    torch.manual_seed(seed)
    np.random.seed(seed)
    targets = _make_targets(model, names)

    # Trainable initial
    npz = NamedPose(model_type=ModelType(str(model.model_type)), batch_size=1)
    npz.intrinsic_pose = torch.nn.Parameter(torch.zeros_like(npz.intrinsic_pose))
    root = torch.nn.Parameter(torch.zeros((1, 3), device=device, dtype=dtype)) if Params.ENABLE_GLOBAL_ORIENT else None
    trans = torch.nn.Parameter(torch.zeros((1, 3), device=device, dtype=dtype)) if Params.ENABLE_GLOBAL_TRANSLATION else None
    betas = torch.nn.Parameter(torch.zeros((1, model.num_betas), device=device, dtype=dtype)) if Params.ENABLE_SHAPE_DEFORMATION else None

    init = UnifiedSmplInputs(named_pose=npz, global_orient=root, trans=trans, betas=betas)

    helper = SmplKeypointFittingHelper.from_model(model)
    helper.set_keypoint_targets(targets, robust=Params.ROBUST_KIND, rho=Params.ROBUST_RHO)
    helper.set_dof_global_orient(Params.ENABLE_GLOBAL_ORIENT)
    helper.set_dof_global_translation(Params.ENABLE_GLOBAL_TRANSLATION)
    helper.set_dof_shape_deform(Params.ENABLE_SHAPE_DEFORMATION)
    helper.set_reg_pose_l2(Params.L2_WEIGHT)
    helper.set_reg_shape_l2(Params.SHAPE_L2_WEIGHT)

    it = helper.init_fitting(init, optimizer="adam", lr=Params.LR, num_iter_per_step=5)
    losses: list[float] = []
    for i, status in zip(range(Params.STEPS), it):
        losses.append(status.loss_total)
    assert losses[0] > losses[-1], "expected loss to decrease with optimization"


@pytest.mark.unit
def test_helper_with_vposer() -> None:
    if not VPOSER_CKPT.exists():
        pytest.skip("VPoser checkpoint missing")
    model = _build_smplx()

    names = [CoreBodyJoint.LEFT_WRIST.value, CoreBodyJoint.RIGHT_WRIST.value]
    targets = _make_targets(model, names)

    npz = NamedPose(model_type=ModelType(str(model.model_type)), batch_size=1)
    npz.intrinsic_pose = torch.nn.Parameter(torch.zeros_like(npz.intrinsic_pose))
    init = UnifiedSmplInputs(named_pose=npz)

    helper = SmplKeypointFittingHelper.from_model(model)
    helper.set_keypoint_targets(targets)
    helper.set_reg_pose_l2(Params.L2_WEIGHT)

    vposer = VPoserModel.from_checkpoint(VPOSER_CKPT, map_location=model.device)
    helper.vposer_init(vposer_model=vposer)
    helper.vposer_set_reg(w_pose_fit=Params.VPOSER_POSE_FIT, w_latent_l2=Params.VPOSER_LATENT_L2)

    it = helper.init_fitting(init, optimizer="adam", lr=Params.LR, num_iter_per_step=5)
    losses: list[float] = []
    for i, status in zip(range(Params.STEPS), it):
        losses.append(status.loss_total)
    assert losses[0] > losses[-1]

