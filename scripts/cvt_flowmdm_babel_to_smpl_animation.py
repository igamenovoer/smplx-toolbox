"""Convert FlowMDM Babel outputs to a pickled list of UnifiedSmplInputs.

This utility reads FlowMDM's extended Babel outputs and constructs a list of
`UnifiedSmplInputs` objects, one per frame, populated with:
- named_pose: intrinsic joints only (no pelvis), using ModelType.SMPLH by default
- global_orient: pelvis AA (B, 3)
- transl: global translation (B, 3)

Notes
-----
- The script expects FlowMDM extended outputs produced by our tasks (see
  `pixi run flowmdm-gen-babel`).
- Primary source is `smpl_params.npy` (preferred). If missing, the script
  falls back to `smplx_pose.npy` + `smplx_global_orient.npy` + `smplx_transl.npy`.
- Batch size per frame is 1 (list entries are per-frame objects).

Examples
--------
Run from workspace root using the dev environment:

    pixi run -e dev python scripts/cvt_flowmdm_babel_to_smpl_animation.py \
        --input_dir tmp/flowmdm-out/babel \
        --output_path tmp/flowmdm-out/babel/unified_smpl_animation.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch

from smplx_toolbox.core.constants import CoreBodyJoint, ModelType
from smplx_toolbox.core.containers import NamedPose, UnifiedSmplInputs


def _load_babel_primary(input_dir: Path):
    """Load primary SMPL params from FlowMDM extended output.

    Returns
    -------
    dict | None
        Dict with keys: 'global_orient' (T,3), 'body_pose' (T,63),
        'left_hand_pose' (T,45) or None, 'right_hand_pose' (T,45) or None,
        'transl' (T,3), 'betas' (10,), or None if not available.
    """
    path = input_dir / "smpl_params.npy"
    if not path.exists():
        return None
    arr = np.load(path, allow_pickle=True)
    # Expect array of dicts, one per generated sample; use first sample
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return arr[0]
    raise RuntimeError(f"Unexpected format for {path}")


def _load_babel_fallback(input_dir: Path):
    """Load fallback SMPLX pose and supporting arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        smplx_pose (T,165), smplx_global_orient (T,3), smplx_transl (T,3).
    """
    pose_p = input_dir / "smplx_pose.npy"
    go_p = input_dir / "smplx_global_orient.npy"
    tr_p = input_dir / "smplx_transl.npy"
    if not (pose_p.exists() and go_p.exists() and tr_p.exists()):
        raise FileNotFoundError(
            "Missing SMPLX fallback arrays: require smplx_pose.npy, "
            "smplx_global_orient.npy, smplx_transl.npy"
        )
    pose_arr = np.load(pose_p, allow_pickle=True)
    go_arr = np.load(go_p, allow_pickle=True)
    tr_arr = np.load(tr_p, allow_pickle=True)
    # Expect object arrays where [0] is (T, dim)
    pose = pose_arr[0]
    go = go_arr[0]
    tr = tr_arr[0]
    return pose, go, tr


def _split_smplx_pose_ordered(pose_flat: np.ndarray) -> dict[str, np.ndarray]:
    """Split SMPLX flat pose (T,165) into named segments.

    Layout in FlowMDM export is:
        [global_orient(3), body(63), left_hand(45), right_hand(45), jaw(3), leye(3), reye(3)]
    """
    T, D = pose_flat.shape
    if D != 165:
        raise ValueError(f"Expected SMPLX pose dim=165, got {D}")
    offs = 0
    def take(n):
        nonlocal offs
        s = pose_flat[:, offs : offs + n]
        offs += n
        return s

    global_orient = take(3)
    body = take(63)
    lh = take(45)
    rh = take(45)
    jaw = take(3)
    le = take(3)
    re = take(3)
    return {
        "global_orient": global_orient.reshape(T, 3),
        "body_pose": body.reshape(T, 21, 3),
        "left_hand_pose": lh.reshape(T, 15, 3),
        "right_hand_pose": rh.reshape(T, 15, 3),
        "jaw_pose": jaw.reshape(T, 1, 3),
        "leye_pose": le.reshape(T, 1, 3),
        "reye_pose": re.reshape(T, 1, 3),
    }


def _build_named_pose_from_smpl_params(
    params: dict, *, model_type: ModelType = ModelType.SMPLH
) -> list[NamedPose]:
    """Construct a NamedPose per frame from primary SMPL params.

    Fills only core body joints (no hands/face) and leaves others at zero.
    """
    T = int(params["body_pose"].shape[0])
    body = params["body_pose"].reshape(T, 21, 3)  # (T,21,3), order = CoreBodyJoint[1:]

    named: list[NamedPose] = []
    for t in range(T):
        npz = NamedPose(model_type=model_type, batch_size=1)
        # Fill core body joints by name in canonical order
        body_idx = 0
        for j in CoreBodyJoint:
            if j == CoreBodyJoint.PELVIS:
                continue
            j_pose = torch.from_numpy(body[t, body_idx]).view(1, 3)
            npz.set_joint_pose_value(j.value, j_pose)
            body_idx += 1
        named.append(npz)
    return named


def _build_named_pose_from_smplx(
    split: dict[str, np.ndarray], *, model_type: ModelType = ModelType.SMPLX
) -> list[NamedPose]:
    """Construct a NamedPose per frame from SMPLX split segments.

    Populates body, hands (if non-zero), and face (if non-zero) segments.
    """
    T = int(split["body_pose"].shape[0])
    named: list[NamedPose] = []

    for t in range(T):
        npz = NamedPose(model_type=model_type, batch_size=1)

        # Core body joints (skip pelvis) — order matches CoreBodyJoint[1:]
        body = split["body_pose"][t]  # (21,3)
        body_idx = 0
        for j in CoreBodyJoint:
            if j == CoreBodyJoint.PELVIS:
                continue
            j_pose = torch.from_numpy(body[body_idx]).view(1, 3)
            npz.set_joint_pose_value(j.value, j_pose)
            body_idx += 1

        # Hands (left, right) — fill if any non-zero
        lh = split["left_hand_pose"][t]  # (15,3)
        rh = split["right_hand_pose"][t]  # (15,3)
        if np.abs(lh).sum() > 0:
            # Use aggregate helper by explicit joint names via set_joint_pose_value
            from smplx_toolbox.core.constants import HandFingerJoint

            for i, name in enumerate([e.value for e in HandFingerJoint if e.name.startswith("LEFT_")]):
                npz.set_joint_pose_value(name, torch.from_numpy(lh[i]).view(1, 3))
        if np.abs(rh).sum() > 0:
            from smplx_toolbox.core.constants import HandFingerJoint

            right_names = [e.value for e in HandFingerJoint if e.name.startswith("RIGHT_")]
            for i, name in enumerate(right_names):
                npz.set_joint_pose_value(name, torch.from_numpy(rh[i]).view(1, 3))

        # Face (jaw, eyes) — fill if any non-zero
        jaw = split["jaw_pose"][t][0]
        le = split["leye_pose"][t][0]
        re = split["reye_pose"][t][0]
        if np.abs(jaw).sum() > 0:
            from smplx_toolbox.core.constants import FaceJoint

            npz.set_joint_pose_value(FaceJoint.JAW.value, torch.from_numpy(jaw).view(1, 3))
        if np.abs(le).sum() > 0 and np.abs(re).sum() > 0:
            from smplx_toolbox.core.constants import FaceJoint

            npz.set_joint_pose_value(FaceJoint.LEFT_EYE_SMPLHF.value, torch.from_numpy(le).view(1, 3))
            npz.set_joint_pose_value(FaceJoint.RIGHT_EYE_SMPLHF.value, torch.from_numpy(re).view(1, 3))

        named.append(npz)
    return named


def convert(input_dir: Path, output_path: Path) -> Path:
    """Convert FlowMDM Babel output directory to list[UnifiedSmplInputs].

    Returns
    -------
    Path
        Path to the written pickle file.
    """
    input_dir = input_dir.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    params = _load_babel_primary(input_dir)
    named_list: List[NamedPose]
    global_orient: np.ndarray
    transl: np.ndarray

    if params is not None:
        # Prefer SMPL-H source (hands may be None → left at zeros)
        named_list = _build_named_pose_from_smpl_params(params, model_type=ModelType.SMPLH)
        global_orient = params["global_orient"].reshape(-1, 3)
        transl = params["transl"].reshape(-1, 3)
    else:
        # Fallback to SMPL-X flat pose
        pose_flat, go, tr = _load_babel_fallback(input_dir)
        split = _split_smplx_pose_ordered(pose_flat)
        named_list = _build_named_pose_from_smplx(split, model_type=ModelType.SMPLX)
        global_orient = go.reshape(-1, 3)
        transl = tr.reshape(-1, 3)

    T = len(named_list)
    if not (len(global_orient) == T and len(transl) == T):
        raise RuntimeError(
            f"Mismatched lengths: poses={T}, global_orient={len(global_orient)}, transl={len(transl)}"
        )

    unified: list[UnifiedSmplInputs] = []
    for t in range(T):
        u = UnifiedSmplInputs(
            named_pose=named_list[t],
            global_orient=torch.from_numpy(global_orient[t]).view(1, 3).to(torch.float32),
            trans=torch.from_numpy(transl[t]).view(1, 3).to(torch.float32),
        )
        unified.append(u)

    with open(output_path, "wb") as f:
        pickle.dump(unified, f)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert FlowMDM Babel outputs to UnifiedSmplInputs animation."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="tmp/flowmdm-out/babel",
        help="Directory containing FlowMDM Babel outputs (extended)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output pickle path (defaults to <input_dir>/unified_smpl_animation.pkl)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else input_dir / "unified_smpl_animation.pkl"
    )

    out = convert(input_dir, output_path)
    print(f"[ok] Wrote UnifiedSmplInputs list ({out})")


if __name__ == "__main__":
    main()

