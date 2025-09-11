"""Show keypoint matching results from a saved artifact (.pkl).

Usage
-----
pixi run -e dev python scripts/show-keypoint-match.py --input tmp/unittests/fitting/smoke_test_keypoint_match.pkl

The .pkl artifact is expected to contain a dictionary with keys:
- model: { model_root, model_type, gender, ext }
- subset_names: list[str]
- targets: dict[name -> (3,)] or (B,3)
- initial: { root_orient, pose_body, (optional) left_hand_pose, right_hand_pose }
- optimized: same as initial

This script reconstructs the model, runs forwards with the initial and
optimized parameters (when present), and visualizes them along with target
keypoints and connection lines. By default, everything present in the
artifact is shown; there are no options to hide items.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import smplx
import torch

from smplx_toolbox.core.unified_model import UnifiedSmplInputs, UnifiedSmplModel
from smplx_toolbox.visualization import SMPLVisualizer, add_connection_lines


def _load_artifact(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def _to_tensor(x: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    arr = np.asarray(x)
    return torch.as_tensor(arr, device=device, dtype=dtype)


def main() -> None:
    ap = argparse.ArgumentParser(description="Show keypoint match artifact")
    ap.add_argument("--input", required=True, help="Path to .pkl artifact")
    ap.add_argument("--style", default="wireframe", choices=["wireframe", "surface", "points"], help="Mesh style")
    args = ap.parse_args()

    art = _load_artifact(Path(args.input))
    model_info = art["model"]
    model_root = model_info.get("model_root", "data/body_models")
    model_type = model_info.get("model_type", "smplx")
    gender = model_info.get("gender", "neutral")
    ext = model_info.get("ext", "pkl")

    base = smplx.create(str(model_root), model_type=model_type, gender=gender, use_pca=False, batch_size=1, ext=ext)
    uni: UnifiedSmplModel = UnifiedSmplModel.from_smpl_model(base)
    device, dtype = uni.device, uni.dtype

    subset_names = list(art.get("subset_names", []))
    targets_map = art.get("targets", {})
    tgt_stack = np.stack([np.asarray(targets_map[nm]) for nm in subset_names], axis=0)

    print("[viewer] Rendering legend:", flush=True)
    print("[viewer] - Targets: RED points", flush=True)
    print("[viewer] - Initial: GRAY mesh + points", flush=True)
    print("[viewer] - Optimized: BLUE mesh + points", flush=True)
    print("[viewer] - Connection lines: match predicted to targets (same color)", flush=True)
    print(f"[viewer] Mesh style: {args.style}", flush=True)

    viz: SMPLVisualizer = SMPLVisualizer.from_model(uni)
    plotter = viz.get_plotter()

    # Targets (red)
    plotter.add_points(tgt_stack, color=(1.0, 0.0, 0.0), render_points_as_spheres=True, point_size=12)

    def _draw_pose(name: str, color: tuple[float, float, float]) -> None:
        d = art.get(name, {})
        if not d:
            return
        kw: dict[str, torch.Tensor] = {}
        for key in ("root_orient", "pose_body", "left_hand_pose", "right_hand_pose", "trans"):
            if key in d:
                kw[key] = _to_tensor(d[key], device, dtype)
        out = uni.forward(UnifiedSmplInputs(**kw))
        viz.add_mesh(out, style=args.style, color=color, opacity=1.0)
        with torch.no_grad():
            sel = uni.select_joints(out.joints, names=subset_names)[0]
        pts = sel.detach().cpu().numpy()
        viz.add_smpl_joints(out, joints=subset_names, labels=False, color=color, size=0.012)
        add_connection_lines(plotter, pts, tgt_stack, color=color, line_width=2, opacity=0.9)

    # Always show all available entries
    _draw_pose("initial", (0.5, 0.5, 0.5))
    _draw_pose("optimized", (0.0, 0.3, 1.0))

    title = str(art.get("title", "Keypoint Match Viewer"))
    print(f"[viewer] Title: {title}", flush=True)
    plotter.add_text(title, font_size=12)
    plotter.reset_camera_clipping_range()
    plotter.show()


if __name__ == "__main__":
    main()
