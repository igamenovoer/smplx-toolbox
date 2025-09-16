"""Show Unified SMPL/SMPL-H/SMPL-X model and optional animation via PyVista.

This viewer loads a SMPL-family model via `smplx`, wraps it with
`UnifiedSmplModel`, and visualizes either a static pose or an animation where
each frame is a `UnifiedSmplInputs` entry (pickled list).

Controls
--------
- Space: Play/Pause
- Left/Right: Previous/Next frame
- R: Reset to frame 0
- Slider: Scrub timeline

Examples
--------
Static model (default zero pose):
    pixi run -e dev python scripts/show-animation-unified-model.py \
        --model-type smplx --body-models-path data/body_models

With animation (list[UnifiedSmplInputs] pickle):
    pixi run -e dev python scripts/show-animation-unified-model.py \
        --anim-file tmp/flowmdm-out/babel/unified_smpl_animation.pkl \
        --model-type smplh --body-models-path data/body_models --autoplay --fps 24
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import os

import numpy as np
import pyvista as pv
import torch

import smplx  # type: ignore
from smplx_toolbox.core.constants import ModelType
from smplx_toolbox.core.containers import UnifiedSmplInputs
from smplx_toolbox.core.unified_model import UnifiedSmplModel


@dataclass
class AnimationData:
    frames: List[UnifiedSmplInputs]
    n_frames: int


def load_animation(pkl_path: Path) -> AnimationData:
    with open(pkl_path, "rb") as f:
        frames = pickle.load(f)
    if not isinstance(frames, list) or not all(
        isinstance(x, UnifiedSmplInputs) for x in frames
    ):
        raise TypeError(
            "Animation pickle must be a list of UnifiedSmplInputs objects"
        )
    return AnimationData(frames=frames, n_frames=len(frames))


def create_unified_model(
    *, model_type: ModelType, body_models_path: Path, gender: str = "neutral"
) -> UnifiedSmplModel:
    # smplx expects the parent folder that contains `smpl`, `smplh`, `smplx` subfolders
    base = smplx.create(
        model_path=str(body_models_path),
        model_type=str(model_type),
        gender=gender,
        use_pca=False,
        batch_size=1,
    )
    return UnifiedSmplModel.from_smpl_model(base)


def precompute_vertices(
    model: UnifiedSmplModel, frames: Optional[List[UnifiedSmplInputs]]
) -> List[np.ndarray]:
    # If no frames, render a single default zero-pose frame
    if not frames:
        zero = UnifiedSmplInputs()
        out = model(zero)
        V = out.vertices.detach().cpu().numpy()[0]
        return [V]

    verts: List[np.ndarray] = []
    for u in frames:
        out = model(u)
        verts.append(out.vertices.detach().cpu().numpy()[0])
    return verts


def faces_to_pv(faces_idx: torch.Tensor) -> np.ndarray:
    """Convert (F,3) long tensor to PyVista faces array."""
    f = faces_idx.detach().cpu().numpy().astype(np.int64)
    F = f.shape[0]
    # PyVista expects [3, i0, i1, i2, 3, j0, j1, j2, ...]
    sizes = np.full((F, 1), 3, dtype=np.int64)
    return np.hstack([sizes, f]).ravel()


def _make_plotter_offscreen() -> pv.Plotter:
    """Create a PyVista plotter suitable for browser/server rendering.

    Uses off_screen=True to avoid any Qt/xcb requirements.
    """
    return pv.Plotter(off_screen=True)


def _is_notebook() -> bool:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        return ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def build_plotter(verts: List[np.ndarray], faces_idx: torch.Tensor) -> tuple[pv.Plotter, pv.PolyData]:
    mesh = pv.PolyData(verts[0], faces_to_pv(faces_idx))
    pl = _make_plotter_offscreen()
    pl.add_mesh(mesh, color="lightgray", smooth_shading=True)
    return pl, mesh


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show Unified SMPL/SMPL-H/SMPL-X model and optional animation."
    )
    parser.add_argument(
        "--anim-file",
        type=str,
        default=None,
        help="Optional pickle containing list[UnifiedSmplInputs]",
    )
    parser.add_argument(
        "--model-type",
        choices=ModelType.values(),
        default="smplx",
        help="Base model type to load (smpl|smplh|smplx). Default: smplx",
    )
    parser.add_argument(
        "--body-models-path",
        type=str,
        default="data/body_models",
        help="Parent directory containing 'smpl', 'smplh', 'smplx' folders",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default="neutral",
        choices=["neutral", "male", "female"],
        help="Model gender",
    )
    parser.add_argument(
        "--fps", type=int, default=24, help="Playback FPS for animation"
    )
    parser.add_argument(
        "--autoplay",
        action="store_true",
        help="Start playing automatically when animation is present",
    )

    args = parser.parse_args()

    anim: Optional[AnimationData] = None
    if args.anim_file is not None:
        anim = load_animation(Path(args.anim_file))

    model = create_unified_model(
        model_type=ModelType(args.model_type),
        body_models_path=Path(args.body_models_path),
        gender=args.gender,
    )

    frames = anim.frames if anim is not None else None
    verts = precompute_vertices(model, frames)

    # Build plotter and mesh
    pl, mesh = build_plotter(verts, model.faces)

    # Update hook to set a specific frame
    def set_frame(i: int) -> None:
        i = int(np.clip(i, 0, len(verts) - 1))
        mesh.points = verts[i]
        pl.render()

    if _is_notebook():
        # Jupyter: prefer client-side rendering for minimal deps
        pv.set_jupyter_backend("client")
        # Expose a simple slider in the scene for scrubbing
        if len(verts) > 1:
            pl.add_slider_widget(
                lambda v: set_frame(int(v)),
                rng=[0, len(verts) - 1],
                value=0,
                title=f"Frame (0..{len(verts)-1})",
                pointa=(0.025, 0.1),
                pointb=(0.31, 0.1),
                style="modern",
            )
        pl.show(jupyter_backend="client")
    else:
        # Browser-based viewer via trame on port 8899
        from trame.app import get_server
        from trame.ui.vuetify import SinglePageLayout
        from trame.widgets import vtk as vtk_widgets, vuetify

        server = get_server(client_type="vue2")
        state, ctrl = server.state, server.controller
        state.frame = 0
        n = len(verts)

        @state.change("frame")
        def _on_frame_change(frame, **_):
            set_frame(int(frame))

        @ctrl.add("prev_frame")
        def _prev_frame():
            state.frame = max(0, int(state.frame) - 1)

        @ctrl.add("next_frame")
        def _next_frame():
            state.frame = min(n - 1, int(state.frame) + 1)

        with SinglePageLayout(server) as layout:
            layout.title.set_text("Unified SMPL Viewer")
            with layout.toolbar:
                vuetify.VSpacer()
            with layout.content:
                view = vtk_widgets.VtkRemoteView(pl.ren_win)
                # Controls
                if n > 1:
                    vuetify.VSlider(v_model=("frame", 0), min=0, max=n - 1, step=1, dense=True, hide_details=True)
                    with vuetify.VRow(classes="ma-0 pa-0"):
                        vuetify.VBtn("Prev", click=ctrl.prev_frame, classes="ma-1")
                        vuetify.VBtn("Next", click=ctrl.next_frame, classes="ma-1")

        # Initial render and start server
        pl.render()
        server.start(port=8899, open_browser=True)


if __name__ == "__main__":
    main()
