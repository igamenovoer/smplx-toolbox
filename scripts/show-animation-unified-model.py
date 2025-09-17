"""Show Unified SMPL/SMPL-H/SMPL-X animation with selectable backend.

This viewer loads a SMPL-family model via `smplx`, wraps it with
`UnifiedSmplModel`, and visualizes an animation where each frame is a
`UnifiedSmplInputs` entry (a pickled list created by the FlowMDM converter).

Controls
--------
- Left/Right: Step backward/forward one frame (basic/qt)
- r: Reset camera view (basic/qt)
- Slider: Scrub timeline (all backends)
- Global axis widget is always visible

Backends
--------
- basic: PyVista on-screen rendering (turntable-style camera controls)
- qt: PyVistaQt via `pyvistaqt` (requires PyQt5 and XCB on Linux)
- browser: Trame-based browser UI (no Qt requirements; optional `--port`)

Examples
--------
    pixi run -e dev python scripts/show-animation-unified-model.py \
        --anim-file tmp/flowmdm-out/babel/unified_smpl_animation.pkl \
        --model-type smplx --body-model-dir data/body_models --backend browser

    pixi run -e dev python scripts/show-animation-unified-model.py \
        --anim-file tmp/flowmdm-out/babel/unified_smpl_animation.pkl \
        --backend browser --port 9000
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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

    Uses ``off_screen=True`` to avoid any Qt/xcb requirements.
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
    pl = pv.Plotter()
    pl.add_mesh(mesh, color="lightgray", smooth_shading=True)
    return pl, mesh


def build_plotter_qt(verts: List[np.ndarray], faces_idx: torch.Tensor):
    try:
        import pyvistaqt as pvqt  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Backend 'qt' requires pyvistaqt and a working Qt installation"
        ) from e
    mesh = pv.PolyData(verts[0], faces_to_pv(faces_idx))
    pl = pvqt.BackgroundPlotter()
    pl.add_mesh(mesh, color="lightgray", smooth_shading=True)
    return pl, mesh


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show Unified SMPL/SMPL-H/SMPL-X animation with selectable backend."
    )
    parser.add_argument(
        "--anim-file",
        type=str,
        required=True,
        help="Pickle containing list[UnifiedSmplInputs]",
    )
    parser.add_argument(
        "--model-type",
        choices=ModelType.values(),
        default="smplx",
        help="Base model type to load (smpl|smplh|smplx). Default: smplx",
    )
    parser.add_argument(
        "--body-model-dir",
        dest="body_models_dir",
        type=str,
        default="data/body_models",
        help="Parent directory containing 'smpl', 'smplh', 'smplx' folders",
    )
    # Back-compat alias
    parser.add_argument(
        "--body-models-path",
        dest="body_models_dir",
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gender",
        type=str,
        default="neutral",
        choices=["neutral", "male", "female"],
        help="Model gender",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["basic", "qt", "browser"],
        default="basic",
        help="Rendering backend: basic|qt|browser",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Browser backend port (auto-pick when omitted)",
    )

    args = parser.parse_args()

    anim: Optional[AnimationData] = load_animation(Path(args.anim_file))

    model = create_unified_model(
        model_type=ModelType(args.model_type),
        body_models_path=Path(args.body_models_dir),
        gender=args.gender,
    )

    frames = anim.frames if anim is not None else None
    verts = precompute_vertices(model, frames)

    # Shared update hook (browser backend uses this helper)
    def make_set_frame(
        _pl,
        _mesh,
        _after_render=None,
    ):
        def _set(i: int) -> None:
            i = int(np.clip(i, 0, len(verts) - 1))
            _mesh.points = verts[i]
            _pl.render()
            if _after_render is not None:
                try:
                    _after_render()
                except Exception:
                    pass
        return _set

    backend = args.backend

    if backend == "browser":
        # Build offscreen plotter for trame to drive
        pl = _make_plotter_offscreen()
        mesh = pv.PolyData(verts[0], faces_to_pv(model.faces))
        pl.add_mesh(mesh, color="lightgray", smooth_shading=True)
        try:
            ground = pv.Plane(
                center=(0.0, 0.0, 0.0),
                direction=(0.0, 0.0, 1.0),
                i_size=10.0,
                j_size=10.0,
                i_resolution=10,
                j_resolution=10,
            )
            pl.add_mesh(
                ground,
                color="#dddddd",
                opacity=0.35,
                show_edges=True,
                edge_color="#bcbcbc",
                line_width=1,
            )
        except Exception:
            pass
        try:
            pl.add_axes()
        except Exception:
            pass
        try:
            pl.enable_custom_trackball_style(
                left_button="environment_rotate",
                shift_left_button="pan",
                ctrl_left_button="spin",
                middle_button="pan",
                right_button="dolly",
            )
        except Exception:
            pass
        # We'll inject a view.update() to push renders to the browser
        view_ref = {}
        def after_render():
            v = view_ref.get("view")
            if v is not None:
                try:
                    v.update()
                except Exception:
                    pass

        set_frame = make_set_frame(pl, mesh, after_render)

        # Browser-based viewer via trame; auto-pick port
        from trame.app import get_server  # type: ignore
        from trame.ui.vuetify import SinglePageLayout  # type: ignore
        from trame.widgets import vtk as vtk_widgets, vuetify  # type: ignore

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
                view_ref["view"] = view
                if n > 1:
                    vuetify.VSlider(
                        v_model=("frame", 0),
                        min=0,
                        max=n - 1,
                        step=1,
                        dense=True,
                        hide_details=True,
                    )
                    with vuetify.VRow(classes="ma-0 pa-0"):
                        vuetify.VBtn("Prev", click=ctrl.prev_frame, classes="ma-1")
                        vuetify.VBtn("Next", click=ctrl.next_frame, classes="ma-1")

        pl.render()
        start_kwargs = {"address": "0.0.0.0", "open_browser": False}
        if args.port is not None:
            start_kwargs["port"] = args.port
        @server.controller.on_server_ready.add
        def _announce(**_kwargs):
            host = "127.0.0.1"
            port = start_kwargs.get("port", getattr(server, "port", None))
            if port is None:
                try:
                    port = server.port  # type: ignore[attr-defined]
                except Exception:
                    port = "?"
            print(f"[viewer] Browser backend available at http://{host}:{port}")
        server.start(**start_kwargs)
        return

    # basic/qt backends share the same interaction model
    if backend == "qt":
        pl, mesh = build_plotter_qt(verts, model.faces)
    else:
        pl, mesh = build_plotter(verts, model.faces)

    try:
        ground = pv.Plane(
            center=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, 1.0),
            i_size=10.0,
            j_size=10.0,
            i_resolution=10,
            j_resolution=10,
        )
        pl.add_mesh(
            ground,
            color="#dddddd",
            opacity=0.35,
            show_edges=True,
            edge_color="#bcbcbc",
            line_width=1,
        )
    except Exception:
        pass

    try:
        pl.add_axes()
    except Exception:
        pass

    # Animation state for basic/qt interactor
    state = {
        "frame": 0,
        "n": len(verts),
        "slider": None,
    }

    def set_frame(i: int, *, update_slider: bool = True) -> None:
        if state["n"] == 0:
            return
        i = int(np.clip(i, 0, state["n"] - 1))
        state["frame"] = i
        mesh.points = verts[i]
        pl.render()
        if update_slider and state["slider"] is not None:
            try:
                rep = state["slider"].GetRepresentation()  # type: ignore[attr-defined]
                rep.SetValue(float(i))  # type: ignore[attr-defined]
            except Exception:
                pass

    n_frames = state["n"]

    # Slider to scrub (keep handle so we can update it when stepping programmatically)
    slider_widget = None
    if n_frames > 1:
        def _on_slider(value: float) -> None:
            set_frame(int(round(value)), update_slider=False)

        slider_widget = pl.add_slider_widget(
            _on_slider,
            rng=[0, n_frames - 1],
            value=0,
            title=f"Frame (0..{n_frames-1})",
            pointa=(0.025, 0.1),
            pointb=(0.31, 0.1),
            style="modern",
        )
        state["slider"] = slider_widget

    # HUD and key bindings
    if backend in {"basic", "qt"}:
        try:
            pl.enable_custom_trackball_style(
                left_button="environment_rotate",
                shift_left_button="pan",
                ctrl_left_button="spin",
                middle_button="pan",
                right_button="dolly",
            )
        except Exception:
            pass

        pl.add_text(
            "←/→: Prev/Next  r: Reset view",
            font_size=10,
        )

        def on_left() -> None:
            if n_frames <= 1:
                return
            set_frame((state["frame"] - 1) % n_frames)

        def on_right() -> None:
            if n_frames <= 1:
                return
            set_frame((state["frame"] + 1) % n_frames)

        def on_reset_view() -> None:
            try:
                pl.reset_camera()
                pl.render()
            except Exception:
                pass

        # Register keyboard shortcuts (with VTK fallback)
        key_bindings_ok = True
        try:
            pl.add_key_event("Left", on_left)
            pl.add_key_event("Right", on_right)
            pl.add_key_event("r", on_reset_view)
        except Exception:
            key_bindings_ok = False

        # Fallback: raw VTK key observer if Plotter.add_key_event isn't available yet
        if not key_bindings_ok:
            try:
                def _vtk_key_handler(obj, evt):
                    try:
                        key = obj.GetKeySym()  # type: ignore[attr-defined]
                    except Exception:
                        return
                    if key == "Left":
                        on_left()
                    elif key == "Right":
                        on_right()
                    elif key in ("r", "R"):
                        on_reset_view()

                if hasattr(pl, "iren") and pl.iren is not None:  # type: ignore[attr-defined]
                    pl.iren.AddObserver("KeyPressEvent", _vtk_key_handler)  # type: ignore[attr-defined]
            except Exception:
                pass

    # Render
    if backend == "qt":
        # Block on Qt event loop
        try:
            pl.app.exec_()  # type: ignore[attr-defined]
        except Exception:
            pl.show()
    else:
        # In notebooks, prefer client backend
        if _is_notebook():
            pv.set_jupyter_backend("client")
            pl.show(jupyter_backend="client")
        else:
            pl.show()


if __name__ == "__main__":
    main()
