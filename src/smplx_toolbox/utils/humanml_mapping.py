"""Utilities for mapping/retargeting HumanML3D (Text2Motion) joints to SMPL‑X.

This module defines the joint index correspondence between the 22‑joint
Text2Motion (T2M) skeleton and the toolbox's unified 22 core body joints
(`CoreBodyJoint`). It also provides helpers to convert HumanML3D 6D joint
rotations to SMPL‑X axis‑angle in the toolbox's canonical order.

Reference
---------
- See `context/refcode/FlowMDM/explain/howto-interpret-flowmdm-output.md`
  for T2M ordering, chains, and the conceptual mapping to SMPL/SMPL‑X.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import numpy as np
from attrs import define, field
from attrs import validators as v
try:  # Prefer SciPy for ndarray rotation utilities when available
    from scipy.spatial.transform import Rotation as sR  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    sR = None  # type: ignore[assignment]

from smplx_toolbox.core.constants import CoreBodyJoint, ModelType
from smplx_toolbox.core.containers import NamedPose
import kornia.geometry.conversions as Kconv
from smplx_toolbox.visualization.utils import add_connection_lines, add_axes


def humanml_joint_mapping() -> Tuple[List[CoreBodyJoint], List[Optional[int]]]:
    """Return the mapping from toolbox core joints → T2M indices.

    The returned lists are aligned: for each core joint in order (excluding
    Pelvis), the corresponding element of the index list is either the T2M
    joint index (0..21) or None if T2M does not explicitly contain that joint
    (e.g., 'left_collar', 'right_collar'). Missing joints should be filled
    with identity rotations during retargeting.

    Order of joints follows `CoreBodyJoint` excluding Pelvis since Pelvis is
    handled as SMPL‑X global orientation.

    Returns
    -------
    joints : list[CoreBodyJoint]
        Core joints in canonical order, excluding Pelvis.
    t2m_indices : list[Optional[int]]
        T2M index per joint or None when not present.
    """
    # Canonical order from CoreBodyJoint, excluding Pelvis (global orient)
    order = [
        CoreBodyJoint.LEFT_HIP,
        CoreBodyJoint.RIGHT_HIP,
        CoreBodyJoint.SPINE1,
        CoreBodyJoint.LEFT_KNEE,
        CoreBodyJoint.RIGHT_KNEE,
        CoreBodyJoint.SPINE2,
        CoreBodyJoint.LEFT_ANKLE,
        CoreBodyJoint.RIGHT_ANKLE,
        CoreBodyJoint.SPINE3,
        CoreBodyJoint.LEFT_FOOT,
        CoreBodyJoint.RIGHT_FOOT,
        CoreBodyJoint.NECK,
        CoreBodyJoint.LEFT_COLLAR,   # not explicitly present in T2M
        CoreBodyJoint.RIGHT_COLLAR,  # not explicitly present in T2M
        CoreBodyJoint.HEAD,
        CoreBodyJoint.LEFT_SHOULDER,
        CoreBodyJoint.RIGHT_SHOULDER,
        CoreBodyJoint.LEFT_ELBOW,
        CoreBodyJoint.RIGHT_ELBOW,
        CoreBodyJoint.LEFT_WRIST,
        CoreBodyJoint.RIGHT_WRIST,
    ]

    # T2M joint order (indices) per howto doc and FlowMDM HumanML pipeline
    # 0 Pelvis, 1 LHip, 2 RHip, 3 Spine1, 4 LKnee, 5 RKnee, 6 Spine2,
    # 7 LAnkle, 8 RAnkle, 9 Spine3, 10 LFoot, 11 RFoot, 12 Neck,
    # 13 LShoulder, 14 RShoulder, 15 Head, 16 LElbow, 17 RElbow,
    # 18 LWrist, 19 RWrist, 20 LHand (proxy), 21 RHand (proxy)
    t2m: Dict[CoreBodyJoint, Optional[int]] = {
        CoreBodyJoint.LEFT_HIP: 1,
        CoreBodyJoint.RIGHT_HIP: 2,
        CoreBodyJoint.SPINE1: 3,
        CoreBodyJoint.LEFT_KNEE: 4,
        CoreBodyJoint.RIGHT_KNEE: 5,
        CoreBodyJoint.SPINE2: 6,
        CoreBodyJoint.LEFT_ANKLE: 7,
        CoreBodyJoint.RIGHT_ANKLE: 8,
        CoreBodyJoint.SPINE3: 9,
        CoreBodyJoint.LEFT_FOOT: 10,
        CoreBodyJoint.RIGHT_FOOT: 11,
        CoreBodyJoint.NECK: 12,
        # Collars are not separate in T2M; keep them as identity
        CoreBodyJoint.LEFT_COLLAR: None,
        CoreBodyJoint.RIGHT_COLLAR: None,
        CoreBodyJoint.HEAD: 15,
        CoreBodyJoint.LEFT_SHOULDER: 13,
        CoreBodyJoint.RIGHT_SHOULDER: 14,
        CoreBodyJoint.LEFT_ELBOW: 16,
        CoreBodyJoint.RIGHT_ELBOW: 17,
        CoreBodyJoint.LEFT_WRIST: 18,
        CoreBodyJoint.RIGHT_WRIST: 19,
    }

    return order, [t2m[j] for j in order]


def _cont6d_to_axis_angle(cont6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to axis‑angle.

    Parameters
    ----------
    cont6d : torch.Tensor
        Tensor of shape (T, J, 6) or (J, 6).

    Returns
    -------
    torch.Tensor
        Axis‑angle tensor of shape (T, J, 3) (or (J, 3)).
    """
    if cont6d.dim() == 2:
        cont6d = cont6d.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]
    x = torch.nn.functional.normalize(x_raw, dim=-1)
    # orthogonalize y against x, then normalize
    y = y_raw - (x * (x * y_raw).sum(dim=-1, keepdim=True))
    y = torch.nn.functional.normalize(y, dim=-1)
    z = torch.cross(x, y, dim=-1)
    mats = torch.stack([x, y, z], dim=-2)  # (..., 3, 3)
    # Kornia expects (...,3,3) and returns (...,3) axis-angle rotvec
    aa = Kconv.rotation_matrix_to_axis_angle(mats.reshape(-1, 3, 3)).reshape(
        mats.shape[:-2] + (3,)
    )
    return aa.squeeze(0) if squeeze else aa


def retarget_t2m_cont6d_to_named_pose(
    cont6d: torch.Tensor,
    *,
    model_type: ModelType = ModelType.SMPLX,
) -> List[NamedPose]:
    """Retarget T2M (HumanML3D) 6D rotations to `NamedPose` sequence.

    Parameters
    ----------
    cont6d : torch.Tensor
        T2M continuous‑6D rotations, shape (T, 22, 6). Index 0 is Pelvis.
    model_type : ModelType, optional
        Target SMPL model type for `NamedPose` (default: SMPLX).

    Returns
    -------
    list[NamedPose]
        A list of `NamedPose` (length T), where each pose contains 21 axis‑angle
        vectors in toolbox canonical order (CoreBodyJoint without Pelvis). For
        joints missing from T2M (collars), identity rotations are used.
    """
    if cont6d.ndim != 3 or cont6d.shape[1] < 22 or cont6d.shape[2] != 6:
        raise ValueError("cont6d must have shape (T, 22, 6)")

    # Convert all to axis‑angle then map into canonical order
    aa_all = _cont6d_to_axis_angle(cont6d)  # (T, 22, 3)

    order, idxs = humanml_joint_mapping()
    T = int(aa_all.shape[0])
    seq: List[NamedPose] = []
    for t in range(T):
        npz = NamedPose(model_type=model_type, batch_size=1)
        for j, t2m_idx in zip(order, idxs):
            if t2m_idx is None:
                v = torch.zeros(1, 3, dtype=torch.float32)
            else:
                v = aa_all[t, int(t2m_idx)].view(1, 3).to(torch.float32)
            npz.set_joint_pose_value(j.value, v)
        seq.append(npz)
    return seq


# ------------------------------
# T2M neutral skeleton utilities
# ------------------------------

# T2M joint names in order (22 joints)
T2M_JOINT_NAMES: List[str] = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_shoulder",
    "right_shoulder",
    "head",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",   # proxy/end effector in T2M
    "right_hand",  # proxy/end effector in T2M
]

# Raw offsets and kinematic chain adapted from HumanML3D (self-contained copy)
T2M_JOINT_COUNT: int = 22
T2M_RAW_OFFSETS: np.ndarray = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
    ],
    dtype=np.float32,
)

T2M_KINEMATIC_CHAIN: List[List[int]] = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]


def _t2m_bone_connections() -> List[Tuple[int, int]]:
    """Return parent-child bone connections for the 22-joint T2M skeleton.

    Derived by connecting consecutive joints within each kinematic chain.
    """
    edges: List[Tuple[int, int]] = []
    for chain in T2M_KINEMATIC_CHAIN:
        for i in range(1, len(chain)):
            edges.append((chain[i - 1], chain[i]))
    return edges


def _cont6d_to_matrix_np(cont6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotations to rotation matrices (NumPy).

    Accepts (..., 6) and returns (..., 3, 3).
    """
    c = np.asarray(cont6d, dtype=np.float32)
    x_raw = c[..., 0:3]
    y_raw = c[..., 3:6]
    # Normalize x and make y orthonormal
    x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)
    z = np.cross(x, y_raw)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    y = np.cross(z, x)
    # Stack as rotation matrix with columns [x y z]
    mat = np.stack([x, y, z], axis=-1).astype(np.float32)
    # Optionally re-orthonormalize via SciPy for numerical stability
    if sR is not None:
        flat = mat.reshape(-1, 3, 3)
        mat = sR.from_matrix(flat).as_matrix().astype(np.float32).reshape(mat.shape)
    return mat


def _parents_from_chain(n_joints: int, chains: List[List[int]]) -> List[int]:
    parents = [0] * n_joints
    parents[0] = -1
    for chain in chains:
        for j in range(1, len(chain)):
            parents[chain[j]] = chain[j - 1]
    return parents


def _fk_t2m_cont6d_np(
    cont6d: np.ndarray,
    root_pos: np.ndarray,
    *,
    offsets: np.ndarray = T2M_RAW_OFFSETS,
    chains: List[List[int]] = T2M_KINEMATIC_CHAIN,
    use_root_rotation: bool = True,
) -> np.ndarray:
    """Forward kinematics for T2M cont6d rotations (NumPy, self-contained).

    Parameters
    ----------
    cont6d : np.ndarray
        (..., 22, 6)
    root_pos : np.ndarray
        (..., 3)
    offsets : np.ndarray
        (22, 3) unit directions per joint.
    chains : list[list[int]]
        Kinematic chains with root at index 0.
    use_root_rotation : bool
        If False, ignore root rotation and use identity for the base.
    """
    c = np.asarray(cont6d, dtype=np.float32)
    if c.ndim == 2:
        c = c[np.newaxis, ...]
    B = int(c.shape[0])
    J = int(c.shape[1])
    assert J == 22 and c.shape[2] == 6, "cont6d must have shape (B,22,6)"

    mats = _cont6d_to_matrix_np(c)  # (B,22,3,3)
    joints = np.zeros((B, J, 3), dtype=np.float32)
    joints[:, 0, :] = np.asarray(root_pos, dtype=np.float32).reshape(B, 3)
    parents = _parents_from_chain(J, chains)

    for chain in chains:
        if use_root_rotation:
            R = mats[:, 0, :, :]  # (B,3,3)
        else:
            R = np.tile(np.eye(3, dtype=np.float32), (B, 1, 1))
        for j in range(1, len(chain)):
            idx = chain[j]
            R = np.matmul(R, mats[:, idx, :, :])  # compose global orientation
            off = np.tile(offsets[idx].reshape(1, 3, 1), (B, 1, 1))  # (B,3,1)
            delta = (R @ off).reshape(B, 3)
            parent = parents[idx]
            joints[:, idx, :] = joints[:, parent, :] + delta
    return joints if cont6d.ndim == 3 else joints[0]


@define(kw_only=True)
class T2MSkeleton:
    """Generic T2M (HumanML3D) skeleton sample with explicit local coordinates.

    This container stores ``joints_local`` in the joint-local coordinate frame.
    To obtain global/world coordinates, apply the similarity transform composed
    from ``scale`` (per‑axis), ``root_orient6d`` (rotation), and ``trans`` as:

        joints_global = R(root_orient6d) @ (S(scale) @ joints_local.T) + trans[:, None]

    where ``R(·)`` converts the 6D rotation representation to a 3x3 matrix.
    and ``S(·)`` is a diagonal scaling matrix built from the 3‑vector ``scale``.

    Important
    - Users providing joint positions must supply local coordinates. If you
      have global/world joints instead, call ``from_global_joints`` to
      construct this structure with the inverse transform applied.

    Attributes
    ----------
    joints_local : np.ndarray
        (22, 3) joints in T2M order, in local coordinates (before root).
    pose6d : np.ndarray
        (22, 6) continuous 6D rotations, identity per joint by default.
    root_orient6d : np.ndarray
        (6,) root orientation in 6D representation.
    trans : np.ndarray
        (3,) root translation.
    scale : np.ndarray
        (3,) global scale applied before the root rotation (default: ones).
    joint_names : list[str]
        T2M joint names in order.
    up_dir_type : str
        Coordinate system tag: one of ``x_up|y_up|z_up`` (default: ``y_up``).
    """

    joints_local: np.ndarray = field(
        factory=lambda: np.zeros((T2M_JOINT_COUNT, 3), dtype=np.float32)
    )
    pose6d: np.ndarray = field(
        factory=lambda: np.tile(
            np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            (T2M_JOINT_COUNT, 1),
        )
    )
    root_orient6d: np.ndarray = field(
        factory=lambda: np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    )
    trans: np.ndarray = field(factory=lambda: np.zeros((3,), dtype=np.float32))
    scale: np.ndarray = field(factory=lambda: np.ones((3,), dtype=np.float32))
    joint_names: List[str] = field(factory=lambda: list(T2M_JOINT_NAMES))
    up_dir_type: str = field(
        default="y_up",
        validator=v.in_(("x_up", "y_up", "z_up")),
        metadata={"desc": "Up-direction tag: one of x_up|y_up|z_up"},
    )

    @classmethod
    def from_global_joints(
        cls,
        joints_global: np.ndarray,
        *,
        pose6d: Optional[np.ndarray] = None,
        root_orient6d: Optional[np.ndarray] = None,
        trans: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
        up_dir_type: str = "y_up",
        joint_names: Optional[List[str]] = None,
    ) -> "T2MSkeleton":
        """Create a T2M skeleton by inverting the root transform on global joints.

        Parameters
        ----------
        joints_global : np.ndarray
            (22, 3) global/world joints in T2M order.
        pose6d : np.ndarray, optional
            (22, 6) T2M per-joint 6D rotations. Defaults to identity.
        root_orient6d : np.ndarray, optional
            (6,) root orientation in 6D. Defaults to identity rotation if None.
        trans : np.ndarray, optional
            (3,) translation vector. Defaults to zeros if None.
        scale : np.ndarray, optional
            (3,) global scale applied before rotation. Defaults to ones.
        up_dir_type : str, optional
            One of ``x_up|y_up|z_up``. Default: ``y_up``.
        joint_names : list[str], optional
            Joint names in order. Defaults to standard T2M names.

        Returns
        -------
        T2MSkeleton
            Skeleton where ``joints_local`` is computed as
            ``S(scale)^{-1} · R(root)^T · (joints_global - trans)``.
        """
        J = 22
        g = np.asarray(joints_global, dtype=np.float32)
        if g.shape != (J, 3):
            raise ValueError(f"joints_global must be (22,3), got {g.shape}")

        if pose6d is None:
            ident6 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
            pose6d = np.tile(ident6, (J, 1)).astype(np.float32)
        else:
            pose6d = np.asarray(pose6d, dtype=np.float32)

        if root_orient6d is None:
            root_orient6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        else:
            root_orient6d = np.asarray(root_orient6d, dtype=np.float32)

        if trans is None:
            trans = np.zeros((3,), dtype=np.float32)
        else:
            trans = np.asarray(trans, dtype=np.float32)

        if scale is None:
            scale = np.ones((3,), dtype=np.float32)
        else:
            scale = np.asarray(scale, dtype=np.float32)

        # Invert root transform: local = R^T * (global - t)
        R_root = _cont6d_to_matrix_np(root_orient6d.reshape(1, 6))[0]
        # Apply inverse scale as part of inverse similarity transform
        invS = np.diag(1.0 / (scale.astype(np.float32) + 1e-8))
        local = (invS @ (R_root.T @ (g - trans).T)).T.astype(np.float32)

        return cls(
            joints_local=local,
            pose6d=pose6d,
            root_orient6d=root_orient6d,
            trans=trans,
            scale=scale,
            joint_names=list(T2M_JOINT_NAMES) if joint_names is None else joint_names,
            up_dir_type=up_dir_type,
        )

    @property
    def world_transform(self) -> np.ndarray:
        """Return the 4x4 left‑multiply world transform matrix.

        The transform composes translation, rotation, and per‑axis scale as:

            T = T_trans · T_rot · T_scale

        so that, for column 4‑vectors X = [x;1], X' = T · X.
        """
        R_root = _cont6d_to_matrix_np(self.root_orient6d.reshape(1, 6))[0]
        sx, sy, sz = [float(v) for v in np.asarray(self.scale, dtype=np.float32).reshape(3)]
        S = np.diag([sx, sy, sz, 1.0]).astype(np.float32)
        T_rot = np.eye(4, dtype=np.float32)
        T_rot[:3, :3] = R_root.astype(np.float32)
        T_trn = np.eye(4, dtype=np.float32)
        T_trn[:3, 3] = np.asarray(self.trans, dtype=np.float32).reshape(3)
        T = (T_trn @ T_rot) @ S
        return T


def create_neutral_t2m_skeleton() -> T2MSkeleton:
    """Create a T2M skeleton in neutral pose (identity 6D), zero root.

    Returns
    -------
    T2MSkeleton
        Container with joints (Y-up), pose6d, zeroed root orient (6,) and trans (3,).
    """
    J = 22
    ident6 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    pose6d = np.tile(ident6, (J, 1)).astype(np.float32)
    root_pos = np.zeros((1, 3), dtype=np.float32)
    joints_global = _fk_t2m_cont6d_np(
        pose6d[np.newaxis, ...], root_pos, use_root_rotation=True
    )[0]
    return T2MSkeleton(
        joints_local=joints_global,  # with identity root, global == local
        pose6d=pose6d,
        root_orient6d=np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        trans=np.zeros((3,), dtype=np.float32),
    )


    # end create_neutral_t2m_skeleton

    
def _apply_root_to_local(
    joints_local: np.ndarray, root_orient6d: np.ndarray, trans: np.ndarray, scale: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compose local joints with root transform → global joints.

    joints_global = R(root_orient6d) @ (S(scale) @ joints_local.T) + trans[:, None]
    """
    R_root = _cont6d_to_matrix_np(root_orient6d.reshape(1, 6))[0]
    if scale is None:
        scaled = joints_local.astype(np.float32)
    else:
        s = np.asarray(scale, dtype=np.float32).reshape(1, 3)
        scaled = (joints_local.astype(np.float32) * s)
    return (R_root @ scaled.T).T + trans.reshape(1, 3)


def _safe_add_labels(pl, points: np.ndarray, labels: List[str], *, font_size: int = 10):
    try:
        return pl.add_point_labels(
            points,
            labels,
            font_size=font_size,
            point_size=0,
            shape_opacity=0,
            always_visible=True,
        )
    except TypeError:
        return pl.add_point_labels(
            points,
            labels,
            font_size=font_size,
            point_size=0,
            shape_opacity=0,
        )


def _auto_scale(points: np.ndarray, fallback: float = 0.15) -> float:
    """Heuristic scale (meters) based on spatial extent for axes/labels."""
    if points.size == 0:
        return fallback
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extent = float(np.linalg.norm(maxs - mins))
    return max(fallback, 0.1 * extent)


def _pairs_from_edges(points: np.ndarray, edges: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    a = np.array([points[i] for i, _ in edges], dtype=float)
    b = np.array([points[j] for _, j in edges], dtype=float)
    return a, b


def _require_pyvista():
    import importlib

    if importlib.util.find_spec("pyvista") is None:
        raise ImportError("pyvista is required for T2MSkeleton.show(); install via 'pip install pyvista'")
    import pyvista as pv  # noqa: F401
    return pv


@define(kw_only=True)
class _ShowActors:
    points_actor: object | None = None
    labels_actor: object | None = None
    bones_actor: object | None = None
    axes_actors: dict | None = None


def _make_plotter(background: bool = False):
    pv = _require_pyvista()
    if background:
        try:
            import pyvistaqt as pvqt  # type: ignore

            return pvqt.BackgroundPlotter()
        except Exception:
            return pv.Plotter()
    return pv.Plotter()


def _add_t2m_skeleton_to_plotter(
    pl,
    joints_global: np.ndarray,
    *,
    labels: bool = True,
    label_font_size: int = 10,
) -> _ShowActors:
    edges = _t2m_bone_connections()
    a, b = _pairs_from_edges(joints_global, edges)
    actors = _ShowActors()
    actors.bones_actor = add_connection_lines(pl, a, b, color=(0.2, 0.6, 0.9), line_width=3)
    try:
        actors.points_actor = pl.add_points(
            joints_global,
            render_points_as_spheres=True,
            point_size=10,
            color=(1.0, 0.9, 0.1),
        )
    except TypeError:
        actors.points_actor = pl.add_mesh(joints_global, color=(1.0, 0.9, 0.1))

    if labels:
        actors.labels_actor = _safe_add_labels(pl, joints_global, list(T2M_JOINT_NAMES), font_size=label_font_size)
    return actors


# Attach method to class after its definition
def _t2m_show(self: "T2MSkeleton", *, background: bool = False, show_axes: bool = True, labels: bool = True):
    """Visualize the T2M skeleton in PyVista.

    Parameters
    ----------
    background : bool, optional
        If True, use a non-blocking BackgroundPlotter when available.
    show_axes : bool, optional
        Whether to render axes at the root joint.
    labels : bool, optional
        Whether to draw joint name labels next to joints.
    """
    pv = _require_pyvista()
    pl = _make_plotter(background=background)
    joints_g = _apply_root_to_local(self.joints_local, self.root_orient6d, self.trans, self.scale)
    _add_t2m_skeleton_to_plotter(pl, joints_g, labels=labels)
    if show_axes:
        scale = _auto_scale(joints_g)
        add_axes(pl, joints_g[0], scale=scale, labels=True)
    pl.add_text("T2M Skeleton", font_size=10)
    pl.show()


# Bind as a method of T2MSkeleton
setattr(T2MSkeleton, "show", _t2m_show)
