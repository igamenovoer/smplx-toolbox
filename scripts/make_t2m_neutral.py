"""Build a neutral T2M (HumanML3D) skeleton sample with zero root transform.

Outputs a pickle with:
- t2m_neutral.joints: (22, 3) joints in T2M order (Y-up), FK from identity 6D.
- t2m_neutral.pose: (22, 6) continuous 6D rotations (all identity; root included).
- t2m_neutral.root_orient: (6,) global/root 6D rotation placeholder (all zeros).
- t2m_neutral.trans: (3,) global translation placeholder (all zeros).

Notes
- Joints are in the original T2M Y-up convention. Keep this for parity checks.
- The 6D identity corresponds to rotation matrix I: [1,0,0, 0,1,0].
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch


def build_t2m_neutral() -> dict[str, np.ndarray]:
    # Import HumanML3D reference code (vendored under context/refcode/HumanML3D)
    import sys

    base = Path(__file__).resolve().parent.parent / "context" / "refcode" / "HumanML3D"
    sys.path.insert(0, str(base))

    # Patch numpy legacy aliases expected by HumanML3D code
    import numpy as _np
    if not hasattr(_np, "float"):  # NumPy >= 2.0
        _np.float = float  # type: ignore[attr-defined]

    from paramUtil import t2m_raw_offsets, t2m_kinematic_chain  # type: ignore
    from common.skeleton import Skeleton  # type: ignore
    from common.quaternion import cont6d_to_matrix_np  # type: ignore

    J = 22
    # Identity 6D for all joints (root included)
    ident6 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    cont6d = np.tile(ident6, (J, 1)).astype(np.float32)  # (22, 6)

    # Root pos zeros, Y-up convention as in T2M
    root_pos = np.zeros((1, 3), dtype=np.float32)

    # Build skeleton with raw offsets (unit directions) and FK
    skel = Skeleton(offset=torch.from_numpy(t2m_raw_offsets), kinematic_tree=t2m_kinematic_chain, device="cpu")  # type: ignore
    skel.set_offset(torch.from_numpy(t2m_raw_offsets).float())
    cont6d_b = cont6d[np.newaxis, ...]  # (1,22,6)
    joints = skel.forward_kinematics_cont6d_np(cont6d_b, root_pos, do_root_R=True)[0].astype(np.float32)

    out = {
        "t2m_neutral.joints": joints,          # (22,3) Y-up
        "t2m_neutral.pose": cont6d,            # (22,6) identity 6D
        "t2m_neutral.root_orient": np.zeros((6,), dtype=np.float32),  # placeholder (0s)
        "t2m_neutral.trans": np.zeros((3,), dtype=np.float32),        # placeholder (0s)
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a neutral T2M (HumanML3D) sample with zero root transform")
    ap.add_argument("--out", type=str, default="tmp/t2m_neutral.pkl", help="Output pickle path")
    args = ap.parse_args()

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    data = build_t2m_neutral()
    with open(args.out, "wb") as f:
        pickle.dump(data, f)
    print(f"[ok] Wrote neutral T2M sample to {args.out}")


if __name__ == "__main__":
    main()
