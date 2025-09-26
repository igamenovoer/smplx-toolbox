import os

import numpy as np

from smplx_toolbox.utils.humanml_mapping import (
    T2MSkeleton,
    create_neutral_t2m_skeleton,
)


def test_t2m_neutral_shapes():
    s = create_neutral_t2m_skeleton()
    # Basic shapes
    assert isinstance(s, T2MSkeleton)
    assert s.joints_local.shape == (22, 3)
    assert s.pose6d.shape == (22, 6)
    assert s.root_orient6d.shape == (6,)
    assert s.trans.shape == (3,)
    # Identity root → global == local
    R = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32).reshape(6)
    assert np.allclose(s.root_orient6d, R)
    assert np.allclose(s.trans, np.zeros((3,), dtype=np.float32))


if __name__ == "__main__":
    # Manual visual check: run this file directly to open a PyVista window
    # Uses background=False to show a blocking viewer
    s = create_neutral_t2m_skeleton()
    try:
        s.show(background=False, show_axes=True, labels=True)
    except Exception as e:
        print(f"[warn] Visualization failed (likely headless env): {e}")

