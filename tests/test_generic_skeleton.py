import numpy as np

from smplx_toolbox.core.skeleton import GenericSkeleton


def test_generic_skeleton_joint_graph_and_fk() -> None:
    # Build a simple chain pelvis -> thigh with 1m Z offset
    names = ["pelvis", "thigh"]
    offsets = [np.array([0.0, 0.0, 1.0], dtype=np.float32)]
    skel = GenericSkeleton.from_chain(names, offsets)

    # Basics
    assert skel.base_joint_name == "pelvis"
    assert set(skel.joint_names) == {"pelvis", "thigh"}

    # FK at default (identity locals except translation offsets)
    T = skel.get_global_transforms()
    Tp = T["pelvis"]
    Tt = T["thigh"]
    assert np.allclose(Tp, np.eye(4), atol=1e-6)
    assert np.allclose(Tt[:3, 3], np.array([0.0, 0.0, 1.0]), atol=1e-6)

    # Joint topology: parent->child edge
    import networkx as nx  # type: ignore

    G: nx.DiGraph = skel.get_joint_topology()
    assert isinstance(G, nx.DiGraph)
    assert set(G.nodes()) == {"pelvis", "thigh"}
    assert set(G.edges()) == {("pelvis", "thigh")}

    # Parent/children helpers
    parents = skel.parents()
    children = skel.children()
    assert parents["pelvis"] is None and parents["thigh"] == "pelvis"
    assert children["pelvis"] == ["thigh"] and children["thigh"] == []
