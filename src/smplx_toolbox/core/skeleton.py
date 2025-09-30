"""Generic human skeleton using a NetworkX joint graph.

This module provides a lightweight, framework‑agnostic skeleton abstraction for
retargeting and kinematics built on top of ``networkx`` only. The skeleton is a
directed acyclic graph (DAG) whose nodes represent joints. Each joint stores a
4×4 homogeneous transform relative to its parent joint (``T_local``). World
transforms are obtained by a forward‑kinematics pass over the DAG in
topological order: ``T_global[child] = T_global[parent] @ T_local[child]``. The
root joint's local transform is defined relative to world.

Notes
-----
- Human motion pipelines (SMPL/SMPL‑X/T2M) primarily operate on joint names and
  local transforms; link/body names are not modeled here.
- ``GenericSkeleton`` is a regular Python class (not an attrs class) because it
  manages a mutable graph and performs validation logic that is clearer without
  attrs magic. The per‑joint data structure ``JointNode`` uses attrs for
  concise, typed value semantics.

Examples
--------
>>> from smplx_toolbox.core.skeleton import GenericSkeleton
>>> names = ["pelvis", "thigh"]
>>> offsets = [np.array([0.0, 0.0, 1.0])]  # thigh is +1m in Z from pelvis
>>> skel = GenericSkeleton.from_chain(names, offsets)
>>> G = skel.get_joint_topology()
>>> T = skel.get_global_transforms()
>>> np.allclose(T["thigh"][:3, 3], [0, 0, 1])
True
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeAlias, cast

import numpy as np
import networkx as nx
from attrs import define, field


Transmat_4x4: TypeAlias = np.ndarray  # (4, 4) homogeneous transform


@define
class JointNode:
    """A joint node with a name and local transform.

    Attributes
    ----------
    name : str
        Joint name (unique key of the node in the graph).
    T_local : Transmat_4x4
        Homogeneous transform from parent joint to this joint.
    index : int | None
        Optional integer index carried for convenience.
    tags : set[str]
        Optional tags for labeling (e.g., "left", "arm").
    """

    name: str
    T_local: Transmat_4x4 = field(converter=_as_transmat)
    index: Optional[int] = None
    tags: set[str] = field(factory=set)

    @T_local.validator
    def _validate_T(self, _attr: Any, value: Transmat_4x4) -> None:
        if not _is_homogeneous(value):  # pragma: no cover - defensive
            raise ValueError("T_local must be a (4,4) homogeneous transform")


class GenericSkeleton:
    """NetworkX‑based joint graph skeleton.

    This class manages a mutable joint DAG and provides forward kinematics and
    topology utilities. It is intentionally a regular class (not attrs).
    """

    def __init__(self, root_name: str, *, name: Optional[str] = None) -> None:
        if not root_name:
            raise ValueError("root_name must be a non-empty string")
        self._g: nx.DiGraph = nx.DiGraph()
        self._root: str = root_name
        self.name: str = name or __import__("uuid").uuid4().hex
        # Initialize the root with identity local transform
        root_node = JointNode(name=root_name, T_local=_eye4())
        self._g.add_node(root_name, joint=root_node)

    # -----------
    # Construction
    # -----------
    @classmethod
    def from_nodes_edges(
        cls,
        nodes: Mapping[str, Transmat_4x4],
        edges: Sequence[Tuple[str, str]],
        *,
        root: str,
        name: Optional[str] = None,
    ) -> "GenericSkeleton":
        if root not in nodes:
            raise ValueError("root must be present in nodes")
        skel = cls(root, name=name)
        # Add/overwrite root local transform
        skel.set_local_transform(root, nodes[root])
        for n, T in nodes.items():
            if n == root:
                continue
            skel._add_joint_node(JointNode(name=n, T_local=T))
        for parent, child in edges:
            skel._add_edge(parent, child)
        skel._validate_graph()
        return skel

    @classmethod
    def from_chain(
        cls,
        names: Sequence[str],
        offsets: Sequence[np.ndarray],
        *,
        name: Optional[str] = None,
    ) -> "GenericSkeleton":
        if len(names) < 1:
            raise ValueError("names must contain at least one joint")
        if len(offsets) != len(names) - 1:
            raise ValueError("offsets length must be len(names)-1")
        skel = cls(names[0], name=name)
        for i in range(1, len(names)):
            p = names[i - 1]
            c = names[i]
            off = np.asarray(offsets[i - 1], dtype=np.float32).reshape(3)
            T = _eye4()
            T[:3, 3] = off
            skel.add_joint(JointNode(name=c, T_local=T), parent=p)
        skel._validate_graph()
        return skel

    # -------
    # Mutators
    # -------
    def add_joint(self, node: JointNode, *, parent: Optional[str]) -> None:
        """Add a joint node with an explicit parent.

        Use ``parent=None`` only when adding the root (must equal base_joint_name).
        """
        if parent is None:
            if node.name != self._root:
                raise ValueError("Only the root may be added with parent=None")
            if node.name in self._g:
                # Replace root's local transform
                self._g.nodes[node.name]["joint"] = node
                return
        else:
            if parent not in self._g:
                raise KeyError(f"parent joint '{parent}' does not exist")
        self._add_joint_node(node)
        if parent is not None:
            self._add_edge(parent, node.name)
        self._validate_graph()

    def _add_joint_node(self, node: JointNode) -> None:
        if node.name in self._g:
            raise ValueError(f"joint '{node.name}' already exists")
        self._g.add_node(node.name, joint=node)

    def _add_edge(self, parent: str, child: str) -> None:
        if parent not in self._g or child not in self._g:
            raise KeyError("both parent and child must exist before adding edge")
        self._g.add_edge(parent, child)
        if not nx.is_directed_acyclic_graph(self._g):
            self._g.remove_edge(parent, child)
            raise ValueError("adding this edge creates a cycle in the joint graph")

    def set_local_transform(self, name: str, T_local: Transmat_4x4) -> None:
        if name not in self._g:
            raise KeyError(f"unknown joint '{name}'")
        T = _as_transmat(T_local)
        if not _is_homogeneous(T):
            raise ValueError("T_local must be a (4,4) homogeneous transform")
        joint: JointNode = self._g.nodes[name]["joint"]
        self._g.nodes[name]["joint"] = JointNode(
            name=joint.name, T_local=T, index=joint.index, tags=set(joint.tags)
        )

    # -------
    # Accessors
    # -------
    @property
    def base_joint_name(self) -> str:
        return self._root

    @property
    def joint_names(self) -> List[str]:
        return list(self._g.nodes)

    def get_local_transform(self, name: str) -> Transmat_4x4:
        if name not in self._g:
            raise KeyError(f"unknown joint '{name}'")
        T = self._g.nodes[name]["joint"].T_local
        return cast(Transmat_4x4, T.copy())

    # ---------
    # Topology
    # ---------
    def get_joint_topology(self) -> nx.DiGraph:
        """Return a copy of the joint DAG (parent→child)."""
        return self._g.copy(as_view=False)

    def parents(self) -> Dict[str, Optional[str]]:
        out: Dict[str, Optional[str]] = {}
        for n in self._g.nodes:
            preds = list(self._g.predecessors(n))
            if n == self._root:
                out[n] = None
            else:
                out[n] = preds[0] if preds else None
        return out

    def children(self) -> Dict[str, List[str]]:
        return {n: list(self._g.successors(n)) for n in self._g.nodes}

    def topo_order(self) -> List[str]:
        return list(nx.topological_sort(self._g))

    # -------------
    # Kinematics
    # -------------
    def get_global_transforms(self) -> Dict[str, Transmat_4x4]:
        """Compute world transforms for all joints via FK in topological order."""
        self._validate_graph()
        T_world: Dict[str, Transmat_4x4] = {}
        order = self.topo_order()
        for n in order:
            T_local = self._g.nodes[n]["joint"].T_local
            if n == self._root:
                T_world[n] = cast(Transmat_4x4, T_local.astype(np.float32, copy=True))
            else:
                preds = list(self._g.predecessors(n))
                if len(preds) != 1:
                    raise ValueError("each joint must have exactly one parent")
                p = preds[0]
                T_world[n] = cast(Transmat_4x4, (T_world[p] @ T_local).astype(np.float32, copy=False))
        return T_world

    def get_global_transform(self, name: str) -> Transmat_4x4:
        if name not in self._g:
            raise KeyError(f"unknown joint '{name}'")
        return self.get_global_transforms()[name]

    # -----------
    # Validation
    # -----------
    def _validate_graph(self) -> None:
        if not nx.is_directed_acyclic_graph(self._g):
            raise ValueError("joint graph must be a DAG")
        if self._root not in self._g:
            raise ValueError("root joint missing from graph")
        # in-degree of root must be 0; others must be 1
        for n in self._g.nodes:
            indeg = self._g.in_degree(n)
            if n == self._root and indeg != 0:
                raise ValueError("root joint must have no parent")
            if n != self._root and indeg != 1:
                raise ValueError("non-root joints must have exactly one parent")
        # Reachability: all nodes reachable from root
        reachable = set(nx.descendants(self._g, self._root)) | {self._root}
        if len(reachable) != self._g.number_of_nodes():
            raise ValueError("all joints must be reachable from the root")

    # -------
    # Summary
    # -------
    def summary(self) -> str:
        return (
            f"GenericSkeleton(name={self.name!r}, base_joint={self._root!r})\n"
            f"  joints: {self._g.number_of_nodes()} | edges: {self._g.number_of_edges()}"
        )


# -----------------
# Helper functions
# -----------------
def _eye4() -> Transmat_4x4:
    return np.eye(4, dtype=np.float32)


def _as_transmat(a: Any) -> Transmat_4x4:
    arr = np.asarray(a, dtype=np.float32)
    if arr.shape != (4, 4):
        raise ValueError("transform must have shape (4,4)")
    return cast(Transmat_4x4, arr)


def _is_homogeneous(T: np.ndarray, eps: float = 1e-6) -> bool:
    if T.shape != (4, 4):
        return False
    tail = T[3]
    return (
        abs(float(tail[0])) < eps
        and abs(float(tail[1])) < eps
        and abs(float(tail[2])) < eps
        and abs(float(tail[3]) - 1.0) < eps
    )
