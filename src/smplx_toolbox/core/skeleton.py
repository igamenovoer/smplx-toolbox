"""Generic human skeleton using a NetworkX joint graph.

This module provides a lightweight, framework‑agnostic skeleton abstraction for
retargeting and kinematics built on top of ``networkx`` only. The skeleton is a
directed acyclic graph (DAG) whose nodes represent joints. Each joint stores a
4×4 homogeneous transform relative to its parent joint (``transmat_wrt_parent``).
World transforms are obtained by a forward‑kinematics pass over the DAG in
topological order: ``T_global[child] = T_global[parent] @ transmat_wrt_parent[child]``.
The root joint's local transform is defined relative to world.

Notes
-----
- Human motion pipelines (SMPL/SMPL‑X/T2M) primarily operate on joint names and
  local transforms; link/body names are not modeled here.
- ``GenericSkeleton`` is a regular Python class (not an attrs class) because it
  manages a mutable graph and performs validation logic that is clearer without
  attrs magic. The per‑joint data structure ``JointNode`` uses attrs for
  concise, typed value semantics.

Classes
-------
JointNode
    Data container for a joint's local transform with respect to its parent.
GenericSkeleton
    NetworkX‑based joint graph with forward kinematics and topology helpers.

Examples
--------
>>> import numpy as np
>>> from smplx_toolbox.core.skeleton import GenericSkeleton
>>> names = ["pelvis", "thigh"]
>>> offsets = [np.array([0.0, 0.0, 1.0], dtype=np.float32)]
>>> skel = GenericSkeleton.from_chain(names, offsets)
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

    This is a pure data model (``attrs``), not a behavior class. It holds the
    4×4 homogeneous transform from the parent joint to this joint.

    Attributes
    ----------
    name : str
        Joint name (unique key of the node in the graph).
    transmat_wrt_parent : Transmat_4x4
        Homogeneous transform from parent joint to this joint.
    index : int or None
        Optional integer index carried for convenience.
    tags : set of str
        Optional tags for labeling (e.g., "left", "arm").
    """

    name: str
    transmat_wrt_parent: Transmat_4x4 = field(converter=_as_transmat)
    index: Optional[int] = None
    tags: set[str] = field(factory=set)

    @transmat_wrt_parent.validator
    def _validate_T(self, _attr: Any, value: Transmat_4x4) -> None:
        if not _is_homogeneous(value):  # pragma: no cover - defensive
            raise ValueError("transmat_wrt_parent must be a (4,4) homogeneous transform")


class GenericSkeleton:
    """NetworkX‑based joint graph skeleton.

    This class manages a mutable joint DAG and provides forward kinematics and
    topology utilities. It is intentionally a regular class (not attrs) to
    encapsulate behavior and validation. Internal state uses ``m_`` prefixes
    per the repository's Python coding guide.

    Attributes
    ----------
    m_joint_graph : networkx.DiGraph
        Directed acyclic graph (parent→child) of joints. Nodes store a
        ``joint: JointNode`` attribute.
    m_root : str
        Name of the root joint.
    m_name : str
        Skeleton label for logging/identification.
    """

    def __init__(self, root_name: str, *, name: Optional[str] = None) -> None:
        """Construct a skeleton with a specified root joint.

        Parameters
        ----------
        root_name : str
            Name of the root joint (e.g., ``"pelvis"``).
        name : str, optional
            Optional skeleton label used for logging/identification.

        Raises
        ------
        ValueError
            If ``root_name`` is empty.
        """
        if not root_name:
            raise ValueError("root_name must be a non-empty string")
        # Member variables use m_ prefix per style guide
        self.m_joint_graph: nx.DiGraph = nx.DiGraph()
        self.m_root: str = root_name
        self.m_name: str = name or __import__("uuid").uuid4().hex
        # Initialize the root with identity local transform
        root_node = JointNode(name=root_name, transmat_wrt_parent=_eye4())
        self.m_joint_graph.add_node(root_name, joint=root_node)

    # Public name accessor to keep backward compatibility
    @property
    def name(self) -> str:
        """Return the skeleton label.

        Returns
        -------
        str
            The skeleton's name/label.
        """
        return self.m_name

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
        """Build a skeleton from nodes and directed edges.

        Parameters
        ----------
        nodes : Mapping[str, Transmat_4x4]
            Mapping from joint name to a 4×4 transform relative to its parent.
            The root's transform is relative to world.
        edges : Sequence[tuple[str, str]]
            Parent→child joint edges. Graph must be acyclic and connected.
        root : str
            Name of the root joint; must be present in ``nodes``.
        name : str, optional
            Skeleton label.

        Returns
        -------
        GenericSkeleton
            A validated skeleton instance.

        Raises
        ------
        ValueError
            If ``root`` not in nodes or graph validation fails.
        KeyError
            If an edge references a missing joint.
        """
        if root not in nodes:
            raise ValueError("root must be present in nodes")
        skel = cls(root, name=name)
        # Add/overwrite root local transform
        skel.set_local_transform(root, nodes[root])
        for n, T in nodes.items():
            if n == root:
                continue
            skel._add_joint_node(JointNode(name=n, transmat_wrt_parent=T))
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
        """Create a simple chain skeleton from names and translation offsets.

        Parameters
        ----------
        names : sequence of str
            Joint names ordered from root to leaf. Must contain at least one
            joint name; the first is treated as the root.
        offsets : sequence of ndarray
            Per‑edge translation offsets (3,) from parent to child. Must have
            length ``len(names) - 1``.
        name : str, optional
            Skeleton label.

        Returns
        -------
        GenericSkeleton
            A validated chain skeleton where each edge is a pure translation.

        Raises
        ------
        ValueError
            If lengths are inconsistent or names empty.
        """
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
            skel.add_joint(JointNode(name=c, transmat_wrt_parent=T), parent=p)
        skel._validate_graph()
        return skel

    # -------
    # Mutators
    # -------
    def add_joint(self, node: JointNode, *, parent: Optional[str]) -> None:
        """Add a joint node with an explicit parent.

        Parameters
        ----------
        node : JointNode
            The joint node to add.
        parent : str or None
            Parent joint name. Use ``None`` only for the root and only if it
            equals ``base_joint_name``.

        Raises
        ------
        ValueError
            If adding a non‑root with ``parent=None``, or if a cycle would be
            created.
        KeyError
            If the specified parent does not exist.
        """
        if parent is None:
            if node.name != self.m_root:
                raise ValueError("Only the root may be added with parent=None")
            if node.name in self.m_joint_graph:
                # Replace root's local transform
                self.m_joint_graph.nodes[node.name]["joint"] = node
                return
        else:
            if parent not in self.m_joint_graph:
                raise KeyError(f"parent joint '{parent}' does not exist")
        self._add_joint_node(node)
        if parent is not None:
            self._add_edge(parent, node.name)
        self._validate_graph()

    def _add_joint_node(self, node: JointNode) -> None:
        """Insert a joint node into the graph without wiring edges.

        Parameters
        ----------
        node : JointNode
            Node to insert. Names must be unique.

        Raises
        ------
        ValueError
            If a node of the same name already exists.
        """
        if node.name in self.m_joint_graph:
            raise ValueError(f"joint '{node.name}' already exists")
        self.m_joint_graph.add_node(node.name, joint=node)

    def _add_edge(self, parent: str, child: str) -> None:
        """Add a directed parent→child edge and validate acyclicity.

        Parameters
        ----------
        parent : str
            Parent joint name.
        child : str
            Child joint name.

        Raises
        ------
        KeyError
            If either endpoint does not exist.
        ValueError
            If adding the edge would create a cycle.
        """
        if parent not in self.m_joint_graph or child not in self.m_joint_graph:
            raise KeyError("both parent and child must exist before adding edge")
        self.m_joint_graph.add_edge(parent, child)
        if not nx.is_directed_acyclic_graph(self.m_joint_graph):
            self.m_joint_graph.remove_edge(parent, child)
            raise ValueError("adding this edge creates a cycle in the joint graph")

    def set_local_transform(self, name: str, T_local: Transmat_4x4) -> None:
        """Set a joint's 4×4 transform relative to its parent.

        Parameters
        ----------
        name : str
            Joint name to update.
        T_local : Transmat_4x4
            Homogeneous transform with last row ``[0, 0, 0, 1]``.

        Raises
        ------
        KeyError
            If the joint name does not exist.
        ValueError
            If the transform is not a valid homogeneous transform.
        """
        if name not in self.m_joint_graph:
            raise KeyError(f"unknown joint '{name}'")
        T = _as_transmat(T_local)
        if not _is_homogeneous(T):
            raise ValueError("transmat_wrt_parent must be a (4,4) homogeneous transform")
        joint: JointNode = self.m_joint_graph.nodes[name]["joint"]
        self.m_joint_graph.nodes[name]["joint"] = JointNode(
            name=joint.name, transmat_wrt_parent=T, index=joint.index, tags=set(joint.tags)
        )

    # -------
    # Accessors
    # -------
    @property
    def base_joint_name(self) -> str:
        """Return the root joint name.

        Returns
        -------
        str
            Name of the root joint.
        """
        return self.m_root

    @property
    def joint_names(self) -> List[str]:
        """Return joint names in the graph.

        Returns
        -------
        list of str
            Joint names (order is the graph's current node iteration order).
        """
        return list(self.m_joint_graph.nodes)

    def get_local_transform(self, name: str) -> Transmat_4x4:
        """Get a copy of the 4×4 transform relative to the parent for a joint.

        Parameters
        ----------
        name : str
            Joint name.

        Returns
        -------
        Transmat_4x4
            Copy of the joint's local transform (4×4).

        Raises
        ------
        KeyError
            If the joint name does not exist.
        """
        if name not in self.m_joint_graph:
            raise KeyError(f"unknown joint '{name}'")
        T = self.m_joint_graph.nodes[name]["joint"].transmat_wrt_parent
        return cast(Transmat_4x4, T.copy())

    # ---------
    # Topology
    # ---------
    def get_joint_topology(self) -> nx.DiGraph:
        """Return a copy of the joint DAG (parent→child).

        Returns
        -------
        networkx.DiGraph
            A copy of the internal joint graph (nodes: joint names; edges:
            parent→child). Modifying the returned graph does not affect the
            skeleton.
        """
        return self.m_joint_graph.copy(as_view=False)

    def parents(self) -> Dict[str, Optional[str]]:
        """Return immediate parents for all joints.

        Returns
        -------
        dict[str, str or None]
            Mapping from joint name to its parent joint name; the root maps to
            ``None``.
        """
        out: Dict[str, Optional[str]] = {}
        for n in self.m_joint_graph.nodes:
            preds = list(self.m_joint_graph.predecessors(n))
            if n == self.m_root:
                out[n] = None
            else:
                out[n] = preds[0] if preds else None
        return out

    def children(self) -> Dict[str, List[str]]:
        """Return immediate children for all joints.

        Returns
        -------
        dict[str, list[str]]
            Mapping from joint name to its list of child joint names.
        """
        return {n: list(self.m_joint_graph.successors(n)) for n in self.m_joint_graph.nodes}

    def topo_order(self) -> List[str]:
        """Return nodes in a valid topological order.

        Returns
        -------
        list[str]
            Joint names in topological order (root appears before descendants).
        """
        return list(nx.topological_sort(self.m_joint_graph))

    # -------------
    # Kinematics
    # -------------
    def get_global_transforms(self) -> Dict[str, Transmat_4x4]:
        """Compute world transforms for all joints via FK in topological order.

        Returns
        -------
        dict[str, Transmat_4x4]
            Mapping from joint name to world transform (4×4 float32 arrays).

        Raises
        ------
        ValueError
            If the graph is invalid (not a rooted DAG with single parents).
        """
        self._validate_graph()
        T_world: Dict[str, Transmat_4x4] = {}
        order = self.topo_order()
        for n in order:
            T_local = self.m_joint_graph.nodes[n]["joint"].transmat_wrt_parent
            if n == self.m_root:
                T_world[n] = cast(Transmat_4x4, T_local.astype(np.float32, copy=True))
            else:
                preds = list(self.m_joint_graph.predecessors(n))
                if len(preds) != 1:
                    raise ValueError("each joint must have exactly one parent")
                p = preds[0]
                T_world[n] = cast(Transmat_4x4, (T_world[p] @ T_local).astype(np.float32, copy=False))
        return T_world

    def get_global_transform(self, name: str) -> Transmat_4x4:
        """Compute a single joint's world transform via FK.

        Parameters
        ----------
        name : str
            Joint name to query.

        Returns
        -------
        Transmat_4x4
            The joint's world transform (4×4 float32 array).

        Raises
        ------
        KeyError
            If the joint name does not exist.
        ValueError
            If the graph is invalid.
        """
        if name not in self.m_joint_graph:
            raise KeyError(f"unknown joint '{name}'")
        return self.get_global_transforms()[name]

    # -----------
    # Validation
    # -----------
    def _validate_graph(self) -> None:
        """Validate that the joint graph is a rooted DAG and connected.

        Raises
        ------
        ValueError
            If the graph is cyclic, missing the root, has incorrect in‑degrees,
            or if some nodes are not reachable from the root.
        """
        if not nx.is_directed_acyclic_graph(self.m_joint_graph):
            raise ValueError("joint graph must be a DAG")
        if self.m_root not in self.m_joint_graph:
            raise ValueError("root joint missing from graph")
        # in-degree of root must be 0; others must be 1
        for n in self.m_joint_graph.nodes:
            indeg = self.m_joint_graph.in_degree(n)
            if n == self.m_root and indeg != 0:
                raise ValueError("root joint must have no parent")
            if n != self.m_root and indeg != 1:
                raise ValueError("non-root joints must have exactly one parent")
        # Reachability: all nodes reachable from root
        reachable = set(nx.descendants(self.m_joint_graph, self.m_root)) | {self.m_root}
        if len(reachable) != self.m_joint_graph.number_of_nodes():
            raise ValueError("all joints must be reachable from the root")

    # -------
    # Summary
    # -------
    def summary(self) -> str:
        """Return a brief multi‑line text summary suitable for logging.

        Returns
        -------
        str
            Human‑readable summary including name, root, and graph sizes.
        """
        return (
            f"GenericSkeleton(name={self.name!r}, base_joint={self.m_root!r})\n"
            f"  joints: {self.m_joint_graph.number_of_nodes()} | "
            f"edges: {self.m_joint_graph.number_of_edges()}"
        )


# -----------------
# Helper functions
# -----------------
def _eye4() -> Transmat_4x4:
    """Return a 4×4 identity transform (float32).

    Returns
    -------
    Transmat_4x4
        4×4 identity matrix with dtype float32.
    """
    return np.eye(4, dtype=np.float32)


def _as_transmat(a: Any) -> Transmat_4x4:
    """Convert input to a 4×4 float32 transform.

    Parameters
    ----------
    a : Any
        Input array‑like to be interpreted as a 4×4 transform.

    Returns
    -------
    Transmat_4x4
        The input converted to ``float32`` with shape ``(4, 4)``.

    Raises
    ------
    ValueError
        If the input does not have shape ``(4, 4)``.
    """
    arr = np.asarray(a, dtype=np.float32)
    if arr.shape != (4, 4):
        raise ValueError("transform must have shape (4,4)")
    return cast(Transmat_4x4, arr)


def _is_homogeneous(T: np.ndarray, eps: float = 1e-6) -> bool:
    """Return True if ``T`` is a valid homogeneous transform matrix.

    A homogeneous transform is defined as a 4×4 matrix whose last row is
    ``[0, 0, 0, 1]`` within a numerical tolerance.

    Parameters
    ----------
    T : np.ndarray
        Matrix to validate.
    eps : float, optional
        Tolerance used for the last row check (default ``1e-6``).

    Returns
    -------
    bool
        ``True`` if ``T`` is 4×4 and its last row equals ``[0,0,0,1]`` within
        ``eps``; otherwise ``False``.
    """
    if T.shape != (4, 4):
        return False
    tail = T[3]
    return (
        abs(float(tail[0])) < eps
        and abs(float(tail[1])) < eps
        and abs(float(tail[2])) < eps
        and abs(float(tail[3]) - 1.0) < eps
    )
