from __future__ import annotations

# Created: 2026-03-10
# Last Updated: 2026-03-10
# Author: Can Bagirgan

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class TensorSpec:
    """
    Describes size/memory information for a single op output tensor.
    """

    id: str
    size_bytes: float


@dataclass
class Op:
    """
    Single operation (node) in the computation graph.
    """

    id: str
    name: str
    device_id: str

    # Dependencies: ids of ops that must finish before this op can start
    dependencies: List[str] = field(default_factory=list)

    # Output tensors and their sizes
    outputs: List[TensorSpec] = field(default_factory=list)

    # Memory model
    memory_footprint: float = 0.0
    persistent_memory: float = 0.0

    # Pre-measured runtimes per device id,
    # e.g. {"cpu:0": 0.01, "gpu:0": 0.002}
    runtime_by_device: Dict[str, float] = field(default_factory=dict)

    def runtime_on_assigned_device(self) -> float:
        if self.device_id not in self.runtime_by_device:
            raise KeyError(
                f"Runtime for device {self.device_id} not provided for op {self.id}"
            )
        return self.runtime_by_device[self.device_id]

    def total_output_size(self) -> float:
        return sum(t.size_bytes for t in self.outputs)


@dataclass
class ComputationGraph:
    """
    Computation graph, assumed to be a directed acyclic graph (DAG).
    """

    ops: Dict[str, Op]

    # Adjacency and in-degree information precomputed for fast access.
    _children: Dict[str, List[str]] = field(default_factory=dict, init=False)
    _parents: Dict[str, List[str]] = field(default_factory=dict, init=False)
    _in_degree: Dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._build_indices()

    def _build_indices(self) -> None:
        self._children = {op_id: [] for op_id in self.ops}
        self._parents = {op_id: list(op.dependencies) for op_id, op in self.ops.items()}
        self._in_degree = {
            op_id: len(op.dependencies) for op_id, op in self.ops.items()
        }

        for op_id, op in self.ops.items():
            for parent_id in op.dependencies:
                if parent_id not in self.ops:
                    raise KeyError(
                        f"Dependency {parent_id} of op {op_id} not found in graph"
                    )
                self._children[parent_id].append(op_id)

    @property
    def in_degree(self) -> Dict[str, int]:
        return dict(self._in_degree)

    def children_of(self, op_id: str) -> List[str]:
        return self._children[op_id]

    def parents_of(self, op_id: str) -> List[str]:
        return self._parents[op_id]

    def topological_order(self) -> List[str]:
        """
        Topological order checking for validation.
        """
        in_deg = dict(self._in_degree)
        ready: List[str] = [op_id for op_id, deg in in_deg.items() if deg == 0]
        order: List[str] = []

        while ready:
            current = ready.pop()
            order.append(current)
            for child in self._children[current]:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    ready.append(child)

        if len(order) != len(self.ops):
            raise ValueError("Graph contains a cycle or disconnected components.")
        return order

    def root_ops(self) -> List[str]:
        return [op_id for op_id, deg in self._in_degree.items() if deg == 0]

    def leaf_ops(self) -> List[str]:
        return [op_id for op_id, children in self._children.items() if not children]
