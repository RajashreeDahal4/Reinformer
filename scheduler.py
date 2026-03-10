from __future__ import annotations

# Created: 2026-03-10
# Last Updated: 2026-03-10
# Author: Can Bagirgan

import heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from graph import ComputationGraph, Op
from hardware import Device, Link, Node


class EventType(str, Enum):
    OP_READY = "op_ready"
    OP_START = "op_start"
    OP_FINISH = "op_finish"
    TRANSFER_START = "transfer_start"
    TRANSFER_FINISH = "transfer_finish"


@dataclass(order=True)
class Event:
    """
    Time-stamped event to be stored in a priority queue.
    """

    time: float
    event_type: EventType = field(compare=False)
    op_id: Optional[str] = field(default=None, compare=False)
    src_device_id: Optional[str] = field(default=None, compare=False)
    dst_device_id: Optional[str] = field(default=None, compare=False)
    tensor_size: float = field(default=0.0, compare=False)


@dataclass
class OpTiming:
    start_time: float
    finish_time: float


@dataclass
class DeviceStats:
    busy_time: float = 0.0


@dataclass
class LinkStats:
    busy_time: float = 0.0


@dataclass
class SimulationResult:
    status: str
    total_runtime: float
    op_timings: Dict[str, OpTiming]
    device_stats: Dict[str, DeviceStats]
    link_stats: Dict[str, LinkStats]
    error_info: Dict[str, float] = field(default_factory=dict)


class _OOMException(Exception):
    """
    Internal helper used to break out of the simulation loop when OOM is detected.
    """

    def __init__(self, result: SimulationResult) -> None:
        super().__init__("Out of memory during simulation.")
        self.result = result


class Simulator:
    """
    Event-based scheduler for a single compute node.
    """

    def __init__(self, node: Node, graph: ComputationGraph) -> None:
        self.node = node
        self.graph = graph

        self.current_time: float = 0.0
        self.event_queue: List[Event] = []

        self.device_available_at: Dict[str, float] = {
            device_id: 0.0 for device_id in self.node.devices
        }
        self.link_available_at: Dict[str, float] = {
            link_id: 0.0 for link_id in self.node.links
        }

        self.op_timings: Dict[str, OpTiming] = {}
        self.device_stats: Dict[str, DeviceStats] = {
            device_id: DeviceStats() for device_id in self.node.devices
        }
        self.link_stats: Dict[str, LinkStats] = {
            link_id: LinkStats() for link_id in self.node.links
        }

        # Remaining dependency counts (mutable copy of in-degree)
        self.remaining_deps: Dict[str, int] = self.graph.in_degree

        # Tracks persistent allocations per device for the simple memory model
        self._persistent_allocated: Dict[str, float] = {
            device_id: 0.0 for device_id in self.node.devices
        }

    def _check_and_allocate_memory(self, op: Op) -> Optional[SimulationResult]:
        """
        Checks device memory capacity before starting an op.
        Returns an OOM SimulationResult if capacity is insufficient,
        otherwise performs allocations and returns None.
        """
        device = self.node.get_device(op.device_id)
        total_required = op.memory_footprint + op.persistent_memory

        if not device.has_capacity_for(total_required):
            return SimulationResult(
                status="OOM",
                total_runtime=self.current_time,
                op_timings=self.op_timings,
                device_stats=self.device_stats,
                link_stats=self.link_stats,
                error_info={
                    "op_id": op.id,
                    "device_id": device.id,
                    "required_memory": device.current_memory_usage + total_required,
                    "capacity": device.memory_capacity,
                },
            )

        # Assume persistent memory is allocated only once per op
        if op.persistent_memory > 0.0:
            device.allocate(op.persistent_memory)
            self._persistent_allocated[device.id] += op.persistent_memory

        if op.memory_footprint > 0.0:
            device.allocate(op.memory_footprint)

        return None

    def _free_non_persistent_memory(self, op: Op) -> None:
        device = self.node.get_device(op.device_id)
        if op.memory_footprint > 0.0:
            device.free(op.memory_footprint)

    def _push_event(self, event: Event) -> None:
        heapq.heappush(self.event_queue, event)

    def _pop_event(self) -> Event:
        return heapq.heappop(self.event_queue)

    def _schedule_initial_ready_ops(self) -> None:
        for op_id, deps in self.remaining_deps.items():
            if deps == 0:
                self._push_event(Event(time=0.0, event_type=EventType.OP_READY, op_id=op_id))

    def _schedule_op_start(self, op: Op) -> None:
        # Check memory capacity before scheduling the op
        oom_result = self._check_and_allocate_memory(op)
        if oom_result is not None:
            # OOM durumunda simülasyonu erken sonlandırmak için queue'yu boşaltıyoruz.
            self.event_queue.clear()
            raise _OOMException(oom_result)

        device = self.node.get_device(op.device_id)
        start_time = max(self.current_time, self.device_available_at[device.id])

        runtime = op.runtime_on_assigned_device()
        finish_time = start_time + runtime

        self.device_stats[device.id].busy_time += runtime
        self.device_available_at[device.id] = finish_time

        self.op_timings[op.id] = OpTiming(start_time=start_time, finish_time=finish_time)

        self._push_event(
            Event(
                time=start_time,
                event_type=EventType.OP_START,
                op_id=op.id,
            )
        )
        self._push_event(
            Event(
                time=finish_time,
                event_type=EventType.OP_FINISH,
                op_id=op.id,
            )
        )

    def _schedule_transfers_for_children(self, op: Op) -> None:
        """
        Schedule transfer events for consumers that live on other devices.
        Simple approximation: treat the total output size as a single large tensor.
        """
        producer_device_id = op.device_id
        output_size = op.total_output_size()
        if output_size <= 0:
            return

        for child_id in self.graph.children_of(op.id):
            child_op = self.graph.ops[child_id]
            consumer_device_id = child_op.device_id
            if consumer_device_id == producer_device_id:
                continue

            link = self.node.get_link(producer_device_id, consumer_device_id)
            if link is None:
                # For now: if there is no direct link, skip the transfer.
                # In more realistic models, this should likely be treated as an error.
                continue

            # Link-level scheduling: allow overlap across different links,
            # but serialize transfers on the same link.
            start_time = max(self.current_time, link.available_at)
            transfer_time = link.latency + output_size / link.bandwidth
            finish_time = start_time + transfer_time

            self.link_stats[link.id].busy_time += transfer_time
            link.available_at = finish_time

            self.link_available_at[link.id] = finish_time

            self._push_event(
                Event(
                    time=start_time,
                    event_type=EventType.TRANSFER_START,
                    op_id=op.id,
                    src_device_id=producer_device_id,
                    dst_device_id=consumer_device_id,
                    tensor_size=output_size,
                )
            )
            self._push_event(
                Event(
                    time=finish_time,
                    event_type=EventType.TRANSFER_FINISH,
                    op_id=op.id,
                    src_device_id=producer_device_id,
                    dst_device_id=consumer_device_id,
                    tensor_size=output_size,
                )
            )

    def run(self) -> SimulationResult:
        self.current_time = 0.0
        self._schedule_initial_ready_ops()

        try:
            while self.event_queue:
                event = self._pop_event()
                self.current_time = event.time

                if event.event_type == EventType.OP_READY and event.op_id is not None:
                    op = self.graph.ops[event.op_id]
                    self._schedule_op_start(op)

                elif event.event_type == EventType.OP_FINISH and event.op_id is not None:
                    op = self.graph.ops[event.op_id]

                    # Free non-persistent memory
                    self._free_non_persistent_memory(op)

                    self._schedule_transfers_for_children(op)

                    # Decrease remaining dependency count for children
                    for child_id in self.graph.children_of(op.id):
                        self.remaining_deps[child_id] -= 1
                        if self.remaining_deps[child_id] == 0:
                            self._push_event(
                                Event(
                                    time=self.current_time,
                                    event_type=EventType.OP_READY,
                                    op_id=child_id,
                                )
                            )

                # For now, OP_START, TRANSFER_START, and TRANSFER_FINISH carry no extra logic.
        except _OOMException as exc:
            return exc.result

        if len(self.op_timings) != len(self.graph.ops):
            return SimulationResult(
                status="ERROR",
                total_runtime=self.current_time,
                op_timings=self.op_timings,
                device_stats=self.device_stats,
                link_stats=self.link_stats,
                error_info={"reason": "Not all ops finished."},
            )

        total_runtime = max(t.finish_time for t in self.op_timings.values()) if self.op_timings else 0.0

        return SimulationResult(
            status="OK",
            total_runtime=total_runtime,
            op_timings=self.op_timings,
            device_stats=self.device_stats,
            link_stats=self.link_stats,
        )

