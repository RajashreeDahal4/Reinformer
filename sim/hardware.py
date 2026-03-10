from __future__ import annotations

# Created: 2026-03-10
# Last Updated: 2026-03-10
# Author: Can Bagirgan

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class DeviceType(str, Enum):
    CPU = "cpu"
    GPU = "gpu"


@dataclass
class Device:
    """
    Common device interface. Contains basic fields for CPU and GPU devices.
    """

    id: str
    type: DeviceType
    memory_capacity: float  # bytes or MB (must be consistent with config)
    compute_speed_factor: float = 1.0
    current_memory_usage: float = 0.0

    # Optional metadata for future extensions
    metadata: Dict[str, float] = field(default_factory=dict)

    def has_capacity_for(self, required_memory: float) -> bool:
        return self.current_memory_usage + required_memory <= self.memory_capacity

    def allocate(self, size: float) -> None:
        self.current_memory_usage += size

    def free(self, size: float) -> None:
        self.current_memory_usage = max(0.0, self.current_memory_usage - size)


@dataclass
class Link:
    """
    Connection between two devices with bandwidth and latency information.
    """

    id: str
    src_device_id: str
    dst_device_id: str
    bandwidth: float  # e.g. GB/s
    latency: float  # seconds

    # Used to serialize transfers on this link
    available_at: float = 0.0


@dataclass
class Node:
    """
    Single compute node. Holds CPU/GPU devices and link topology.
    """

    id: str
    devices: Dict[str, Device]
    links: Dict[str, Link]
    host_memory_capacity: Optional[float] = None

    # src-dst -> link_id map, for fast lookup
    _link_index: Dict[Tuple[str, str], str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        for link_id, link in self.links.items():
            self._link_index[(link.src_device_id, link.dst_device_id)] = link_id

    def get_device(self, device_id: str) -> Device:
        return self.devices[device_id]

    def get_link(self, src_device_id: str, dst_device_id: str) -> Optional[Link]:
        link_id = self._link_index.get((src_device_id, dst_device_id))
        if link_id is None:
            return None
        return self.links[link_id]

    @property
    def cpus(self) -> List[Device]:
        return [d for d in self.devices.values() if d.type == DeviceType.CPU]

    @property
    def gpus(self) -> List[Device]:
        return [d for d in self.devices.values() if d.type == DeviceType.GPU]
