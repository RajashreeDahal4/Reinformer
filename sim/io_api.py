from __future__ import annotations

# Created: 2026-03-10
# Last Updated: 2026-03-10
# Author: Can Bagirgan

from dataclasses import asdict
import json
from typing import Any, Dict, List

from graph import ComputationGraph, Op, TensorSpec
from hardware import Device, DeviceType, Link, Node
from scheduler import SimulationResult, Simulator
from pathlib import Path


def _build_node_from_config(config: Dict[str, Any]) -> Node:
    hw = config["hardware_config"]

    devices: Dict[str, Device] = {}
    for dev_cfg in hw["devices"]:
        dev = Device(
            id=dev_cfg["id"],
            type=DeviceType(dev_cfg["type"]),
            memory_capacity=float(dev_cfg["memory_capacity"]),
            compute_speed_factor=float(dev_cfg.get("compute_speed_factor", 1.0)),
        )
        devices[dev.id] = dev

    links: Dict[str, Link] = {}
    for link_cfg in hw.get("links", []):
        link = Link(
            id=link_cfg["id"],
            src_device_id=link_cfg["src_device_id"],
            dst_device_id=link_cfg["dst_device_id"],
            bandwidth=float(link_cfg["bandwidth"]),
            latency=float(link_cfg["latency"]),
        )
        links[link.id] = link

    node = Node(
        id=hw.get("id", "node0"),
        devices=devices,
        links=links,
        host_memory_capacity=float(hw.get("host_memory_capacity", 0.0)) or None,
    )
    return node


def _build_graph_from_config(config: Dict[str, Any]) -> ComputationGraph:
    gr = config["graph"]

    ops: Dict[str, Op] = {}
    for op_cfg in gr["ops"]:
        outputs: List[TensorSpec] = []
        for t in op_cfg.get("output_tensors", []):
            outputs.append(
                TensorSpec(
                    id=t["id"],
                    size_bytes=float(t["size_bytes"]),
                )
            )

        op = Op(
            id=op_cfg["id"],
            name=op_cfg.get("name", op_cfg["id"]),
            device_id=op_cfg["device_id"],
            dependencies=list(op_cfg.get("dependencies", [])),
            outputs=outputs,
            memory_footprint=float(op_cfg.get("memory_footprint", 0.0)),
            persistent_memory=float(op_cfg.get("persistent_memory", 0.0)),
            runtime_by_device={
                dev_id: float(rt)
                for dev_id, rt in op_cfg.get("runtime_on_device", {}).items()
            },
        )
        ops[op.id] = op

    return ComputationGraph(ops=ops)


def simulate(config: Dict[str, Any]) -> SimulationResult:
    node = _build_node_from_config(config)
    graph = _build_graph_from_config(config)

    simulator = Simulator(node=node, graph=graph)
    return simulator.run()


def simulate_from_files(hardware_config_path: str, graph_path: str) -> SimulationResult:
    """
    hardware_config.json -> only hardware information (id, devices, links, host_memory_capacity, ...)
    graph.json -> only graph information (ops list, etc.)
    """
    with open(hardware_config_path, "r", encoding="utf-8") as f:
        hw = json.load(f)

    with open(graph_path, "r", encoding="utf-8") as f:
        gr = json.load(f)

    config: Dict[str, Any] = {
        "schema_version": 1,
        "hardware_config": hw,
        "graph": gr,
    }

    return simulate(config)


def load_config_from_files(hardware_config_path: str, graph_path: str):
    parent_dir_path = Path(__file__).resolve().parent
    hardware_config_path = parent_dir_path / hardware_config_path
    graph_path = parent_dir_path / graph_path

    with open(hardware_config_path, "r", encoding="utf-8") as f:
        hw = json.load(f)

    with open(graph_path, "r", encoding="utf-8") as f:
        gr = json.load(f)

    config: Dict[str, Any] = {
        "schema_version": 1,
        "hardware_config": hw,
        "graph": gr,
    }
    return config


def simulation_result_to_dict(result: SimulationResult) -> Dict[str, Any]:

    return {
        "status": result.status,
        "total_runtime": result.total_runtime,
        "op_timings": {
            op_id: {"start_time": t.start_time, "finish_time": t.finish_time}
            for op_id, t in result.op_timings.items()
        },
        "device_stats": {
            dev_id: {"busy_time": stats.busy_time}
            for dev_id, stats in result.device_stats.items()
        },
        "link_stats": {
            link_id: {"busy_time": stats.busy_time}
            for link_id, stats in result.link_stats.items()
        },
        "error_info": dict(result.error_info),
    }
