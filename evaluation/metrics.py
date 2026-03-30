import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Make "sim/io_api.py" importable when this file is placed at repo root
REPO_ROOT = Path(__file__).resolve().parent
SIM_DIR = REPO_ROOT / "sim"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from io_api import simulate_from_files, simulation_result_to_dict  


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = max(0, min(len(sorted_vals) - 1, math.ceil(q * len(sorted_vals)) - 1))
    return sorted_vals[idx]


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# graph metrics for bert onnx outputs
def compute_graph_metrics(graph: Dict[str, Any]) -> Dict[str, Any]:
    ops = graph.get("ops", [])
    op_by_id = {op["id"]: op for op in ops}

    num_ops = len(ops) #getting the number of operations
    num_edges = sum(len(op.get("dependencies", [])) for op in ops)

    total_output_bytes = 0.0
    total_output_tensors = 0
    cross_device_edges = 0
    cross_device_bytes = 0.0
    ops_per_device: Dict[str, int] = {}

    for op in ops:
        dev = op.get("device_id", "unknown")
        ops_per_device[dev] = ops_per_device.get(dev, 0) + 1

        outputs = op.get("output_tensors", [])
        for t in outputs:
            total_output_bytes += float(t.get("size_bytes", 0.0))
            total_output_tensors += 1

    for child in ops:
        child_dev = child.get("device_id", "unknown")
        for parent_id in child.get("dependencies", []):
            parent = op_by_id.get(parent_id)
            if parent is None:
                continue
            parent_dev = parent.get("device_id", "unknown")
            if parent_dev != child_dev:
                cross_device_edges += 1
                # counting all parent outputs as data is potentially transferred
                cross_device_bytes += sum(
                    float(t.get("size_bytes", 0.0))
                    for t in parent.get("output_tensors", [])
                )

    placement_percent = {
        dev: _safe_div(count, num_ops) * 100.0
        for dev, count in ops_per_device.items()
    }

    return {
        "num_ops": num_ops,
        "num_edges": num_edges,
        "avg_dependencies_per_op": _safe_div(num_edges, num_ops),
        "total_output_tensors": total_output_tensors,
        "total_output_bytes": total_output_bytes,
        "avg_output_bytes_per_op": _safe_div(total_output_bytes, num_ops),
        "cross_device_edges": cross_device_edges,
        "cross_device_bytes": cross_device_bytes,
        "ops_per_device": ops_per_device,
        "ops_percent_per_device": placement_percent,
    }


def compare_graphs(original_graph: Dict[str, Any], optimized_graph: Dict[str, Any]) -> Dict[str, Any]:
    a = compute_graph_metrics(original_graph)
    b = compute_graph_metrics(optimized_graph)

    return {
        "original_num_ops": a["num_ops"],
        "optimized_num_ops": b["num_ops"],
        "original_num_edges": a["num_edges"],
        "optimized_num_edges": b["num_edges"],
        "op_reduction_ratio": _safe_div(a["num_ops"] - b["num_ops"], a["num_ops"]),
        "edge_reduction_ratio": _safe_div(a["num_edges"] - b["num_edges"], a["num_edges"]),
        "output_bytes_change_ratio": _safe_div(
            a["total_output_bytes"] - b["total_output_bytes"],
            a["total_output_bytes"],
        ),
    }


#evaluation metrics based on runtimes.

def compute_runtime_coverage_metrics(
    graph: Dict[str, Any],
    cpu_device_id: str = "cpu:0",
    gpu_device_id: str = "gpu:0",
) -> Dict[str, Any]:
    ops = graph.get("ops", [])
    total_ops = len(ops)

    cpu_runtimes = []
    gpu_runtimes = []
    cpu_gpu_ratios = []

    cpu_count = 0
    gpu_count = 0
    both_count = 0

    for op in ops:
        rod = op.get("runtime_on_device", {})

        cpu_rt = rod.get(cpu_device_id)
        gpu_rt = rod.get(gpu_device_id)

        if cpu_rt is not None:
            cpu_count += 1
            cpu_runtimes.append(float(cpu_rt))

        if gpu_rt is not None:
            gpu_count += 1
            gpu_runtimes.append(float(gpu_rt))

        if cpu_rt is not None and gpu_rt is not None and float(gpu_rt) > 0:
            both_count += 1
            cpu_gpu_ratios.append(float(cpu_rt) / float(gpu_rt))

    cpu_runtimes_sorted = sorted(cpu_runtimes)
    gpu_runtimes_sorted = sorted(gpu_runtimes)
    ratios_sorted = sorted(cpu_gpu_ratios)

    return {
        "total_ops": total_ops,
        "cpu_runtime_coverage": _safe_div(cpu_count, total_ops),
        "gpu_runtime_coverage": _safe_div(gpu_count, total_ops),
        "both_runtime_coverage": _safe_div(both_count, total_ops),
        "cpu_runtime_seconds_mean": statistics.mean(cpu_runtimes) if cpu_runtimes else 0.0,
        "cpu_runtime_seconds_median": statistics.median(cpu_runtimes) if cpu_runtimes else 0.0,
        "cpu_runtime_seconds_p95": _quantile(cpu_runtimes_sorted, 0.95),
        "gpu_runtime_seconds_mean": statistics.mean(gpu_runtimes) if gpu_runtimes else 0.0,
        "gpu_runtime_seconds_median": statistics.median(gpu_runtimes) if gpu_runtimes else 0.0,
        "gpu_runtime_seconds_p95": _quantile(gpu_runtimes_sorted, 0.95),
        "cpu_over_gpu_ratio_mean": statistics.mean(cpu_gpu_ratios) if cpu_gpu_ratios else 0.0,
        "cpu_over_gpu_ratio_median": statistics.median(cpu_gpu_ratios) if cpu_gpu_ratios else 0.0,
        "cpu_over_gpu_ratio_min": min(cpu_gpu_ratios) if cpu_gpu_ratios else 0.0,
        "cpu_over_gpu_ratio_max": max(cpu_gpu_ratios) if cpu_gpu_ratios else 0.0,}