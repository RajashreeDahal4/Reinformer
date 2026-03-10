"""
Measure per-op runtimes for a BERT ONNX model on CPU and GPU using ONNX Runtime
profiler, then update the graph JSON with runtime_on_device for each op.

Each op gets runtime_on_device entries for every measured backend, e.g.:
  "runtime_on_device": { "gpu:0": 0.001, "cpu:0": 0.01 }

Prerequisites: ONNX and graph JSON from bert_onnx_pipeline.py (export or extract).
- For original graph: python bert_onnx_pipeline.py export --onnx bert.onnx --graph graph_bert_onnx.json
- For fused graph:     python bert_onnx_pipeline.py extract --onnx bert_optimized.onnx --output graph_bert_optimized.json

Usage:
  python measure_bert_runtimes.py [--onnx bert.onnx] [--graph graph_bert_onnx.json] [--output ...] [--cpu-only] [--gpu-only]
"""
from __future__ import annotations

# Created: 2026-03-10
# Last Updated: 2026-03-10
# Author: Can Bagirgan

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _get_available_providers() -> List[str]:
    try:
        import onnxruntime as ort
        return ort.get_available_providers()
    except Exception:
        return []


def _load_onnx_inputs_from_model(onnx_path: str) -> Dict[str, np.ndarray]:
    """Infer input names and shapes from ONNX and create dummy numpy inputs."""
    import onnx
    model = onnx.load(onnx_path)
    graph = model.graph
    initializer_names = {init.name for init in graph.initializer}
    inputs = {}
    for inp in graph.input:
        if inp.name in initializer_names:
            continue
        shape = []
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                shape.append(int(d.dim_value))
            else:
                shape.append(1)
        # float32 or int64
        elem_type = inp.type.tensor_type.elem_type
        if elem_type == 2:  # INT32
            inputs[inp.name] = np.zeros(shape, dtype=np.int32)
        elif elem_type == 7:  # INT64
            inputs[inp.name] = np.zeros(shape, dtype=np.int64)
        else:
            inputs[inp.name] = np.zeros(shape, dtype=np.float32)
    return inputs


def _run_profiled_session(
    onnx_path: str,
    providers: List[str],
    feed: Dict[str, np.ndarray],
    warmup: int,
    runs: int,
) -> str:
    """Run inference with profiling and return path to the profiling JSON file."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.enable_profiling = True
    sess = ort.InferenceSession(
        onnx_path,
        opts,
        providers=providers,
    )
    # Warmup
    for _ in range(warmup):
        sess.run(None, feed)
    # Profiled runs (last run's profile is what we get)
    for _ in range(runs):
        sess.run(None, feed)
    return sess.end_profiling()


def _extract_events_list(data: Any) -> List[Any]:
    """Get list of trace events from ORT profiler JSON (format can vary)."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "traceEvents" in data:
            return data["traceEvents"] if isinstance(data["traceEvents"], list) else []
        # Some builds put events under other keys
        for key in ("events", "trace_events", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
        # Nested: sometimes events are under a top-level key that is a list of events
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "name" in v[0]:
                return v
    return []


def _collect_events_with_dur(events: List[Any]) -> Dict[str, List[float]]:
    """Build name -> [durations in us] from events that have 'name' and 'dur'."""
    out: Dict[str, List[float]] = {}
    for ev in events:
        if not isinstance(ev, dict):
            continue
        raw_name = ev.get("name")
        dur = ev.get("dur")
        if raw_name is not None and dur is not None and dur > 0:
            name = str(raw_name)
            if name.endswith("_kernel_time"):
                name = name[: -len("_kernel_time")]
            out.setdefault(name, []).append(float(dur))
    return out


def _parse_profiler_json(prof_path: str) -> Dict[str, List[float]]:
    """
    Parse ONNX Runtime profiler output (Chrome trace-like JSON).
    Prefer events with cat=="Node" (op-level); fallback to any event with name+dur.
    Returns dict: node_name -> list of durations in microseconds.
    """
    with open(prof_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    events = _extract_events_list(data)
    node_name_to_durs: Dict[str, List[float]] = {}
    other_name_to_durs: Dict[str, List[float]] = {}
    for ev in events:
        if not isinstance(ev, dict):
            continue
        raw_name = ev.get("name")
        dur = ev.get("dur")
        if raw_name is None or dur is None or dur <= 0:
            continue
        name = str(raw_name)
        if name.endswith("_kernel_time"):
            name = name[: -len("_kernel_time")]
        d = float(dur)
        if ev.get("cat") == "Node":
            node_name_to_durs.setdefault(name, []).append(d)
        else:
            other_name_to_durs.setdefault(name, []).append(d)
    # Prefer Node timings; for names only in "other", use them so we get more coverage
    for name, durs in other_name_to_durs.items():
        if name not in node_name_to_durs:
            node_name_to_durs[name] = durs
    # If no events matched (e.g. different cat or format), take any event with name+dur
    if not node_name_to_durs and events:
        node_name_to_durs = _collect_events_with_dur(events)
    return node_name_to_durs


def _aggregate_durations(name_to_durs: Dict[str, List[float]]) -> Dict[str, float]:
    """Average durations per name (in microseconds)."""
    out: Dict[str, float] = {}
    for name, durs in name_to_durs.items():
        if durs:
            out[name] = sum(durs) / len(durs)
    return out


def _onnx_node_names(onnx_path: str) -> List[str]:
    """Return list of node names in graph order (same convention as bert_onnx_pipeline)."""
    import onnx
    model = onnx.load(onnx_path)
    return [
        node.name or f"{node.op_type}_{idx}"
        for idx, node in enumerate(model.graph.node)
    ]


def _name_to_index_times(
    name_to_avg_us: Dict[str, float],
    onnx_path: str,
) -> Dict[int, float]:
    """Map profiler name -> duration to graph op index -> duration (microseconds)."""
    names = _onnx_node_names(onnx_path)
    index_to_us: Dict[int, float] = {}
    for idx, name in enumerate(names):
        if name in name_to_avg_us:
            index_to_us[idx] = name_to_avg_us[name]
    return index_to_us


def measure_runtimes(
    onnx_path: str,
    warmup: int = 5,
    runs: int = 20,
    providers_cpu: Optional[List[str]] = None,
    providers_gpu: Optional[List[str]] = None,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Measure per-node runtimes on CPU and GPU. Each node gets timing on each backend.
    Returns (index_to_cpu_us, index_to_gpu_us): op index -> avg duration in microseconds.
    """
    available = _get_available_providers()
    if not available:
        raise RuntimeError("onnxruntime not installed. pip install onnxruntime [onnxruntime-gpu]")
    feed = _load_onnx_inputs_from_model(onnx_path)
    cpu_name_to_us: Dict[str, float] = {}
    gpu_name_to_us: Dict[str, float] = {}
    if providers_cpu is None:
        providers_cpu = ["CPUExecutionProvider"] if "CPUExecutionProvider" in available else []
    if providers_gpu is None:
        providers_gpu = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else []
    # Always run CPU first (only CPU provider) so CUDA load failures don't affect it
    if providers_cpu:
        prof_file = _run_profiled_session(onnx_path, providers_cpu, feed, warmup, runs)
        name_to_durs = _parse_profiler_json(prof_file)
        cpu_name_to_us = _aggregate_durations(name_to_durs)
        try:
            os.remove(prof_file)
        except OSError:
            pass
    # GPU: try CUDA; on failure (e.g. missing libcublasLt) skip GPU timings and continue
    if providers_gpu and "CUDAExecutionProvider" in available:
        try:
            prof_file = _run_profiled_session(onnx_path, providers_gpu, feed, warmup, runs)
            name_to_durs = _parse_profiler_json(prof_file)
            gpu_name_to_us = _aggregate_durations(name_to_durs)
            try:
                os.remove(prof_file)
            except OSError:
                pass
        except Exception as e:
            print(f"Warning: GPU session failed ({e}). Skipping GPU timings. Use --cpu-only to avoid CUDA errors.")
    index_to_cpu_us = _name_to_index_times(cpu_name_to_us, onnx_path)
    index_to_gpu_us = _name_to_index_times(gpu_name_to_us, onnx_path)
    return index_to_cpu_us, index_to_gpu_us


def update_graph_with_runtimes(
    graph_path: str,
    index_to_cpu_us: Dict[int, float],
    index_to_gpu_us: Dict[int, float],
    output_path: Optional[str] = None,
    device_id_cpu: str = "cpu:0",
    device_id_gpu: str = "gpu:0",
) -> None:
    """
    Load graph JSON and set runtime_on_device for each op by index.
    Each op gets runtime_on_device[device_id_cpu] and/or [device_id_gpu] (in seconds).
    """
    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)
    ops = graph.get("ops", [])
    for i, op in enumerate(ops):
        rod = op.setdefault("runtime_on_device", {})
        if i in index_to_cpu_us:
            rod[device_id_cpu] = index_to_cpu_us[i] / 1e6
        if i in index_to_gpu_us:
            rod[device_id_gpu] = index_to_gpu_us[i] / 1e6
    out = Path(output_path or graph_path)
    with out.open("w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)
    n_cpu = sum(1 for op in ops if device_id_cpu in op.get("runtime_on_device", {}))
    n_gpu = sum(1 for op in ops if device_id_gpu in op.get("runtime_on_device", {}))
    print(f"Updated {out}: {len(ops)} ops, {n_cpu} with {device_id_cpu}, {n_gpu} with {device_id_gpu}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure ONNX op runtimes on CPU/GPU and update graph JSON. Use with ONNX/graph from bert_onnx_pipeline.py."
    )
    parser.add_argument("--onnx", default="bert.onnx", help="Path to bert.onnx")
    parser.add_argument("--graph", default="graph_bert_onnx.json", help="Path to graph JSON")
    parser.add_argument("--output", default=None, help="Output graph path (default: overwrite --graph)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs before profiling")
    parser.add_argument("--runs", type=int, default=20, help="Profiled runs (last run used for profile)")
    parser.add_argument("--cpu-only", action="store_true", help="Only measure CPU (use when CUDA segfaults or is missing)")
    parser.add_argument("--gpu-only", action="store_true", help="Only measure GPU and merge into graph (run after --cpu-only to avoid segfault)")
    args = parser.parse_args()
    onnx_path = Path(args.onnx)
    graph_path = Path(args.graph)
    if not onnx_path.is_file():
        raise SystemExit(
            f"ONNX file not found: {onnx_path}. "
            "Generate with: python bert_onnx_pipeline.py export --onnx bert.onnx --graph graph_bert_onnx.json"
        )
    if not graph_path.is_file():
        raise SystemExit(
            f"Graph file not found: {graph_path}. "
            "Generate with bert_onnx_pipeline.py export or extract."
        )
    available = _get_available_providers()
    print(f"Available providers: {available}")
    if args.gpu_only:
        providers_cpu = []
        providers_gpu = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else []
        if not providers_gpu:
            raise SystemExit("--gpu-only requested but CUDAExecutionProvider not available.")
    elif args.cpu_only:
        providers_cpu = ["CPUExecutionProvider"] if "CPUExecutionProvider" in available else []
        providers_gpu = []
    else:
        providers_cpu = ["CPUExecutionProvider"] if "CPUExecutionProvider" in available else None
        providers_gpu = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else None
        if providers_gpu:
            print("Tip: If you get a segfault, run with --cpu-only first, then --gpu-only with --graph out.json --output out.json")
    device_id_cpu = "cpu:0"
    device_id_gpu = "gpu:0"
    index_to_cpu_us, index_to_gpu_us = measure_runtimes(
        str(onnx_path),
        warmup=args.warmup,
        runs=args.runs,
        providers_cpu=providers_cpu,
        providers_gpu=providers_gpu,
    )
    print(f"CPU: {len(index_to_cpu_us)} ops with timing (device_id={device_id_cpu}).")
    print(f"GPU: {len(index_to_gpu_us)} ops with timing (device_id={device_id_gpu}).")
    update_graph_with_runtimes(
        str(graph_path),
        index_to_cpu_us,
        index_to_gpu_us,
        output_path=args.output,
        device_id_cpu=device_id_cpu,
        device_id_gpu=device_id_gpu,
    )


if __name__ == "__main__":
    main()
