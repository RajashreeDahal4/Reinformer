"""Generate baseline simulation results using existing simulator.

Outputs baseline_results.json for CPU-only, GPU-only, and Random baselines.
"""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent
SIM_DIR     = REPO_ROOT / "sim"
sys.path.insert(0, str(SIM_DIR))

from io_api import _build_graph_from_config, _build_node_from_config, simulation_result_to_dict
from scheduler import Simulator

HW_CONFIG_PATH = SIM_DIR / "hardware_config.json"
GRAPH_PATH     = REPO_ROOT / "graph_bert_optimized.json"
OUT_PATH       = SCRIPT_DIR / "baseline_results.json"
RANDOM_PATH    = REPO_ROOT / "graph_bert_onnx_random.json"


def load_config() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "hardware_config": json.loads(HW_CONFIG_PATH.read_text(encoding="utf-8")),
        "graph": json.loads(GRAPH_PATH.read_text(encoding="utf-8")),
    }


def simulate(config: Dict[str, Any], placement: Dict[str, str]) -> Dict:
    "This function is to simulate the results for cpu only and gpu only."
    cfg = copy.deepcopy(config)
    for op in cfg["graph"]["ops"]:
        dev = placement.get(op["id"], op["device_id"])
        op["device_id"] = dev
        op.setdefault("runtime_on_device", {}).setdefault(dev, 0.0)
    result = Simulator(
        node=_build_node_from_config(cfg),
        graph=_build_graph_from_config(cfg),
    ).run()
    return simulation_result_to_dict(result)


def simulate_random_file(ref_config: Dict[str, Any]) -> Dict[str, Any]:
    """Load graph_bert_onnx_random.json, fill missing runtimes from optimized graph, simulate."""
    hw  = ref_config["hardware_config"]
    ref = {op["id"]: op.get("runtime_on_device", {}) for op in ref_config["graph"]["ops"]}

    rand_ops = json.loads(RANDOM_PATH.read_text(encoding="utf-8"))["ops"]
    for op in rand_ops:
        dev = op["device_id"]
        if dev not in op.get("runtime_on_device", {}):
            op.setdefault("runtime_on_device", {})[dev] = ref.get(op["id"], {}).get(dev, 0.0)

    cfg = {"schema_version": 1, "hardware_config": hw, "graph": {"ops": rand_ops}}
    result = Simulator(
        node=_build_node_from_config(cfg),
        graph=_build_graph_from_config(cfg),
    ).run()
    placement = {op["id"]: op["device_id"] for op in rand_ops}
    return {"placement": placement, "simulation_result": simulation_result_to_dict(result)}


def main() -> None:
    config = load_config()
    ops = config["graph"]["ops"]
    print(f"Graph: {len(ops)} ops  ({GRAPH_PATH.name})\n")

    results: Dict[str, Dict] = {}

    for label, device in [("CPU-only", "cpu:0"), ("GPU-only", "gpu:0")]:
        pl = {op["id"]: device for op in ops}
        sim = simulate(config, pl)
        results[label] = {"placement": pl, "simulation_result": sim}
        print(f"  {label}: {sim['total_runtime']*1000:.3f} ms  (status: {sim['status']})")

    # Random: load from graph_bert_onnx_random.json
    if not RANDOM_PATH.is_file():
        print(f"  [skip] Random: {RANDOM_PATH.name} not found")
    else:
        entry = simulate_random_file(config)
        results["Random"] = entry
        rt = entry["simulation_result"]["total_runtime"]
        print(f"  Random: {rt*1000:.3f} ms  (status: {entry['simulation_result']['status']})")

    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved → {OUT_PATH.name}")


if __name__ == "__main__":
    main()