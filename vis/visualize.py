from __future__ import annotations

"""
Visualize a BERT computation graph JSON (from bert_onnx_pipeline.py / measure_bert_runtimes.py)
as a Graphviz DOT file, optionally exported to PDF/SVG.

Example:
  python visualize.py --graph graph_bert_optimized.json --dot bert_layers.dot
  dot -Tpdf bert_layers.dot -o bert_layers.pdf

You can also focus on a single encoder layer:
  python visualize.py --graph graph_bert_optimized.json --dot layer0.dot --focus-layer 0
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


LAYER_RE = re.compile(r"encoder/layer\.(\d+)/")


def get_layer_id(op_name: str) -> str:
    m = LAYER_RE.search(op_name)
    if m:
        return f"layer_{m.group(1)}"
    if "embeddings" in op_name:
        return "embeddings"
    return "other"


def short_label(op: Dict[str, Any]) -> str:
    target = op.get("target") or ""
    name = op.get("name") or ""
    if target:
        return target
    if name:
        return name.split("/")[-1]
    return op.get("id", "")


def op_color(op: Dict[str, Any]) -> str:
    """Rough color coding by op type to improve readability."""
    target = (op.get("target") or "").lower()
    name = (op.get("name") or "").lower()
    t = target or name
    if "gemm" in t or "matmul" in t:
        return "#ffcccc"  # light red
    if "layernorm" in t or "normalization" in t:
        return "#cce5ff"  # light blue
    if "softmax" in t:
        return "#e0ccff"  # light purple
    if "add" in t or "residual" in t:
        return "#d4edda"  # light green
    if "reshape" in t or "transpose" in t or "squeeze" in t or "unsqueeze" in t:
        return "#f8f9fa"  # near white
    return "#ffffff"  # default white


def load_ops(graph_path: Path) -> List[Dict[str, Any]]:
    with graph_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    ops = data.get("ops", [])
    if not isinstance(ops, list):
        raise SystemExit("Invalid graph JSON: 'ops' must be a list.")
    return ops


def write_dot(
    ops: List[Dict[str, Any]],
    out_path: Path,
    focus_layer: int | None = None,
) -> None:
    # Optional layer filtering
    if focus_layer is not None:
        pattern = f"encoder/layer.{focus_layer}/"
        filtered_ops: List[Dict[str, Any]] = []
        ids_kept = set()
        for op in ops:
            name = op.get("name") or ""
            if pattern in name:
                filtered_ops.append(op)
                ids_kept.add(op["id"])
        # Keep dependencies among kept ops only
        for op in filtered_ops:
            deps = [d for d in op.get("dependencies", []) if d in ids_kept]
            op["dependencies"] = deps
        ops = filtered_ops

    # Group by layer id
    layers: Dict[str, List[Dict[str, Any]]] = {}
    for op in ops:
        layer_id = get_layer_id(op.get("name", ""))
        layers.setdefault(layer_id, []).append(op)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("digraph G {\n")
        f.write("  rankdir=LR;\n")
        f.write('  node [shape=box, style=filled, fontsize=9];\n')

        # Subgraphs per layer for readability
        for layer_id, layer_ops in sorted(layers.items()):
            f.write(f'  subgraph cluster_{layer_id} {{\n')
            f.write(f'    label="{layer_id}";\n')
            f.write('    style=filled;\n')
            f.write('    color="#eeeeee";\n')
            for op in layer_ops:
                op_id = op["id"]
                label = short_label(op).replace('"', "'")
                color = op_color(op)
                f.write(f'    "{op_id}" [label="{label}", fillcolor="{color}"];\n')
            f.write("  }\n")

        # Edges
        for op in ops:
            for dep in op.get("dependencies", []):
                f.write(f'  "{dep}" -> "{op["id"]}";\n')

        f.write("}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export BERT computation graph JSON to Graphviz DOT."
    )
    p.add_argument(
        "--graph",
        default="graph_bert_optimized.json",
        help="Input graph JSON path (default: graph_bert_optimized.json)",
    )
    p.add_argument(
        "--dot",
        default="bert_layers.dot",
        help="Output DOT file path (default: bert_layers.dot)",
    )
    p.add_argument(
        "--focus-layer",
        type=int,
        default=None,
        help="If set, only visualize encoder/layer.<N>/ ops (e.g. 0, 1, ...).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    graph_path = Path(args.graph)
    if not graph_path.is_file():
        raise SystemExit(f"Graph file not found: {graph_path}")
    out_path = Path(args.dot)
    ops = load_ops(graph_path)
    write_dot(ops, out_path, focus_layer=args.focus_layer)
    print(f"Wrote DOT graph to {out_path}")
    print("Render with e.g.: dot -Tpdf bert_layers.dot -o bert_layers.pdf")


if __name__ == "__main__":
    main()

