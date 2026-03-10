"""
Unified pipeline: BERT → ONNX export, graph extraction from ONNX, and ORT optimization/fusion.

Modes:
  export   – Export BERT to ONNX and extract graph JSON (original 911 ops).
  extract  – Extract graph JSON from an existing ONNX file.
  optimize – Save ORT optimized ONNX and infer fusion groups (original_op_ids per fused node).

Usage:
  python bert_onnx_pipeline.py export [--onnx bert.onnx] [--graph graph_bert_onnx.json]
  python bert_onnx_pipeline.py extract --onnx bert.onnx [--output graph.json]
  python bert_onnx_pipeline.py optimize --onnx bert.onnx [--out-optimized bert_optimized.onnx] [--out-fusion fusion_groups.json]
"""
from __future__ import annotations

# Created: 2026-03-10
# Last Updated: 2026-03-10
# Author: Can Bagirgan

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import onnx
from onnx import TensorProto, shape_inference


# ---------- Shared ONNX helpers ----------

def _onnx_elem_size_bytes(elem_type: int) -> int:
    mapping = {
        TensorProto.FLOAT: 4,
        TensorProto.UINT8: 1,
        TensorProto.INT8: 1,
        TensorProto.UINT16: 2,
        TensorProto.INT16: 2,
        TensorProto.INT32: 4,
        TensorProto.INT64: 8,
        TensorProto.BOOL: 1,
        TensorProto.FLOAT16: 2,
        TensorProto.DOUBLE: 8,
        TensorProto.COMPLEX64: 8,
        TensorProto.COMPLEX128: 16,
    }
    return mapping.get(elem_type, 4)


def _onnx_shape_to_size_bytes(tensor_type: Any) -> float:
    dims = tensor_type.shape.dim
    size = 1
    for d in dims:
        if d.dim_value > 0:
            size *= int(d.dim_value)
        else:
            return 0.0
    return float(size * _onnx_elem_size_bytes(tensor_type.elem_type))


def _onnx_graph_to_ops(graph: Any) -> List[Dict[str, Any]]:
    """Build simulator-compatible op list from an ONNX graph (after shape inference)."""
    name_to_vi: Dict[str, Any] = {}
    for vi in list(graph.value_info) + list(graph.output) + list(graph.input):
        if vi.name:
            name_to_vi[vi.name] = vi

    tensor_producer: Dict[str, str] = {}
    for idx, node in enumerate(graph.node):
        op_id = f"op{idx}"
        for out_name in node.output:
            if out_name:
                tensor_producer[out_name] = op_id

    ops: List[Dict[str, Any]] = []
    for idx, node in enumerate(graph.node):
        op_id = f"op{idx}"
        dep_ids = []
        for inp in node.input:
            prod = tensor_producer.get(inp)
            if prod is not None and prod != op_id and prod not in dep_ids:
                dep_ids.append(prod)

        outputs = []
        memory_footprint = 0.0
        for out_name in node.output:
            vi = name_to_vi.get(out_name)
            if vi is None:
                continue
            tensor_type = vi.type.tensor_type
            size_bytes = _onnx_shape_to_size_bytes(tensor_type)
            memory_footprint += size_bytes
            shape = [int(d.dim_value) for d in tensor_type.shape.dim if d.dim_value > 0]
            outputs.append({
                "id": f"{op_id}_{out_name}",
                "size_bytes": size_bytes,
                "shape": shape,
            })

        ops.append({
            "id": op_id,
            "name": node.name or f"{node.op_type}_{idx}",
            "fx_op": "onnx_node",
            "target": node.op_type,
            "device_id": "gpu:0",
            "dependencies": dep_ids,
            "output_tensors": outputs,
            "memory_footprint": memory_footprint,
            "persistent_memory": 0.0,
            "runtime_on_device": {},
        })

    return ops


# ---------- Mode: export (BERT → ONNX + graph) ----------

def _export_bert_to_onnx(
    onnx_path: str,
    model_name: str = "bert-base-uncased",
    batch_size: int = 1,
    seq_len: int = 16,
) -> None:
    import torch
    import torch.nn as nn
    from transformers import BertModel, BertTokenizer

    class BertForOnnx(nn.Module):
        def __init__(self, bert: BertModel) -> None:
            super().__init__()
            self.bert = bert

        def forward(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            )
            return out[0], out[1]

    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert = BertModel.from_pretrained(model_name)
    bert.eval()
    wrapper = BertForOnnx(bert).eval()

    dummy = tokenizer(
        ["hello world"] * batch_size,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]

    with torch.no_grad():
        test_out = wrapper(input_ids, attention_mask)
    assert isinstance(test_out, tuple) and len(test_out) == 2
    print(f"Wrapper forward OK  ->  shapes: {test_out[0].shape}, {test_out[1].shape}")

    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state", "pooler_output"],
        opset_version=14,
        do_constant_folding=True,
        dynamic_axes=None,
        export_params=True,
        verbose=False,
        dynamo=False,
    )
    print(f"ONNX export done  ->  {onnx_path}")


def cmd_export(args: argparse.Namespace) -> None:
    _export_bert_to_onnx(
        args.onnx,
        model_name=args.model_name,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    model = onnx.load(args.onnx)
    inferred = shape_inference.infer_shapes(model)
    ops = _onnx_graph_to_ops(inferred.graph)
    graph = {"schema_version": 1, "ops": ops}
    out_path = Path(args.graph)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)
    print(f"Wrote {len(ops)} ops to {out_path}")


# ---------- Mode: extract (ONNX → graph JSON) ----------

def cmd_extract(args: argparse.Namespace) -> None:
    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise SystemExit(f"ONNX file not found: {onnx_path}")
    model = onnx.load(str(onnx_path))
    inferred = shape_inference.infer_shapes(model)
    ops = _onnx_graph_to_ops(inferred.graph)
    graph = {"schema_version": 1, "ops": ops}
    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)
    print(f"Wrote {len(ops)} ops to {out_path}")


# ---------- Mode: optimize (ORT fused ONNX + fusion groups) ----------

def _save_optimized_onnx(onnx_path: str, out_path: str, providers: List[str]) -> None:
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.optimized_model_filepath = out_path
    ort.InferenceSession(onnx_path, opts, providers=providers)
    print(f"Saved optimized model -> {out_path}")


def _build_tensor_producer_and_nodes(original_graph: Any) -> Tuple[Dict[str, str], Dict[str, Any]]:
    tensor_producer: Dict[str, str] = {}
    op_id_to_node: Dict[str, Any] = {}
    for idx, node in enumerate(original_graph.node):
        op_id = f"op{idx}"
        op_id_to_node[op_id] = node
        for out_name in node.output:
            if out_name:
                tensor_producer[out_name] = op_id
    return tensor_producer, op_id_to_node


def _collect_fusion_group(
    optimized_node: Any,
    tensor_producer: Dict[str, str],
    op_id_to_node: Dict[str, Any],
) -> Set[str]:
    opt_inputs = set(optimized_node.input)
    group: Set[str] = set()

    def add_ancestors(value_name: str) -> None:
        if value_name in opt_inputs or not value_name:
            return
        op_id = tensor_producer.get(value_name)
        if not op_id or op_id in group:
            return
        group.add(op_id)
        node = op_id_to_node.get(op_id)
        if not node:
            return
        for inp in node.input:
            if inp and inp not in opt_inputs:
                add_ancestors(inp)

    for out_name in optimized_node.output:
        add_ancestors(out_name)
    return group


def _infer_fusion_groups(original_onnx_path: str, optimized_onnx_path: str) -> List[Dict[str, Any]]:
    orig = onnx.load(original_onnx_path)
    opt = onnx.load(optimized_onnx_path)
    tensor_producer, op_id_to_node = _build_tensor_producer_and_nodes(orig.graph)
    result: List[Dict[str, Any]] = []
    for opt_idx, opt_node in enumerate(opt.graph.node):
        group = _collect_fusion_group(opt_node, tensor_producer, op_id_to_node)
        if not group:
            result.append({
                "optimized_index": opt_idx,
                "optimized_name": opt_node.name or f"{opt_node.op_type}_{opt_idx}",
                "optimized_op_type": opt_node.op_type,
                "original_op_ids": [],
                "note": "no_mapping",
            })
        else:
            result.append({
                "optimized_index": opt_idx,
                "optimized_name": opt_node.name or f"{opt_node.op_type}_{opt_idx}",
                "optimized_op_type": opt_node.op_type,
                "original_op_ids": sorted(group, key=lambda x: int(x.replace("op", ""))),
            })
    return result


def cmd_optimize(args: argparse.Namespace) -> None:
    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise SystemExit(
            f"ONNX file not found: {onnx_path}. "
            "Generate it first with: python bert_onnx_pipeline.py export --onnx bert.onnx --graph graph_bert_onnx.json"
        )
    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("onnxruntime not installed. pip install onnxruntime")

    providers = [args.provider] if args.provider in ort.get_available_providers() else ["CPUExecutionProvider"]
    _save_optimized_onnx(str(onnx_path), args.out_optimized, providers)

    groups = _infer_fusion_groups(str(onnx_path), args.out_optimized)
    fused_count = sum(1 for g in groups if len(g["original_op_ids"]) > 1)
    single_count = sum(1 for g in groups if len(g["original_op_ids"]) == 1)
    unmapped = sum(1 for g in groups if not g["original_op_ids"])

    out = {
        "schema_version": 1,
        "original_onnx": str(onnx_path),
        "optimized_onnx": args.out_optimized,
        "provider_used": providers[0],
        "fusion_groups": groups,
        "summary": {
            "optimized_node_count": len(groups),
            "groups_with_multiple_original_ops": fused_count,
            "groups_with_single_original_op": single_count,
            "unmapped_optimized_nodes": unmapped,
        },
    }
    out_path = Path(args.out_fusion)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"Summary: {len(groups)} optimized nodes, {fused_count} fused (2+ ops), {single_count} 1:1, {unmapped} unmapped.")


# ---------- CLI ----------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BERT ONNX pipeline: export, extract graph, or optimize (fusion).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # export
    p_export = sub.add_parser("export", help="Export BERT to ONNX and extract graph JSON")
    p_export.add_argument("--onnx", default="bert.onnx", help="Output ONNX path")
    p_export.add_argument("--graph", default="graph_bert_onnx.json", help="Output graph JSON path")
    p_export.add_argument("--model-name", default="bert-base-uncased", help="HuggingFace model name")
    p_export.add_argument("--batch-size", type=int, default=1)
    p_export.add_argument("--seq-len", type=int, default=16)
    p_export.set_defaults(func=cmd_export)

    # extract
    p_extract = sub.add_parser("extract", help="Extract graph JSON from an existing ONNX file")
    p_extract.add_argument("--onnx", required=True, help="ONNX model path")
    p_extract.add_argument("--output", default="graph.json", help="Output graph JSON path")
    p_extract.set_defaults(func=cmd_extract)

    # optimize
    p_optimize = sub.add_parser("optimize", help="Save ORT optimized ONNX and fusion_groups.json")
    p_optimize.add_argument("--onnx", default="bert.onnx", help="Original ONNX path")
    p_optimize.add_argument("--out-optimized", default="bert_optimized.onnx", help="Optimized ONNX path")
    p_optimize.add_argument("--out-fusion", default="fusion_groups.json", help="Fusion groups JSON path")
    p_optimize.add_argument("--provider", default="CPUExecutionProvider", help="ORT provider for optimization")
    p_optimize.set_defaults(func=cmd_optimize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
