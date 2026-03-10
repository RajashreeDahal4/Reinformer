from __future__ import annotations

# Created: 2026-03-10
# Last Updated: 2026-03-10
# Author: Can Bagirgan

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import onnx
from onnx import TensorProto, shape_inference
from transformers import BertModel, BertTokenizer


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


def _onnx_shape_to_size_bytes(tensor_type) -> float:
    dims = tensor_type.shape.dim
    size = 1
    for d in dims:
        if d.dim_value > 0:
            size *= int(d.dim_value)
        else:
            return 0.0
    elem_size = _onnx_elem_size_bytes(tensor_type.elem_type)
    return float(size * elem_size)


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
        # return_dict=False makes BertModel return a plain tuple:
        #   (last_hidden_state, pooler_output, ...)
        return out[0], out[1]


def extract_bert_graph_onnx(
    model_name: str = "bert-base-uncased",
    batch_size: int = 1,
    seq_len: int = 16,
) -> Dict[str, Any]:
    """
    Extract then parse the ONNX graph into a simulator-compatible JSON (without runtimes).
    """
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

    # Sanity check: make sure the wrapper runs without error
    with torch.no_grad():
        test_out = wrapper(input_ids, attention_mask)
    assert isinstance(test_out, tuple) and len(test_out) == 2
    print(f"Wrapper forward OK  ->  shapes: {test_out[0].shape}, {test_out[1].shape}")

    # Export using the LEGACY TorchScript-based exporter (dynamo=False).
    # This avoids the SystemError in PyTorch 2.10's new exporter.
    # The wrapper avoids HF's forward-signature issues during tracing.
    onnx_path = "bert.onnx"
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

    # Load and run shape inference
    model_onnx = onnx.load(onnx_path)
    inferred = shape_inference.infer_shapes(model_onnx)
    graph = inferred.graph

    # Build lookup: tensor_name -> ValueInfo (for shape/dtype)
    name_to_vi: Dict[str, Any] = {}
    for vi in list(graph.value_info) + list(graph.output) + list(graph.input):
        if vi.name:
            name_to_vi[vi.name] = vi

    # First pass: record which node produces which tensor
    tensor_producer: Dict[str, str] = {}
    for idx, node in enumerate(graph.node):
        op_id = f"op{idx}"
        for out_name in node.output:
            if out_name:
                tensor_producer[out_name] = op_id

    # Second pass: build op records
    ops: List[Dict[str, Any]] = []
    for idx, node in enumerate(graph.node):
        op_id = f"op{idx}"

        dep_ids: List[str] = []
        for inp in node.input:
            prod = tensor_producer.get(inp)
            if prod is not None and prod != op_id and prod not in dep_ids:
                dep_ids.append(prod)

        outputs: List[Dict[str, Any]] = []
        memory_footprint = 0.0
        for out_name in node.output:
            vi = name_to_vi.get(out_name)
            if vi is None:
                continue
            tensor_type = vi.type.tensor_type
            size_bytes = _onnx_shape_to_size_bytes(tensor_type)
            memory_footprint += size_bytes
            shape = [int(d.dim_value) for d in tensor_type.shape.dim if d.dim_value > 0]
            outputs.append(
                {
                    "id": f"{op_id}_{out_name}",
                    "size_bytes": size_bytes,
                    "shape": shape,
                }
            )

        op_record: Dict[str, Any] = {
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
        }
        ops.append(op_record)

    return {"schema_version": 1, "ops": ops}


def main() -> None:
    graph = extract_bert_graph_onnx()
    out_path = Path("graph_bert_onnx.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)
    print(f"Wrote {len(graph['ops'])} ops to {out_path}")


if __name__ == "__main__":
    main()
