"""Export the Stage 7C windowed offloading policy to ONNX."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.windowed_offloading_policy import (
    WindowedPolicyConfig,
    WindowedTransformerOffloadingPolicy,
    count_parameters,
)


def main() -> None:
    output_dir = ROOT / "results" / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "windowed_policy_fp32.onnx"
    validation_path = output_dir / "stage7c_windowed_policy_onnx_validation.json"

    torch.manual_seed(42)
    config = WindowedPolicyConfig()
    model = WindowedTransformerOffloadingPolicy(config)
    model.eval()
    dummy_input = torch.randn(1, 64, config.feature_dim, dtype=torch.float32)

    with torch.no_grad():
        torch_output = model(dummy_input).detach().cpu().numpy()

    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["task_window"],
        output_names=["action_scores"],
        dynamic_axes={
            "task_window": {0: "batch_size", 1: "seq_len"},
            "action_scores": {0: "batch_size", 1: "seq_len"},
        },
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(
        ["action_scores"],
        {"task_window": dummy_input.detach().cpu().numpy()},
    )[0]

    validation = {
        "onnx_path": str(model_path),
        "input_shape": list(dummy_input.shape),
        "torch_output_shape": list(torch_output.shape),
        "onnx_output_shape": list(onnx_output.shape),
        "expected_output_shape": [1, 64, 4],
        "shape_match": list(onnx_output.shape) == [1, 64, 4],
        "max_abs_diff": float(np.max(np.abs(torch_output - onnx_output))),
        "parameter_count": count_parameters(model),
        "model_size_bytes": model_path.stat().st_size,
        "config": config.__dict__,
    }
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")

    print("Stage 7C windowed policy ONNX export complete")
    print(f"ONNX model: {model_path}")
    print(f"Input shape: {validation['input_shape']}")
    print(f"ONNX output shape: {validation['onnx_output_shape']}")
    print(f"Parameter count: {validation['parameter_count']}")
    print(f"Max abs diff: {validation['max_abs_diff']:.8f}")
    print(f"Validation JSON: {validation_path}")


if __name__ == "__main__":
    main()
