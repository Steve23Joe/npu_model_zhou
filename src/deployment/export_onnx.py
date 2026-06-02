"""Export the lightweight offloading policy model to ONNX."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.models.policy_mlp import PolicyMLP


def create_policy_model(seed: int = 42) -> PolicyMLP:
    """Create a deterministic lightweight policy model for export."""
    torch.manual_seed(seed)
    model = PolicyMLP(input_dim=10, hidden_dim=64, output_dim=4)
    model.eval()
    return model


def export_policy_to_onnx(
    output_path: str | Path,
    validation_path: str | Path | None = None,
    seed: int = 42,
) -> dict[str, object]:
    """Export `PolicyMLP` to ONNX and validate output shape with ONNX Runtime."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    model = create_policy_model(seed=seed)
    dummy_input = torch.rand(1, model.input_dim, dtype=torch.float32)
    with torch.no_grad():
        torch_output = model(dummy_input).detach().cpu().numpy()

    torch.onnx.export(
        model,
        dummy_input,
        output,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["state"],
        output_names=["action_scores"],
        dynamic_axes={
            "state": {0: "batch_size"},
            "action_scores": {0: "batch_size"},
        },
    )

    onnx_output = run_onnx_inference(output, dummy_input.detach().cpu().numpy())
    max_abs_diff = float(np.max(np.abs(torch_output - onnx_output)))
    validation = {
        "fp32_onnx_path": str(output),
        "input_shape": list(dummy_input.shape),
        "torch_output_shape": list(torch_output.shape),
        "onnx_output_shape": list(onnx_output.shape),
        "max_abs_diff": max_abs_diff,
        "shape_match": list(torch_output.shape) == list(onnx_output.shape),
    }

    if validation_path is not None:
        path = Path(validation_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(validation, indent=2), encoding="utf-8")

    return validation


def run_onnx_inference(model_path: str | Path, state: np.ndarray) -> np.ndarray:
    """Run ONNX Runtime inference for the exported policy."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    output = session.run(["action_scores"], {"state": state.astype(np.float32)})[0]
    return np.asarray(output)
