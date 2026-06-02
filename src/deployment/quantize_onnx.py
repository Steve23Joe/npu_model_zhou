"""Apply ONNX Runtime INT8 dynamic quantization."""

from __future__ import annotations

import json
from pathlib import Path


def quantize_onnx_model(
    input_path: str | Path,
    output_path: str | Path,
    validation_path: str | Path | None = None,
) -> dict[str, object]:
    """Create an INT8 ONNX model using ONNX Runtime dynamic quantization."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    source = Path(input_path)
    target = Path(output_path)
    if not source.exists():
        raise FileNotFoundError(f"FP32 ONNX model not found: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(source),
        model_output=str(target),
        weight_type=QuantType.QInt8,
    )

    validation = {
        "fp32_onnx_path": str(source),
        "int8_onnx_path": str(target),
        "fp32_size_bytes": source.stat().st_size,
        "int8_size_bytes": target.stat().st_size,
        "size_reduction_ratio": 1.0 - (target.stat().st_size / source.stat().st_size),
    }

    if validation_path is not None:
        path = Path(validation_path)
        existing: dict[str, object] = {}
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
        existing.update(validation)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    return validation
