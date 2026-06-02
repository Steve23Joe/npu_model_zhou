"""Quantize the Stage 4 ONNX policy model to INT8."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.deployment.quantize_onnx import quantize_onnx_model


def main() -> None:
    output_dir = ROOT / "results" / "onnx"
    fp32_path = output_dir / "offloading_policy_fp32.onnx"
    int8_path = output_dir / "offloading_policy_int8.onnx"
    validation_path = output_dir / "stage4_onnx_validation.json"

    validation = quantize_onnx_model(
        input_path=fp32_path,
        output_path=int8_path,
        validation_path=validation_path,
    )

    print("Stage 4 ONNX INT8 quantization complete")
    print(f"FP32 ONNX: {fp32_path}")
    print(f"INT8 ONNX: {int8_path}")
    print(f"FP32 size bytes: {validation['fp32_size_bytes']}")
    print(f"INT8 size bytes: {validation['int8_size_bytes']}")
    print(f"Size reduction ratio: {validation['size_reduction_ratio']:.3f}")
    print(f"Validation JSON: {validation_path}")


if __name__ == "__main__":
    main()
