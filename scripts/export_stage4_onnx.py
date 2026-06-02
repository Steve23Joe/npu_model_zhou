"""Export the Stage 4 lightweight policy MLP to ONNX."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.deployment.export_onnx import export_policy_to_onnx


def main() -> None:
    output_dir = ROOT / "results" / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "offloading_policy_fp32.onnx"
    validation_path = output_dir / "stage4_onnx_validation.json"

    validation = export_policy_to_onnx(
        output_path=model_path,
        validation_path=validation_path,
        seed=42,
    )

    print("Stage 4 ONNX export complete")
    print(f"FP32 ONNX: {model_path}")
    print(f"Input shape: {validation['input_shape']}")
    print(f"PyTorch output shape: {validation['torch_output_shape']}")
    print(f"ONNX output shape: {validation['onnx_output_shape']}")
    print(f"Max abs diff: {validation['max_abs_diff']:.8f}")
    print(f"Validation JSON: {validation_path}")


if __name__ == "__main__":
    main()
