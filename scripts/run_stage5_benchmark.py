"""Run Stage 5 ONNX CPU and simulated NPU benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.deployment.benchmark_onnx_cpu import benchmark_onnx_cpu
from src.deployment.simulated_npu_runtime import SimulatedNPURuntime


def main() -> None:
    output_dir = ROOT / "results" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_dir = ROOT / "results" / "onnx"
    fp32_path = onnx_dir / "offloading_policy_fp32.onnx"
    int8_path = onnx_dir / "offloading_policy_int8.onnx"

    benchmark_rows = [
        benchmark_onnx_cpu(fp32_path, warmup_runs=20, measured_runs=300, seed=42),
        benchmark_onnx_cpu(int8_path, warmup_runs=20, measured_runs=300, seed=42),
    ]
    benchmark_df = pd.DataFrame(benchmark_rows)

    cpu_csv = output_dir / "stage5_onnx_cpu_benchmark.csv"
    cpu_json = output_dir / "stage5_onnx_cpu_benchmark.json"
    benchmark_df.to_csv(cpu_csv, index=False)
    cpu_json.write_text(
        json.dumps(benchmark_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    rng = np.random.default_rng(42)
    request_pattern = rng.poisson(lam=3.0, size=300).astype(int).tolist()
    npu_runtime = SimulatedNPURuntime(
        base_latency_ms=0.35,
        queue_limit=16,
        batch_size=4,
        batch_speedup=2.5,
        power_watt=6.0,
        failure_rate=0.01,
        seed=42,
    )
    npu_summary = npu_runtime.run_workload(request_pattern)
    npu_json = output_dir / "stage5_simulated_npu_summary.json"
    npu_json.write_text(json.dumps(npu_summary, indent=2), encoding="utf-8")

    print("Stage 5 ONNX CPU benchmark")
    print(
        benchmark_df[
            [
                "model_name",
                "mean_latency_ms",
                "p50_latency_ms",
                "p95_latency_ms",
                "throughput_inferences_per_second",
                "model_size_kb",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("")
    print("Stage 5 simulated NPU summary")
    for key, value in npu_summary.items():
        print(f"{key}: {value:.4f}")
    print(f"Saved CPU CSV: {cpu_csv}")
    print(f"Saved CPU JSON: {cpu_json}")
    print(f"Saved simulated NPU JSON: {npu_json}")


if __name__ == "__main__":
    main()
