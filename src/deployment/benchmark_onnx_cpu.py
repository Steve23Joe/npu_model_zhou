"""ONNX Runtime CPU benchmarking utilities."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np


def benchmark_onnx_cpu(
    model_path: str | Path,
    warmup_runs: int = 20,
    measured_runs: int = 200,
    input_dim: int = 10,
    seed: int = 42,
) -> dict[str, float | str]:
    """Benchmark one ONNX model with ONNX Runtime CPU execution."""
    import onnxruntime as ort

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    rng = np.random.default_rng(seed)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    for _ in range(warmup_runs):
        sample = rng.random((1, input_dim), dtype=np.float32)
        session.run([output_name], {input_name: sample})

    latencies_ms: list[float] = []
    for _ in range(measured_runs):
        sample = rng.random((1, input_dim), dtype=np.float32)
        start = time.perf_counter()
        session.run([output_name], {input_name: sample})
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)

    latency_array = np.array(latencies_ms, dtype=np.float64)
    mean_latency_ms = float(np.mean(latency_array))
    throughput = 1000.0 / mean_latency_ms if mean_latency_ms > 0 else 0.0

    return {
        "model_name": path.stem,
        "model_path": str(path),
        "mean_latency_ms": mean_latency_ms,
        "p50_latency_ms": float(np.percentile(latency_array, 50)),
        "p95_latency_ms": float(np.percentile(latency_array, 95)),
        "throughput_inferences_per_second": float(throughput),
        "model_size_kb": path.stat().st_size / 1024.0,
        "warmup_runs": float(warmup_runs),
        "measured_runs": float(measured_runs),
    }
