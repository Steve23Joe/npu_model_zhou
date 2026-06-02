# Stage 5 - Benchmark

## Scope

Stage 5 benchmarks FP32 and INT8 ONNX inference on CPU and adds a queue-aware simulated NPU runtime for system-level edge inference experiments.

## Benchmark Design

The CPU benchmark uses ONNX Runtime with `CPUExecutionProvider`.

For each model:

- Run 20 warmup inferences.
- Run 300 measured inferences.
- Use random normalized state vectors with shape `[1, 10]`.
- Measure wall-clock inference latency with `time.perf_counter`.

Measured metrics:

- `mean_latency_ms`
- `p50_latency_ms`
- `p95_latency_ms`
- `throughput_inferences_per_second`
- `model_size_kb`

## FP32 vs INT8 Comparison

Command:

```powershell
python scripts/run_stage5_benchmark.py
```

Actual result:

```text
            model_name  mean_latency_ms  p50_latency_ms  p95_latency_ms  throughput_inferences_per_second  model_size_kb
offloading_policy_fp32           0.0200          0.0197          0.0211                        50002.4978        20.9844
offloading_policy_int8           0.0132          0.0131          0.0136                        75589.6064         9.8125
```

In this run, INT8 is smaller and faster:

- FP32 size: about `20.98 KB`
- INT8 size: about `9.81 KB`
- FP32 mean latency: about `0.0200 ms`
- INT8 mean latency: about `0.0132 ms`

These values are very small because the model is a compact MLP. Timing may vary by CPU and system load.

## Simulated NPU Design

`SimulatedNPURuntime` models a simple edge NPU scheduler with:

- `base_latency_ms`: base inference latency for a small request.
- `queue_limit`: maximum queue depth before overload.
- `batch_size`: requests processed per scheduling tick.
- `batch_speedup`: speedup factor from batching.
- `power_watt`: NPU power draw used for energy estimation.
- `failure_rate`: random failure probability for non-overloaded requests.

It simulates:

- NPU inference latency.
- Queue congestion.
- NPU utilization.
- Failure when the queue is overloaded.
- Random failure from `failure_rate`.

The Stage 5 script uses a Poisson request pattern with deterministic seed.

Actual simulated NPU summary:

```text
requests: 906.0000
ticks: 300.0000
mean_latency_ms: 0.5786
p95_latency_ms: 0.6588
mean_energy_j: 0.0035
success_rate: 0.9967
failure_rate_observed: 0.0033
overload_rate: 0.0000
mean_utilization: 0.2254
final_queue_depth: 0.0000
```

## Output Files

- `results/benchmark/stage5_onnx_cpu_benchmark.csv`
- `results/benchmark/stage5_onnx_cpu_benchmark.json`
- `results/benchmark/stage5_simulated_npu_summary.json`

## Test Result

```text
12 passed in 3.99s
```

## Result Interpretation

The INT8 ONNX model is about half the size of the FP32 model and shows lower mean latency in this CPU benchmark. The simulated NPU runtime shows low utilization and no overload for the selected request pattern, so it can handle this workload with high success rate.

The benchmark confirms that the exported offloading policy is small enough for edge inference experiments and that INT8 quantization is useful for the lightweight deployment path.

## Future Real Ascend NPU Deployment Direction

The next deployment step is replacing the simulated runtime with a real Ascend NPU path:

- Convert or compile the ONNX model for the Ascend toolchain.
- Run inference through the Ascend runtime APIs.
- Replace simulated latency, utilization, and failure metrics with device measurements.
- Compare CPU, simulated NPU, and real NPU results using the same benchmark schema.
