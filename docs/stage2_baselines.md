# Stage 2 - Baselines

## Scope

Stage 2 implements rule-based offloading baselines for the Local, Edge CPU, Edge NPU, and Cloud simulation from Stage 1. Each policy is evaluated on the same fixed set of 500 generated tasks.

## Policies

### LocalOnly

Always selects action `0 = Local CPU`. This is the simplest no-offloading baseline.

### CloudOnly

Selects action `3 = Cloud` for public tasks. If Cloud would violate privacy, it falls back to Edge CPU. If Edge CPU would also violate privacy, it falls back to Local CPU.

### GreedyLatency

Evaluates all privacy-valid actions with the compute-node simulator and selects the action with the lowest predicted latency.

### GreedyEnergy

Evaluates all privacy-valid actions with the compute-node simulator and selects the action with the lowest predicted energy.

### ThresholdOffloading

Uses simple thresholds over:

- `deadline_ms`
- `bandwidth_mbps`
- `edge_queue_len`
- `npu_queue_len`
- `privacy_level`

The policy avoids offloading highly private tasks, prefers Edge NPU for tight deadlines and compute-heavy tasks when bandwidth and queue length are acceptable, uses Edge CPU as a moderate fallback, and uses Cloud only for non-private tasks when edge queues are high.

## Metric Definitions

- `avg_latency_ms`: mean task latency.
- `p50_latency_ms`: median task latency.
- `p95_latency_ms`: 95th percentile task latency.
- `success_rate`: fraction of tasks that met deadline and privacy constraints.
- `deadline_miss_rate`: fraction of tasks exceeding deadline.
- `privacy_violation_rate`: fraction of tasks violating node privacy constraints.
- `avg_energy_j`: mean task energy estimate.
- `npu_selection_rate`: fraction of tasks assigned to Edge NPU.
- `cloud_selection_rate`: fraction of tasks assigned to Cloud.

## Run Command

```powershell
python scripts/run_stage2_baselines.py
pytest
```

Outputs:

- `results/baselines/stage2_baseline_summary.csv`
- `results/baselines/stage2_baseline_summary.json`
- `results/baselines/stage2_baseline_details.csv`

## Example Output

```text
Stage 2 baseline comparison
              policy  success_rate  avg_latency_ms  p95_latency_ms  avg_energy_j  deadline_miss_rate  privacy_violation_rate  npu_selection_rate  cloud_selection_rate
      greedy_latency         0.156         783.796        1592.195         2.842               0.844                   0.000               0.180                 0.006
       greedy_energy         0.146         985.252        2396.986         2.520               0.854                   0.000               0.436                 0.000
          local_only         0.130         866.807        1615.542         3.406               0.870                   0.000               0.000                 0.000
threshold_offloading         0.106        1090.395        2389.632         2.994               0.894                   0.000               0.214                 0.034
          cloud_only         0.070        2120.897        7159.265         3.901               0.930                   0.000               0.000                 0.358
```

Test result:

```text
11 passed in 0.15s
```

## Result Interpretation

GreedyLatency achieves the best success rate and the lowest average latency because it directly optimizes predicted execution time under privacy constraints. GreedyEnergy uses the NPU more often and lowers average energy, but it misses more deadlines. CloudOnly performs poorly in this simulation because upload latency and RTT dominate for many tasks.

All final Stage 2 policies have `privacy_violation_rate = 0.000`.

## Next Improvement Direction

Stage 3 should train a DQN policy using the Gymnasium environment. The DQN can learn tradeoffs between latency, energy, queue state, privacy, and deadline pressure instead of relying on fixed rules.
