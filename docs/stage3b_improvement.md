# Stage 3B - Calibration, Improved DQN, and Safety Fallback

## Problem Found In Stage 3

The original Stage 3 results showed low overall success rates and the DQN did not outperform `GreedyLatency`. The environment had strict deadlines relative to transfer and compute latency, which made many tasks fail regardless of policy.

Stage 3B recalibrates the environment, improves reward shaping, increases DQN training budget, and adds a safety fallback wrapper around DQN decisions.

## Environment Calibration Changes

Task generation was adjusted in `src/env/task_generator.py` and mirrored in `configs/env_config.yaml`:

- `data_size_mb`: `0.05` to `6.0`
- `cpu_cycles`: `0.1` to `6.0`
- `deadline_ms`: `110.0` to `680.0`
- `bandwidth_mbps`: `20.0` to `160.0`
- Queue ranges were reduced to keep congestion possible but less dominant.

Compute node defaults in `src/env/compute_nodes.py` were adjusted:

- Local CPU speed increased to `8.0` Gcycles/s.
- Edge CPU speed increased to `42.0` Gcycles/s.
- Edge NPU speed increased to `120.0` Gcycles/s.
- Cloud remains fast but has higher fixed network cost and strict privacy limits.

Calibrated Stage 2 baseline results from `results/baselines/stage2_baseline_summary.json`:

| Policy | Success Rate | Avg Latency ms | P95 Latency ms | Deadline Miss Rate | NPU Rate | Cloud Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| greedy_latency | 0.666 | 302.436 | 701.036 | 0.334 | 0.390 | 0.000 |
| greedy_energy | 0.620 | 348.421 | 747.947 | 0.380 | 0.628 | 0.000 |
| threshold_offloading | 0.570 | 372.274 | 749.456 | 0.430 | 0.302 | 0.040 |
| local_only | 0.494 | 396.094 | 730.247 | 0.506 | 0.000 | 0.000 |
| cloud_only | 0.492 | 445.123 | 1055.243 | 0.508 | 0.000 | 0.358 |

This meets the calibration goal: LocalOnly is around `0.30` to `0.50`, GreedyLatency is around `0.55` to `0.75`, CloudOnly does not dominate, and Edge NPU is useful.

## Reward Changes

`src/env/reward.py` now uses task context:

- Positive reward for success.
- Extra success bonus for higher-priority tasks.
- Penalty for latency normalized by the task deadline.
- Penalty for energy.
- Strong penalty for deadline miss.
- Strong penalty for privacy violation.
- Mild penalty for choosing Edge NPU when NPU queue is high.
- Mild bonus for choosing Edge NPU on compute-heavy urgent tasks when the queue is low.

The reward remains explainable and uses only simulated task/result fields.

## DQN Training Changes

`configs/train_config.yaml` and `scripts/train_stage3_dqn.py` were updated:

- `total_timesteps`: `12000`
- `learning_rate`: `0.0005`
- `buffer_size`: `30000`
- `learning_starts`: `500`
- `batch_size`: `128`
- `gamma`: `0.97`
- `exploration_fraction`: `0.35`
- `target_update_interval`: `500`
- Evaluation callback every `2000` steps.

Training metadata is saved to:

- `results/rl/stage3b_dqn_training_metadata.json`

Actual metadata shows training completed in `11.9208` seconds and saved:

- `results/rl/stage3b_dqn_model.zip`
- `results/rl/stage3_dqn_model.zip`

## Safety Fallback Rules

`src/algorithms/dqn_with_fallback.py` implements `DQNWithFallback`.

Fallback rules:

- If DQN selects an action that violates the task privacy level, switch to Edge CPU or Local.
- If the deadline is very tight and DQN chooses Cloud, switch to `GreedyLatency`.
- If DQN chooses Edge NPU while NPU queue is above the overload threshold, avoid NPU and choose the best non-NPU latency action.
- If the selected action has high deadline-miss risk, switch to `GreedyLatency`.

## Before Vs After Result Table

The table below compares raw DQN vs DQNWithFallback using actual values from `results/rl/stage3b_eval_summary.json`.

| Policy | Success Rate | Avg Latency ms | P95 Latency ms | Deadline Miss Rate | Privacy Violation Rate | NPU Rate | Cloud Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dqn | 0.634 | 292.911 | 654.016 | 0.320 | 0.050 | 0.332 | 0.000 |
| dqn_with_fallback | 0.652 | 306.707 | 658.767 | 0.348 | 0.000 | 0.306 | 0.000 |

Full Stage 3B evaluation from `results/rl/stage3b_eval_summary.json`:

| Policy | Success Rate | Avg Latency ms | P50 Latency ms | P95 Latency ms | Deadline Miss Rate | Privacy Violation Rate | Avg Energy J |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| greedy_latency | 0.652 | 301.693 | 272.782 | 658.767 | 0.348 | 0.000 | 0.980 |
| dqn_with_fallback | 0.652 | 306.707 | 276.594 | 658.767 | 0.348 | 0.000 | 1.019 |
| dqn | 0.634 | 292.911 | 251.919 | 654.016 | 0.320 | 0.050 | 0.930 |
| greedy_energy | 0.622 | 339.333 | 289.777 | 745.966 | 0.378 | 0.000 | 0.865 |
| threshold_offloading | 0.558 | 375.543 | 344.328 | 746.184 | 0.442 | 0.000 | 1.062 |
| local_only | 0.496 | 394.411 | 406.384 | 715.213 | 0.504 | 0.000 | 1.543 |
| cloud_only | 0.494 | 431.451 | 376.210 | 937.085 | 0.506 | 0.000 | 1.113 |

## Limitations

- DQNWithFallback improves over raw DQN, but it does not exceed GreedyLatency on success rate; it ties GreedyLatency at `0.652`.
- The fallback reduces unsafe behavior by removing DQN privacy violations, but this increases average latency.
- Training is still intentionally short for a normal Windows CPU.
- The environment remains synthetic and should not be interpreted as real hardware performance.

## Run Commands

```powershell
python scripts/run_stage2_baselines.py
python scripts/train_stage3_dqn.py
python scripts/run_stage3_eval.py
python scripts/run_stage3b_eval.py
pytest
```

## Next Step

The next useful improvement is to tune DQN against a validation set and compare policies on multiple random seeds. That would show whether the GreedyLatency tie is robust or just a single-seed result.
