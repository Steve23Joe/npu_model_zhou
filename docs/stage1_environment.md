# Stage 1 - Environment

## Scope

Stage 1 implements the core Local, Edge CPU, Edge NPU, and Cloud offloading simulation environment. The task is a data-based decision problem, not a computer vision workload.

Implemented files:

- `src/env/task_generator.py`
- `src/env/compute_nodes.py`
- `src/env/reward.py`
- `src/env/offloading_env.py`
- `scripts/run_stage1_env_smoke_test.py`
- `tests/test_task_generator.py`
- `tests/test_compute_nodes.py`
- `tests/test_offloading_env.py`

## State Design

Each task contains:

- `task_id`
- `data_size_mb`
- `cpu_cycles`
- `deadline_ms`
- `priority`
- `privacy_level`
- `bandwidth_mbps`
- `local_queue_len`
- `edge_queue_len`
- `npu_queue_len`
- `cloud_rtt_ms`

The Gymnasium observation is a normalized `float32` vector with 10 features:

```text
[
  data_size_mb / 20,
  cpu_cycles / 10,
  deadline_ms / 500,
  priority / 3,
  privacy_level / 2,
  bandwidth_mbps / 100,
  local_queue_len / 8,
  edge_queue_len / 12,
  npu_queue_len / 10,
  cloud_rtt_ms / 120
]
```

Values are clipped to `[0, 1]`.

## Action Design

The action space is `gymnasium.spaces.Discrete(4)`:

```text
0 = Local CPU
1 = Edge CPU
2 = Edge NPU
3 = Cloud
```

The environment uses the new Gymnasium API:

```python
observation, info = env.reset()
observation, reward, terminated, truncated, info = env.step(action)
```

## Latency And Energy Formulas

`cpu_cycles` is interpreted as giga-cycles.

Compute latency:

```text
compute_ms = cpu_cycles / compute_gcycles_per_s * 1000
```

Upload latency for Edge CPU, Edge NPU, and Cloud:

```text
transfer_ms = data_size_mb * 8 / bandwidth_mbps * 1000
```

Queue delay:

```text
queue_ms = queue_length * node_queue_delay_ms
```

Cloud additionally includes:

```text
network_ms = cloud_rtt_ms
```

Total latency:

```text
latency_ms = fixed_latency_ms + compute_ms + transfer_ms + queue_ms + network_ms
```

Energy:

```text
compute_energy_j = node_power_w * compute_ms / 1000
transfer_energy_j = radio_power_w * transfer_ms / 1000
energy_j = compute_energy_j + transfer_energy_j
```

Privacy policy:

- Local CPU allows privacy levels `0`, `1`, and `2`.
- Edge CPU and Edge NPU allow privacy levels `0` and `1`.
- Cloud allows privacy level `0` only.

## Reward Design

The reward is a simple negative-cost objective with a success bonus:

```text
reward = success_reward
       - latency_weight * latency_ms / 1000
       - energy_weight * energy_j
       - deadline_miss_penalty if deadline missed
       - privacy_violation_penalty if privacy violated
```

Default values:

- `success_reward = 2.0`
- `latency_weight = 1.0`
- `energy_weight = 0.05`
- `deadline_miss_penalty = 3.0`
- `privacy_violation_penalty = 5.0`

## Run Commands

```powershell
python scripts/run_stage1_env_smoke_test.py
pytest
```

## Example Output

```text
Stage 1 environment smoke test
Steps: 100
Success rate: 0.060
Deadline miss rate: 0.940
Privacy violation rate: 0.270
Mean reward: -7.044
Mean latency ms: 2769.504
Mean energy J: 4.492
Saved: D:\npu20260602\results\stage1_env_smoke_test.json
```

Test result:

```text
7 passed in 0.13s
```

The smoke test summary is saved to `results/stage1_env_smoke_test.json`.

## Next Steps

Stage 2 should implement baseline policies that consume this environment and compare Local-only, Cloud-only, greedy latency, greedy energy, and threshold offloading decisions.
