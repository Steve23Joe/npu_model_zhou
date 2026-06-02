# Stage 3 - DQN Training

## Scope

Stage 3 trains and evaluates a learning-based offloading policy using Stable-Baselines3 DQN. The training budget is intentionally small so the script runs quickly and produces a usable model artifact for later ONNX work.

## DQN State Space

The DQN observes the same normalized 10-feature vector from `OffloadingEnv`:

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

The observation space is `Box(low=0, high=1, shape=(10,), dtype=float32)`.

## DQN Action Space

The action space is `Discrete(4)`:

```text
0 = Local CPU
1 = Edge CPU
2 = Edge NPU
3 = Cloud
```

## Reward Function

The reward is inherited from Stage 1:

```text
reward = success_reward
       - latency_weight * latency_ms / 1000
       - energy_weight * energy_j
       - deadline_miss_penalty if deadline missed
       - privacy_violation_penalty if privacy violated
```

This encourages successful low-latency and low-energy execution while strongly penalizing deadline misses and privacy violations.

## Training Command

```powershell
python scripts/train_stage3_dqn.py
```

Output:

```text
Stage 3 DQN training complete
Saved model: D:\npu20260602\results\rl\stage3_dqn_model.zip
```

The model is saved to:

- `results/rl/stage3_dqn_model.zip`

## Evaluation Command

```powershell
python scripts/run_stage3_eval.py
```

Outputs:

- `results/rl/stage3_dqn_eval_summary.csv`
- `results/rl/stage3_dqn_eval_summary.json`

## Result Comparison

Evaluation used 500 fixed generated tasks.

```text
              policy  success_rate  avg_latency_ms  p95_latency_ms  avg_energy_j  deadline_miss_rate  npu_selection_rate  cloud_selection_rate
      greedy_latency         0.156         780.412        1511.997         2.902               0.844               0.136                 0.020
                 dqn         0.138         847.231        1587.696         3.070               0.862               0.114                 0.000
          local_only         0.120         863.062        1582.771         3.389               0.880               0.000                 0.000
threshold_offloading         0.098        1144.963        2882.873         3.025               0.902               0.180                 0.066
```

Test result:

```text
11 passed in 4.09s
```

## Limitations

- The DQN training budget is only 2,000 timesteps, so it is a quick functional baseline rather than a tuned RL result.
- The environment is synthetic and uses simple latency, energy, queue, and privacy formulas.
- DQN currently trains directly from random generated tasks without curriculum learning or reward tuning.
- The learned DQN is close to LocalOnly and GreedyLatency but does not yet outperform GreedyLatency.

## Next Improvement Direction

Stage 4 should export a lightweight policy network to ONNX and apply INT8 quantization. The trained DQN model is available as an RL artifact, while `src/models/policy_mlp.py` provides a compact PyTorch MLP shape suitable for export work.
