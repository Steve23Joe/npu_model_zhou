# npu-edge-offloading

NPU-aware edge task offloading and lightweight inference optimization system.

This project focuses on a data-based offloading decision task, not computer vision. It will simulate a Local, Edge CPU, Edge NPU, and Cloud environment; evaluate rule-based and reinforcement-learning policies; export a lightweight policy network to ONNX; apply INT8 quantization; and benchmark FP32 vs INT8 inference plus CPU vs simulated NPU execution.

## Stages

1. Stage 0: Project skeleton, configs, docs, and import smoke tests.
2. Stage 1: Task generator, compute node simulator, offloading environment, and reward logic.
3. Stage 2: Rule-based baselines for local-only, cloud-only, greedy latency, greedy energy, and threshold policies.
4. Stage 3: DQN training and evaluation for offloading decisions.
5. Stage 4: Lightweight MLP policy export to ONNX and INT8 quantization.
6. Stage 5: FP32 vs INT8 benchmark and CPU vs simulated NPU benchmark.

## Expected Final Outputs

- Synthetic edge task generation.
- Local, Edge CPU, Edge NPU, and Cloud simulation.
- Rule-based baseline results.
- DQN offloading policy results.
- ONNX FP32 policy model.
- INT8 quantized policy model.
- Benchmark reports and plots under `results/`.
- Final report and resume-ready summary.

## Setup

```powershell
python -m pip install -r requirements.txt
python -m pytest
```

All experiment outputs should be saved under `results/`.
