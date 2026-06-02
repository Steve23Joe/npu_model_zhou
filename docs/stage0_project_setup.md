# Stage 0 - Project Setup

## Project Goal

Build a simple, modular NPU-aware edge task offloading and lightweight inference optimization system. The main task is a tabular/data-based offloading decision problem, not a computer vision workload.

The final system should simulate Local, Edge CPU, Edge NPU, and Cloud execution targets; compare rule-based and DQN offloading policies; export a lightweight policy model to ONNX; apply INT8 quantization; and benchmark FP32 vs INT8 plus CPU vs simulated NPU inference.

## Directory Structure

```text
npu-edge-offloading/
  README.md
  requirements.txt
  configs/
    env_config.yaml
    train_config.yaml
    benchmark_config.yaml
  src/
    env/
    algorithms/
    models/
    deployment/
    utils/
  scripts/
  tests/
  docs/
```

## Stage Plan

- Stage 0: Create skeleton, configs, docs, placeholders, and import smoke tests.
- Stage 1: Implement task generation, compute node simulation, offloading environment, and reward logic.
- Stage 2: Implement and evaluate rule-based baselines.
- Stage 3: Train and evaluate a DQN offloading policy.
- Stage 4: Export the lightweight policy model to ONNX and quantize it to INT8.
- Stage 5: Benchmark FP32 vs INT8 inference and CPU vs simulated NPU runtime.

## Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

For Stage 0 import smoke tests, the placeholders avoid heavy runtime usage beyond basic Python imports.

## Run Tests

```powershell
python -m pytest
```

Expected result:

```text
4 passed
```

## Actual Results

Completed on 2026-06-02.

```text
4 passed in 0.20s
```

Stage 1 placeholder smoke script:

```text
{'observation': {'data_size_mb': 1.0}, 'reward': 0.0, 'done': True, 'info': {'action': 'local'}}
```
