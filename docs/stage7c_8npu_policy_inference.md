# Stage 7C - 8-NPU Windowed Policy Inference

## Why The Original Model Is Too Small

The original lightweight policy model is intentionally tiny:

- Input shape: `[1, 10]`
- Output shape: `[1, 4]`
- FP32 ONNX size: about `21 KB`
- INT8 ONNX size: about `10 KB`

That model is useful for edge deployment and quantization demos, but it is too small to create visible load across 8 NPU cards during recording. CPU inference is already extremely fast, so real NPU utilization is unlikely to be visible.

## Why Use A Window-Level Offloading Policy

Stage 7C adds a larger project-relevant model instead of a synthetic matrix multiplication load. The new model predicts offloading actions for a whole scheduling window of tasks:

- Input: `[batch_size, seq_len, feature_dim]`
- Default benchmark input: `[512, 64, 24]`
- Output: `[batch_size, seq_len, 4]`

This keeps the benchmark tied to the project: each output is still an offloading action score for Local CPU, Edge CPU, Edge NPU, or Cloud.

## Model Architecture

`WindowedTransformerOffloadingPolicy` is implemented in:

- `src/models/windowed_offloading_policy.py`

Default architecture:

- Input projection: `feature_dim -> d_model`
- TransformerEncoder with `batch_first=True`
- MLP action head
- `feature_dim = 24`
- `d_model = 256`
- `nhead = 8`
- `num_layers = 4`
- `dim_feedforward = 1024`
- `num_actions = 4`
- `dropout = 0.0`

Actual exported model metadata from `results/onnx/stage7c_windowed_policy_onnx_validation.json`:

- Parameter count: `3232772`
- ONNX size bytes: `3558725`
- Input shape: `[1, 64, 24]`
- Output shape: `[1, 64, 4]`
- Max PyTorch vs ONNX absolute difference: `0.0000003874`

## ONNX Export

Command:

```powershell
python scripts/export_stage7c_windowed_policy_onnx.py
```

Output files:

- `results/onnx/windowed_policy_fp32.onnx`
- `results/onnx/stage7c_windowed_policy_onnx_validation.json`

## Run On The NPU Server

Command:

```bash
python scripts/run_stage7c_8npu_policy_inference.py \
  --num-devices 8 \
  --duration-sec 10 \
  --batch-size 512 \
  --seq-len 64 \
  --feature-dim 24 \
  --d-model 256 \
  --num-layers 4 \
  --dtype float16
```

Monitor during recording:

```bash
watch -n 0.5 npu-smi info
```

Output file:

- `results/npu/stage7c_8npu_policy_inference.json`

If fewer than 8 NPU devices are visible, the script uses the visible devices and prints a warning. If `torch_npu` or NPU devices are unavailable, the script saves an explicit unavailable status and does not fake metrics.

Local run result in this environment:

```text
status: unavailable
reason: No visible torch_npu devices; benchmark was not run.
torch_npu_error: No module named 'torch_npu'
requested_devices: 8
visible_devices: 0
parameter_count: 3232772
```

## Safe, Normal, And Heavy Settings

Safe:

```bash
python scripts/run_stage7c_8npu_policy_inference.py --num-devices 8 --duration-sec 5 --batch-size 128 --seq-len 32 --d-model 128 --num-layers 2 --dtype float16
```

Normal:

```bash
python scripts/run_stage7c_8npu_policy_inference.py --num-devices 8 --duration-sec 10 --batch-size 512 --seq-len 64 --d-model 256 --num-layers 4 --dtype float16
```

Heavy:

```bash
python scripts/run_stage7c_8npu_policy_inference.py --num-devices 8 --duration-sec 20 --batch-size 768 --seq-len 96 --d-model 384 --num-layers 6 --dim-feedforward 1536 --dtype float16
```

## Difference From A Synthetic Load Script

This benchmark runs the project model itself:

- It consumes task-window features.
- It emits per-task offloading action scores.
- It uses Transformer attention across tasks in the scheduling window.
- It measures real model inference throughput per NPU process.

It does not run a fake infinite loop or an unrelated matrix multiplication workload.

## Limitations

- This model is not yet trained; it is a deployment-scale inference model for NPU load and throughput recording.
- Actual per-NPU metrics require a server with `torch_npu` and visible NPU devices.
- The benchmark duration is finite by design and should be increased only when needed for recording.
- Large batch and sequence settings can consume significant NPU memory.
