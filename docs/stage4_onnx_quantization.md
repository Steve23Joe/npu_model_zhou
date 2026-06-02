# Stage 4 - ONNX Quantization

## Scope

Stage 4 exports a lightweight PyTorch `PolicyMLP` to ONNX and applies ONNX Runtime dynamic INT8 quantization.

Stable-Baselines3 DQN models can be awkward to export directly because the learned policy is wrapped inside SB3-specific objects. For this stage, the project uses a standalone `PolicyMLP` with the same environment input size and action output size.

## Model

`PolicyMLP` consumes the normalized offloading state vector:

```text
input_dim = 10
hidden layers = 64, 64
output_dim = 4
```

The output is a vector of four action scores:

```text
0 = Local CPU
1 = Edge CPU
2 = Edge NPU
3 = Cloud
```

## Why ONNX Export Is Used

ONNX provides a portable model format that can be executed outside PyTorch with ONNX Runtime. This is useful for edge deployment because the policy can be loaded by a lightweight runtime instead of a full training framework.

## Why INT8 Quantization Is Used

INT8 quantization reduces model size and can improve inference latency on CPU or NPU-like runtimes. This project uses ONNX Runtime dynamic quantization, which is simple and does not require calibration data for this small MLP.

## Export Command

```powershell
python scripts/export_stage4_onnx.py
```

Output:

```text
Stage 4 ONNX export complete
FP32 ONNX: D:\npu20260602\results\onnx\offloading_policy_fp32.onnx
Input shape: [1, 10]
PyTorch output shape: [1, 4]
ONNX output shape: [1, 4]
Max abs diff: 0.00000003
Validation JSON: D:\npu20260602\results\onnx\stage4_onnx_validation.json
```

## Quantization Command

```powershell
python scripts/quantize_stage4_onnx.py
```

Output:

```text
Stage 4 ONNX INT8 quantization complete
FP32 ONNX: D:\npu20260602\results\onnx\offloading_policy_fp32.onnx
INT8 ONNX: D:\npu20260602\results\onnx\offloading_policy_int8.onnx
FP32 size bytes: 21488
INT8 size bytes: 10048
Size reduction ratio: 0.532
Validation JSON: D:\npu20260602\results\onnx\stage4_onnx_validation.json
```

The ONNX Runtime quantizer also prints a preprocessing recommendation warning. It is acceptable for this stage because dynamic quantization works without calibration or graph preprocessing.

## Validation Result

Saved validation file:

- `results/onnx/stage4_onnx_validation.json`

Key values:

```text
input_shape = [1, 10]
torch_output_shape = [1, 4]
onnx_output_shape = [1, 4]
shape_match = true
max_abs_diff = 0.00000003
fp32_size_bytes = 21488
int8_size_bytes = 10048
size_reduction_ratio = 0.532
```

Test result:

```text
11 passed in 3.92s
```

## Limitations

- The exported MLP is a standalone lightweight policy model, not a direct export of the Stable-Baselines3 DQN wrapper.
- The model weights are deterministic but not trained for optimal decisions in this stage.
- Dynamic INT8 quantization reduces size but may not always improve latency for very small networks.
- Stage 5 is needed to measure actual FP32 vs INT8 inference speed.

## Next Improvement Direction

Stage 5 should benchmark FP32 ONNX vs INT8 ONNX inference latency and compare CPU execution against the simulated NPU runtime.
