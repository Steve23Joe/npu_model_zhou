"""Run real multi-NPU inference for the Stage 7C windowed policy."""

from __future__ import annotations

import argparse
import json
from multiprocessing import Queue
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.windowed_offloading_policy import (
    WindowedPolicyConfig,
    WindowedTransformerOffloadingPolicy,
    count_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-devices", type=int, default=8)
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--warmup-iters", type=int, default=10)
    return parser.parse_args()


def worker(rank: int, args: argparse.Namespace, result_queue: Queue) -> None:
    """Run inference on one NPU and report measured throughput."""
    import torch_npu  # noqa: F401

    device_name = f"npu:{rank}"
    torch.npu.set_device(device_name)
    device = torch.device(device_name)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    config = WindowedPolicyConfig(
        feature_dim=args.feature_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
    )
    model = WindowedTransformerOffloadingPolicy(config).to(device=device, dtype=dtype)
    model.eval()
    task_window = torch.randn(
        args.batch_size,
        args.seq_len,
        args.feature_dim,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        for _ in range(args.warmup_iters):
            _ = model(task_window)
        torch.npu.synchronize()

        started = time.perf_counter()
        deadline = started + args.duration_sec
        iterations = 0
        while time.perf_counter() < deadline:
            _ = model(task_window)
            iterations += 1
        torch.npu.synchronize()
        elapsed_sec = time.perf_counter() - started

    windows = iterations * args.batch_size
    tasks = windows * args.seq_len
    result_queue.put(
        {
            "rank": rank,
            "device": device_name,
            "iterations": iterations,
            "elapsed_sec": elapsed_sec,
            "windows_per_sec": windows / elapsed_sec if elapsed_sec > 0 else 0.0,
            "tasks_per_sec": tasks / elapsed_sec if elapsed_sec > 0 else 0.0,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "feature_dim": args.feature_dim,
            "dtype": args.dtype,
        }
    )


def get_visible_npu_count() -> tuple[int, str | None]:
    """Return visible NPU count, or an error message when torch_npu is unavailable."""
    try:
        import torch_npu  # noqa: F401

        return int(torch.npu.device_count()), None
    except Exception as exc:  # pragma: no cover - depends on NPU host.
        return 0, str(exc)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_path = ROOT / "results" / "npu" / "stage7c_8npu_policy_inference.json"
    print("Monitor NPU load while running:")
    print("watch -n 0.5 npu-smi info")

    visible_devices, error = get_visible_npu_count()
    config = WindowedPolicyConfig(
        feature_dim=args.feature_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
    )
    parameter_count = count_parameters(WindowedTransformerOffloadingPolicy(config))

    if visible_devices == 0:
        payload = {
            "status": "unavailable",
            "reason": "No visible torch_npu devices; benchmark was not run.",
            "torch_npu_error": error,
            "requested_devices": args.num_devices,
            "visible_devices": visible_devices,
            "parameter_count": parameter_count,
            "config": vars(args),
            "per_device": [],
        }
        write_json(output_path, payload)
        print("WARNING: no visible NPU devices. No inference metrics were generated.")
        print(f"Saved status JSON: {output_path}")
        return

    used_devices = min(args.num_devices, visible_devices)
    if visible_devices < args.num_devices:
        print(
            f"WARNING: requested {args.num_devices} devices, "
            f"but only {visible_devices} are visible. Using {used_devices}."
        )

    result_queue: Queue = mp.Queue()
    mp.spawn(worker, args=(args, result_queue), nprocs=used_devices, join=True)
    per_device = [result_queue.get() for _ in range(used_devices)]
    per_device.sort(key=lambda item: int(item["rank"]))
    total_tasks_per_sec = sum(float(item["tasks_per_sec"]) for item in per_device)
    total_windows_per_sec = sum(float(item["windows_per_sec"]) for item in per_device)
    payload = {
        "status": "completed",
        "requested_devices": args.num_devices,
        "visible_devices": visible_devices,
        "used_devices": used_devices,
        "parameter_count": parameter_count,
        "config": vars(args),
        "total_windows_per_sec": total_windows_per_sec,
        "total_tasks_per_sec": total_tasks_per_sec,
        "per_device": per_device,
    }
    write_json(output_path, payload)

    print("Stage 7C 8-NPU policy inference benchmark")
    print(f"Parameter count: {parameter_count}")
    for item in per_device:
        print(
            f"{item['device']}: iterations={item['iterations']} "
            f"windows/s={item['windows_per_sec']:.2f} "
            f"tasks/s={item['tasks_per_sec']:.2f}"
        )
    print(f"Total windows/s: {total_windows_per_sec:.2f}")
    print(f"Total tasks/s: {total_tasks_per_sec:.2f}")
    print(f"Saved JSON: {output_path}")


if __name__ == "__main__":
    main()
