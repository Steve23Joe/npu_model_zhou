"""Run a Stage 1 smoke test with random offloading actions."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.env.offloading_env import OffloadingEnv


def main() -> None:
    env = OffloadingEnv(seed=42, max_steps=100)
    env.action_space.seed(42)
    observation, reset_info = env.reset(seed=42)
    metrics = {
        "steps": 0,
        "successes": 0,
        "deadline_misses": 0,
        "privacy_violations": 0,
        "total_reward": 0.0,
        "total_latency_ms": 0.0,
        "total_energy_j": 0.0,
    }

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = int(env.action_space.sample())
        observation, reward, terminated, truncated, info = env.step(action)
        metrics["steps"] += 1
        metrics["successes"] += int(info["success"])
        metrics["deadline_misses"] += int(info["deadline_missed"])
        metrics["privacy_violations"] += int(info["privacy_violation"])
        metrics["total_reward"] += float(reward)
        metrics["total_latency_ms"] += float(info["latency_ms"])
        metrics["total_energy_j"] += float(info["energy_j"])

    steps = int(metrics["steps"])
    summary = {
        "initial_task_id": reset_info["task_id"],
        "steps": steps,
        "success_rate": metrics["successes"] / steps,
        "deadline_miss_rate": metrics["deadline_misses"] / steps,
        "privacy_violation_rate": metrics["privacy_violations"] / steps,
        "mean_reward": metrics["total_reward"] / steps,
        "mean_latency_ms": metrics["total_latency_ms"] / steps,
        "mean_energy_j": metrics["total_energy_j"] / steps,
        "last_observation_shape": list(observation.shape),
    }

    output_path = ROOT / "results" / "stage1_env_smoke_test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Stage 1 environment smoke test")
    print(f"Steps: {summary['steps']}")
    print(f"Success rate: {summary['success_rate']:.3f}")
    print(f"Deadline miss rate: {summary['deadline_miss_rate']:.3f}")
    print(f"Privacy violation rate: {summary['privacy_violation_rate']:.3f}")
    print(f"Mean reward: {summary['mean_reward']:.3f}")
    print(f"Mean latency ms: {summary['mean_latency_ms']:.3f}")
    print(f"Mean energy J: {summary['mean_energy_j']:.3f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
