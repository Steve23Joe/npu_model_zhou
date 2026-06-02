"""Evaluate DQN against selected Stage 2 baselines."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Protocol

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.algorithms import GreedyLatency, LocalOnly, ThresholdOffloading
from src.algorithms.rl_policy import RLPolicy
from src.env.compute_nodes import ComputeNode, OffloadingAction, create_default_nodes
from src.env.task_generator import Task, TaskGenerator
from src.utils.metrics import summarize_offloading_results


class Policy(Protocol):
    name: str

    def select_action(
        self,
        task: Task,
        nodes: dict[OffloadingAction, ComputeNode],
    ) -> OffloadingAction:
        """Select one offloading action."""


def evaluate_policy(
    policy: Policy,
    tasks: list[Task],
    nodes: dict[OffloadingAction, ComputeNode],
) -> dict[str, object]:
    """Evaluate one policy on a fixed task set."""
    rows: list[dict[str, object]] = []
    for task in tasks:
        action = policy.select_action(task, nodes)
        result = nodes[action].execute(task)
        rows.append(
            {
                "policy": policy.name,
                "task_id": task.task_id,
                "action": result.action.name,
                "latency_ms": result.latency_ms,
                "energy_j": result.energy_j,
                "success": result.success,
                "deadline_missed": result.deadline_missed,
                "privacy_violation": result.privacy_violation,
            }
        )

    summary = summarize_offloading_results(rows)
    return {"policy": policy.name, **summary}


def main() -> None:
    output_dir = ROOT / "results" / "rl"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "stage3b_dqn_model.zip"
    if not model_path.exists():
        model_path = output_dir / "stage3_dqn_model.zip"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model {model_path}. Run python scripts/train_stage3_dqn.py first."
        )

    tasks = TaskGenerator(seed=3030).sample_batch(500)
    nodes = create_default_nodes()
    policies: list[Policy] = [
        LocalOnly(),
        GreedyLatency(),
        ThresholdOffloading(),
        RLPolicy(model_path, name="dqn"),
    ]

    summary_rows = [evaluate_policy(policy, tasks, nodes) for policy in policies]
    summary_df = pd.DataFrame(summary_rows).sort_values("success_rate", ascending=False)

    summary_csv = output_dir / "stage3_dqn_eval_summary.csv"
    summary_json = output_dir / "stage3_dqn_eval_summary.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps(summary_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    columns = [
        "policy",
        "success_rate",
        "avg_latency_ms",
        "p95_latency_ms",
        "avg_energy_j",
        "deadline_miss_rate",
        "npu_selection_rate",
        "cloud_selection_rate",
    ]
    print("Stage 3 DQN evaluation")
    print(summary_df[columns].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Saved CSV: {summary_csv}")
    print(f"Saved JSON: {summary_json}")


if __name__ == "__main__":
    main()
