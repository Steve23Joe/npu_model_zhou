"""Evaluate Stage 2 rule-based offloading baselines."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Protocol

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.algorithms import (
    CloudOnly,
    GreedyEnergy,
    GreedyLatency,
    LocalOnly,
    ThresholdOffloading,
)
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
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Evaluate one policy on a fixed task list."""
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
    summary["num_tasks"] = float(len(tasks))
    return rows, summary


def main() -> None:
    output_dir = ROOT / "results" / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = TaskGenerator(seed=2026).sample_batch(500)
    nodes = create_default_nodes()
    policies: list[Policy] = [
        LocalOnly(),
        CloudOnly(),
        GreedyLatency(),
        GreedyEnergy(),
        ThresholdOffloading(),
    ]

    summary_rows: list[dict[str, object]] = []
    all_result_rows: list[dict[str, object]] = []
    for policy in policies:
        rows, summary = evaluate_policy(policy, tasks, nodes)
        all_result_rows.extend(rows)
        summary_rows.append({"policy": policy.name, **summary})

    summary_df = pd.DataFrame(summary_rows).sort_values("success_rate", ascending=False)
    results_df = pd.DataFrame(all_result_rows)

    summary_csv = output_dir / "stage2_baseline_summary.csv"
    summary_json = output_dir / "stage2_baseline_summary.json"
    detail_csv = output_dir / "stage2_baseline_details.csv"

    summary_df.to_csv(summary_csv, index=False)
    results_df.to_csv(detail_csv, index=False)
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
        "privacy_violation_rate",
        "npu_selection_rate",
        "cloud_selection_rate",
    ]
    print("Stage 2 baseline comparison")
    print(summary_df[columns].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Saved CSV: {summary_csv}")
    print(f"Saved JSON: {summary_json}")


if __name__ == "__main__":
    main()
