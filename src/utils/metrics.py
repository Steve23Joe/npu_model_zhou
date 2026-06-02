"""Metric helpers for baseline and training summaries."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.env.compute_nodes import OffloadingAction


def mean(values: Sequence[float]) -> float:
    """Compute the mean of a non-empty sequence."""
    if not values:
        raise ValueError("values must not be empty")
    return float(sum(values) / len(values))


def summarize_offloading_results(rows: list[dict[str, object]]) -> dict[str, float]:
    """Summarize per-task offloading results for one policy."""
    if not rows:
        raise ValueError("rows must not be empty")

    latencies = np.array([float(row["latency_ms"]) for row in rows], dtype=np.float64)
    energies = np.array([float(row["energy_j"]) for row in rows], dtype=np.float64)
    successes = np.array([bool(row["success"]) for row in rows], dtype=np.float64)
    deadline_misses = np.array([bool(row["deadline_missed"]) for row in rows], dtype=np.float64)
    privacy_violations = np.array(
        [bool(row["privacy_violation"]) for row in rows],
        dtype=np.float64,
    )
    actions = [str(row["action"]) for row in rows]

    return {
        "avg_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "success_rate": float(np.mean(successes)),
        "deadline_miss_rate": float(np.mean(deadline_misses)),
        "privacy_violation_rate": float(np.mean(privacy_violations)),
        "avg_energy_j": float(np.mean(energies)),
        "npu_selection_rate": _selection_rate(actions, OffloadingAction.EDGE_NPU.name),
        "cloud_selection_rate": _selection_rate(actions, OffloadingAction.CLOUD.name),
    }


def _selection_rate(actions: list[str], action_name: str) -> float:
    return float(sum(action == action_name for action in actions) / len(actions))
