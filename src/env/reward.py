"""Reward helpers for task offloading decisions."""

from __future__ import annotations

from dataclasses import dataclass

from src.env.compute_nodes import ExecutionResult, OffloadingAction
from src.env.task_generator import Task


@dataclass(frozen=True)
class RewardConfig:
    """Weights used to combine success, latency, energy, and penalties."""

    success_reward: float = 2.0
    priority_success_bonus: float = 0.6
    latency_weight: float = 1.2
    energy_weight: float = 0.04
    deadline_miss_penalty: float = 4.0
    privacy_violation_penalty: float = 6.0
    npu_congestion_penalty: float = 0.4
    npu_good_fit_bonus: float = 0.5


def compute_reward(
    result: ExecutionResult,
    task: Task | None = None,
    config: RewardConfig | None = None,
) -> float:
    """Compute an explainable reward from an execution result.

    The reward is positive for successful tasks, adds priority-sensitive credit,
    and subtracts normalized latency, energy, deadline-miss, privacy, and
    avoidable NPU congestion costs.
    """
    reward_config = config or RewardConfig()
    reward = reward_config.success_reward if result.success else 0.0
    if result.success and task is not None:
        reward += reward_config.priority_success_bonus * (task.priority / 3.0)

    deadline_ms = task.deadline_ms if task is not None else 1000.0
    normalized_latency = result.latency_ms / max(deadline_ms, 1.0)
    reward -= reward_config.latency_weight * normalized_latency
    reward -= reward_config.energy_weight * result.energy_j

    if result.deadline_missed:
        reward -= reward_config.deadline_miss_penalty
    if result.privacy_violation:
        reward -= reward_config.privacy_violation_penalty
    if task is not None and result.action == OffloadingAction.EDGE_NPU:
        if task.npu_queue_len >= 6:
            reward -= reward_config.npu_congestion_penalty
        if task.cpu_cycles >= 3.0 and task.deadline_ms <= 450.0 and task.npu_queue_len <= 3:
            reward += reward_config.npu_good_fit_bonus

    return float(reward)
