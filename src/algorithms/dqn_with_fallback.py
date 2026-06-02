"""DQN policy with explainable safety fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.algorithms.greedy_latency import GreedyLatency
from src.env.compute_nodes import ComputeNode, OffloadingAction
from src.env.task_generator import Task


class BasePolicy(Protocol):
    name: str

    def select_action(
        self,
        task: Task,
        nodes: dict[OffloadingAction, ComputeNode],
    ) -> OffloadingAction:
        """Select an offloading action."""


@dataclass(frozen=True)
class FallbackConfig:
    """Thresholds for switching from DQN to a safer rule-based decision."""

    tight_deadline_ms: float = 260.0
    npu_queue_threshold: int = 7
    deadline_risk_ratio: float = 0.9


class DQNWithFallback:
    """Wrap a DQN policy with safety checks for privacy, deadline, and queues."""

    name = "dqn_with_fallback"

    def __init__(
        self,
        dqn_policy: BasePolicy,
        config: FallbackConfig | None = None,
        fallback_policy: GreedyLatency | None = None,
    ) -> None:
        self.dqn_policy = dqn_policy
        self.config = config or FallbackConfig()
        self.fallback_policy = fallback_policy or GreedyLatency()

    def select_action(
        self,
        task: Task,
        nodes: dict[OffloadingAction, ComputeNode],
    ) -> OffloadingAction:
        """Select DQN action unless a safety fallback rule is triggered."""
        dqn_action = self.dqn_policy.select_action(task, nodes)

        if task.privacy_level > nodes[dqn_action].max_privacy_level:
            return self._privacy_fallback(task, nodes)

        if task.deadline_ms <= self.config.tight_deadline_ms and dqn_action == OffloadingAction.CLOUD:
            return self.fallback_policy.select_action(task, nodes)

        if (
            dqn_action == OffloadingAction.EDGE_NPU
            and task.npu_queue_len > self.config.npu_queue_threshold
        ):
            return self._best_non_npu_latency_action(task, nodes)

        selected_result = nodes[dqn_action].execute(task)
        if selected_result.latency_ms >= task.deadline_ms * self.config.deadline_risk_ratio:
            return self.fallback_policy.select_action(task, nodes)

        return dqn_action

    @staticmethod
    def _privacy_fallback(
        task: Task,
        nodes: dict[OffloadingAction, ComputeNode],
    ) -> OffloadingAction:
        if task.privacy_level <= nodes[OffloadingAction.EDGE_CPU].max_privacy_level:
            return OffloadingAction.EDGE_CPU
        return OffloadingAction.LOCAL_CPU

    @staticmethod
    def _best_non_npu_latency_action(
        task: Task,
        nodes: dict[OffloadingAction, ComputeNode],
    ) -> OffloadingAction:
        valid_actions = [
            action
            for action, node in nodes.items()
            if action != OffloadingAction.EDGE_NPU and task.privacy_level <= node.max_privacy_level
        ]
        if not valid_actions:
            return OffloadingAction.LOCAL_CPU
        return min(valid_actions, key=lambda action: nodes[action].execute(task).latency_ms)
