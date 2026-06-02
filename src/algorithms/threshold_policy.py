"""Threshold-based rule policy for offloading decisions."""

from __future__ import annotations

from dataclasses import dataclass

from src.env.compute_nodes import ComputeNode, OffloadingAction
from src.env.task_generator import Task


@dataclass(frozen=True)
class ThresholdConfig:
    """Thresholds used by the rule-based policy."""

    tight_deadline_ms: float = 120.0
    good_bandwidth_mbps: float = 35.0
    max_edge_queue_len: int = 6
    max_npu_queue_len: int = 4


class ThresholdOffloading:
    """Use simple system thresholds to choose a compute target."""

    name = "threshold_offloading"

    def __init__(self, config: ThresholdConfig | None = None) -> None:
        self.config = config or ThresholdConfig()

    def select_action(
        self,
        task: Task,
        nodes: dict[OffloadingAction, ComputeNode],
    ) -> OffloadingAction:
        """Choose Local, Edge CPU, Edge NPU, or Cloud from interpretable rules."""
        _ = nodes
        if task.privacy_level >= 2:
            return OffloadingAction.LOCAL_CPU

        if task.bandwidth_mbps < self.config.good_bandwidth_mbps:
            return OffloadingAction.LOCAL_CPU

        if task.deadline_ms <= self.config.tight_deadline_ms:
            if task.npu_queue_len <= self.config.max_npu_queue_len:
                return OffloadingAction.EDGE_NPU
            return OffloadingAction.EDGE_CPU

        if task.npu_queue_len <= self.config.max_npu_queue_len and task.cpu_cycles >= 2.0:
            return OffloadingAction.EDGE_NPU

        if task.edge_queue_len <= self.config.max_edge_queue_len:
            return OffloadingAction.EDGE_CPU

        if task.privacy_level > 0:
            return OffloadingAction.LOCAL_CPU

        return OffloadingAction.CLOUD


def select_action(task: Task, nodes: dict[OffloadingAction, ComputeNode]) -> int:
    """Functional wrapper for compatibility with simple scripts."""
    return int(ThresholdOffloading().select_action(task, nodes))
