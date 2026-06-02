"""Cloud-only baseline policy with a privacy fallback."""

from __future__ import annotations

from src.env.compute_nodes import ComputeNode, OffloadingAction
from src.env.task_generator import Task


class CloudOnly:
    """Choose Cloud unless the task is guaranteed to violate cloud privacy."""

    name = "cloud_only"

    def select_action(
        self,
        task: Task,
        nodes: dict[OffloadingAction, ComputeNode],
    ) -> OffloadingAction:
        """Return Cloud for public tasks, otherwise the nearest valid fallback."""
        cloud = nodes[OffloadingAction.CLOUD]
        if task.privacy_level > cloud.max_privacy_level:
            edge_cpu = nodes[OffloadingAction.EDGE_CPU]
            if task.privacy_level > edge_cpu.max_privacy_level:
                return OffloadingAction.LOCAL_CPU
            return OffloadingAction.EDGE_CPU
        return OffloadingAction.CLOUD


def select_action(task: Task, nodes: dict[OffloadingAction, ComputeNode]) -> int:
    """Functional wrapper for compatibility with simple scripts."""
    return int(CloudOnly().select_action(task, nodes))
