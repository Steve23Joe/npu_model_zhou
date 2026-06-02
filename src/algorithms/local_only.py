"""Local-only baseline policy."""

from __future__ import annotations

from src.env.compute_nodes import ComputeNode, OffloadingAction
from src.env.task_generator import Task


class LocalOnly:
    """Always execute tasks on the local CPU."""

    name = "local_only"

    def select_action(
        self,
        task: Task,
        nodes: dict[OffloadingAction, ComputeNode],
    ) -> OffloadingAction:
        """Return the Local CPU action."""
        _ = task, nodes
        return OffloadingAction.LOCAL_CPU


def select_action(task: Task, nodes: dict[OffloadingAction, ComputeNode]) -> int:
    """Functional wrapper for compatibility with simple scripts."""
    return int(LocalOnly().select_action(task, nodes))
