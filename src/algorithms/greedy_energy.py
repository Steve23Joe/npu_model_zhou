"""Greedy energy baseline policy."""

from __future__ import annotations

from src.env.compute_nodes import ComputeNode, OffloadingAction
from src.env.task_generator import Task


class GreedyEnergy:
    """Choose the valid action with the lowest predicted energy."""

    name = "greedy_energy"

    def select_action(
        self,
        task: Task,
        nodes: dict[OffloadingAction, ComputeNode],
    ) -> OffloadingAction:
        """Estimate all privacy-valid actions and return the lowest-energy one."""
        valid_results = [
            node.execute(task)
            for node in nodes.values()
            if task.privacy_level <= node.max_privacy_level
        ]
        if not valid_results:
            return OffloadingAction.LOCAL_CPU
        return min(valid_results, key=lambda result: result.energy_j).action


def select_action(task: Task, nodes: dict[OffloadingAction, ComputeNode]) -> int:
    """Functional wrapper for compatibility with simple scripts."""
    return int(GreedyEnergy().select_action(task, nodes))
