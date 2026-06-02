"""Stable-Baselines3 DQN policy wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import DQN

from src.env.compute_nodes import ComputeNode, OffloadingAction
from src.env.task_generator import Task


class RLPolicy:
    """Wrapper around a trained Stable-Baselines3 DQN model."""

    name = "dqn"

    def __init__(self, model_path: str | Path, name: str = "dqn") -> None:
        self.name = name
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"DQN model not found: {self.model_path}")
        self.model = DQN.load(str(self.model_path))

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """Predict an action from a normalized environment observation."""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)

    def select_action(
        self,
        task: Task,
        nodes: dict[OffloadingAction, ComputeNode],
    ) -> OffloadingAction:
        """Select an action for a task using the trained DQN."""
        _ = nodes
        observation = task_to_observation(task)
        return OffloadingAction(self.predict(observation))


def task_to_observation(task: Task) -> np.ndarray:
    """Convert a task to the normalized Stage 1 observation vector."""
    return np.array(
        [
            task.data_size_mb / 6.0,
            task.cpu_cycles / 6.0,
            task.deadline_ms / 680.0,
            task.priority / 3.0,
            task.privacy_level / 2.0,
            task.bandwidth_mbps / 160.0,
            task.local_queue_len / 6.0,
            task.edge_queue_len / 8.0,
            task.npu_queue_len / 7.0,
            task.cloud_rtt_ms / 130.0,
        ],
        dtype=np.float32,
    ).clip(0.0, 1.0)
