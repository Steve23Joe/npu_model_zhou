"""Gymnasium environment for NPU-aware task offloading."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.env.compute_nodes import (
    ComputeNode,
    ExecutionResult,
    OffloadingAction,
    create_default_nodes,
)
from src.env.reward import RewardConfig, compute_reward
from src.env.task_generator import Task, TaskGenerator


class OffloadingEnv(gym.Env[np.ndarray, int]):
    """Task offloading environment with four discrete execution options."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: int = 42,
        max_steps: int = 200,
        task_generator: TaskGenerator | None = None,
        nodes: dict[OffloadingAction, ComputeNode] | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.task_generator = task_generator or TaskGenerator(seed=seed)
        self.nodes = nodes or create_default_nodes()
        self.reward_config = reward_config or RewardConfig()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        self.current_task: Task | None = None
        self.steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and return the first normalized observation."""
        super().reset(seed=seed)
        _ = options
        if seed is not None:
            self.task_generator.reset(seed)
        self.steps = 0
        self.current_task = self.task_generator.sample()
        return self._observation(self.current_task), {"task_id": self.current_task.task_id}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute the selected action and advance to the next task."""
        if self.current_task is None:
            raise RuntimeError("reset must be called before step")
        if not self.action_space.contains(action):
            raise ValueError("action must be in {0, 1, 2, 3}")

        offloading_action = OffloadingAction(int(action))
        result = self.nodes[offloading_action].execute(self.current_task)
        reward = compute_reward(result, self.current_task, self.reward_config)
        info = self._info(self.current_task, result)

        self.steps += 1
        terminated = False
        truncated = self.steps >= self.max_steps
        self.current_task = self.task_generator.sample()
        observation = self._observation(self.current_task)
        return observation, reward, terminated, truncated, info

    def _observation(self, task: Task) -> np.ndarray:
        """Normalize task and system features to roughly [0, 1]."""
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

    @staticmethod
    def _info(task: Task, result: ExecutionResult) -> dict[str, Any]:
        return {
            "task_id": task.task_id,
            "action": result.action.name,
            "latency_ms": result.latency_ms,
            "energy_j": result.energy_j,
            "success": result.success,
            "deadline_missed": result.deadline_missed,
            "privacy_violation": result.privacy_violation,
        }
