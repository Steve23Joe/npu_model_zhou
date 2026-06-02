"""Synthetic task generation for edge offloading decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Task:
    """A data-processing task and the system state observed at arrival time."""

    task_id: int
    data_size_mb: float
    cpu_cycles: float
    deadline_ms: float
    priority: int
    privacy_level: int
    bandwidth_mbps: float
    local_queue_len: int
    edge_queue_len: int
    npu_queue_len: int
    cloud_rtt_ms: float


@dataclass(frozen=True)
class TaskGeneratorConfig:
    """Ranges used by the random task generator."""

    data_size_mb_range: tuple[float, float] = (0.05, 6.0)
    cpu_cycles_range: tuple[float, float] = (0.1, 6.0)
    deadline_ms_range: tuple[float, float] = (110.0, 680.0)
    priority_range: tuple[int, int] = (1, 3)
    privacy_level_range: tuple[int, int] = (0, 2)
    bandwidth_mbps_range: tuple[float, float] = (20.0, 160.0)
    local_queue_len_range: tuple[int, int] = (0, 6)
    edge_queue_len_range: tuple[int, int] = (0, 8)
    npu_queue_len_range: tuple[int, int] = (0, 7)
    cloud_rtt_ms_range: tuple[float, float] = (35.0, 130.0)


class TaskGenerator:
    """Generate reproducible random tasks from simple uniform distributions."""

    def __init__(
        self,
        seed: int = 42,
        config: TaskGeneratorConfig | None = None,
    ) -> None:
        self.seed = seed
        self.config = config or TaskGeneratorConfig()
        self._rng = np.random.default_rng(seed)
        self._next_task_id = 0

    def reset(self, seed: int | None = None) -> None:
        """Reset the random stream and task id counter."""
        if seed is not None:
            self.seed = seed
        self._rng = np.random.default_rng(self.seed)
        self._next_task_id = 0

    def sample(self) -> Task:
        """Sample one task."""
        task = Task(
            task_id=self._next_task_id,
            data_size_mb=self._uniform_float(self.config.data_size_mb_range),
            cpu_cycles=self._uniform_float(self.config.cpu_cycles_range),
            deadline_ms=self._uniform_float(self.config.deadline_ms_range),
            priority=self._uniform_int(self.config.priority_range),
            privacy_level=self._uniform_int(self.config.privacy_level_range),
            bandwidth_mbps=self._uniform_float(self.config.bandwidth_mbps_range),
            local_queue_len=self._uniform_int(self.config.local_queue_len_range),
            edge_queue_len=self._uniform_int(self.config.edge_queue_len_range),
            npu_queue_len=self._uniform_int(self.config.npu_queue_len_range),
            cloud_rtt_ms=self._uniform_float(self.config.cloud_rtt_ms_range),
        )
        self._next_task_id += 1
        return task

    def sample_batch(self, count: int) -> list[Task]:
        """Sample a list of tasks."""
        if count < 0:
            raise ValueError("count must be non-negative")
        return [self.sample() for _ in range(count)]

    def _uniform_float(self, value_range: tuple[float, float]) -> float:
        low, high = value_range
        return float(self._rng.uniform(low, high))

    def _uniform_int(self, value_range: tuple[int, int]) -> int:
        low, high = value_range
        return int(self._rng.integers(low, high + 1))
