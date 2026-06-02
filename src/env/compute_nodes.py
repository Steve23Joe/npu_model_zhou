"""Compute node simulation for Local, Edge CPU, Edge NPU, and Cloud."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from src.env.task_generator import Task


class OffloadingAction(IntEnum):
    """Discrete offloading actions used by the environment."""

    LOCAL_CPU = 0
    EDGE_CPU = 1
    EDGE_NPU = 2
    CLOUD = 3


@dataclass(frozen=True)
class ExecutionResult:
    """Result of executing a task on one compute option."""

    action: OffloadingAction
    latency_ms: float
    energy_j: float
    success: bool
    deadline_missed: bool
    privacy_violation: bool


@dataclass(frozen=True)
class ComputeNode:
    """A simple execution target profile.

    `cpu_cycles` is interpreted as giga-cycles. Transfer time is derived from
    task data size in MB and bandwidth in Mbps.
    """

    name: str
    action: OffloadingAction
    compute_gcycles_per_s: float
    power_w: float
    queue_delay_ms: float
    max_privacy_level: int
    fixed_latency_ms: float = 0.0
    upload_required: bool = False
    include_cloud_rtt: bool = False
    radio_power_w: float = 1.2

    def execute(self, task: Task) -> ExecutionResult:
        """Estimate latency, energy, deadline status, and privacy status."""
        compute_ms = self.estimate_compute_latency_ms(task.cpu_cycles)
        transfer_ms = self.estimate_transfer_latency_ms(task)
        queue_ms = self.estimate_queue_delay_ms(task)
        network_ms = task.cloud_rtt_ms if self.include_cloud_rtt else 0.0
        latency_ms = self.fixed_latency_ms + compute_ms + transfer_ms + queue_ms + network_ms

        compute_energy_j = self.power_w * (compute_ms / 1000.0)
        transfer_energy_j = self.radio_power_w * (transfer_ms / 1000.0)
        energy_j = compute_energy_j + transfer_energy_j

        deadline_missed = latency_ms > task.deadline_ms
        privacy_violation = task.privacy_level > self.max_privacy_level
        success = not deadline_missed and not privacy_violation

        return ExecutionResult(
            action=self.action,
            latency_ms=latency_ms,
            energy_j=energy_j,
            success=success,
            deadline_missed=deadline_missed,
            privacy_violation=privacy_violation,
        )

    def estimate_compute_latency_ms(self, cpu_cycles: float) -> float:
        """Estimate compute-only latency in milliseconds."""
        return (cpu_cycles / self.compute_gcycles_per_s) * 1000.0

    def estimate_transfer_latency_ms(self, task: Task) -> float:
        """Estimate one-way upload time for offloaded tasks."""
        if not self.upload_required:
            return 0.0
        return (task.data_size_mb * 8.0 / task.bandwidth_mbps) * 1000.0

    def estimate_queue_delay_ms(self, task: Task) -> float:
        """Estimate queue delay using the target-specific queue length."""
        if self.action == OffloadingAction.LOCAL_CPU:
            queue_len = task.local_queue_len
        elif self.action == OffloadingAction.EDGE_NPU:
            queue_len = task.npu_queue_len
        elif self.action == OffloadingAction.EDGE_CPU:
            queue_len = task.edge_queue_len
        else:
            queue_len = 0
        return queue_len * self.queue_delay_ms


def create_default_nodes() -> dict[OffloadingAction, ComputeNode]:
    """Create default Local, Edge CPU, Edge NPU, and Cloud profiles."""
    return {
        OffloadingAction.LOCAL_CPU: ComputeNode(
            name="local_cpu",
            action=OffloadingAction.LOCAL_CPU,
            compute_gcycles_per_s=8.0,
            power_w=4.0,
            queue_delay_ms=3.0,
            max_privacy_level=2,
        ),
        OffloadingAction.EDGE_CPU: ComputeNode(
            name="edge_cpu",
            action=OffloadingAction.EDGE_CPU,
            compute_gcycles_per_s=42.0,
            power_w=9.0,
            queue_delay_ms=2.5,
            max_privacy_level=1,
            fixed_latency_ms=5.0,
            upload_required=True,
        ),
        OffloadingAction.EDGE_NPU: ComputeNode(
            name="edge_npu",
            action=OffloadingAction.EDGE_NPU,
            compute_gcycles_per_s=120.0,
            power_w=7.0,
            queue_delay_ms=2.0,
            max_privacy_level=1,
            fixed_latency_ms=8.0,
            upload_required=True,
        ),
        OffloadingAction.CLOUD: ComputeNode(
            name="cloud",
            action=OffloadingAction.CLOUD,
            compute_gcycles_per_s=180.0,
            power_w=22.0,
            queue_delay_ms=0.0,
            max_privacy_level=0,
            fixed_latency_ms=25.0,
            upload_required=True,
            include_cloud_rtt=True,
        ),
    }
