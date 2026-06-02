from src.algorithms import (
    CloudOnly,
    GreedyEnergy,
    GreedyLatency,
    LocalOnly,
    ThresholdOffloading,
)
from src.env.compute_nodes import OffloadingAction, create_default_nodes
from src.env.task_generator import Task
from src.utils.metrics import summarize_offloading_results


def make_task(
    privacy_level: int = 0,
    deadline_ms: float = 300.0,
    bandwidth_mbps: float = 80.0,
    edge_queue_len: int = 1,
    npu_queue_len: int = 1,
    cpu_cycles: float = 4.0,
) -> Task:
    return Task(
        task_id=0,
        data_size_mb=1.0,
        cpu_cycles=cpu_cycles,
        deadline_ms=deadline_ms,
        priority=1,
        privacy_level=privacy_level,
        bandwidth_mbps=bandwidth_mbps,
        local_queue_len=0,
        edge_queue_len=edge_queue_len,
        npu_queue_len=npu_queue_len,
        cloud_rtt_ms=40.0,
    )


def test_local_only_and_cloud_privacy_fallback() -> None:
    nodes = create_default_nodes()

    assert LocalOnly().select_action(make_task(), nodes) == OffloadingAction.LOCAL_CPU
    assert CloudOnly().select_action(make_task(privacy_level=0), nodes) == OffloadingAction.CLOUD
    assert CloudOnly().select_action(make_task(privacy_level=1), nodes) == OffloadingAction.EDGE_CPU
    assert CloudOnly().select_action(make_task(privacy_level=2), nodes) == OffloadingAction.LOCAL_CPU


def test_greedy_policies_return_privacy_valid_actions() -> None:
    nodes = create_default_nodes()
    task = make_task(privacy_level=2)

    latency_action = GreedyLatency().select_action(task, nodes)
    energy_action = GreedyEnergy().select_action(task, nodes)

    assert latency_action == OffloadingAction.LOCAL_CPU
    assert energy_action == OffloadingAction.LOCAL_CPU


def test_threshold_policy_uses_npu_for_tight_deadline_when_available() -> None:
    nodes = create_default_nodes()
    task = make_task(deadline_ms=80.0, bandwidth_mbps=90.0, npu_queue_len=1)

    assert ThresholdOffloading().select_action(task, nodes) == OffloadingAction.EDGE_NPU


def test_summarize_offloading_results() -> None:
    rows = [
        {
            "action": "EDGE_NPU",
            "latency_ms": 10.0,
            "energy_j": 1.0,
            "success": True,
            "deadline_missed": False,
            "privacy_violation": False,
        },
        {
            "action": "CLOUD",
            "latency_ms": 30.0,
            "energy_j": 3.0,
            "success": False,
            "deadline_missed": True,
            "privacy_violation": False,
        },
    ]

    summary = summarize_offloading_results(rows)

    assert summary["avg_latency_ms"] == 20.0
    assert summary["success_rate"] == 0.5
    assert summary["npu_selection_rate"] == 0.5
    assert summary["cloud_selection_rate"] == 0.5
