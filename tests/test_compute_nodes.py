from src.env.compute_nodes import OffloadingAction, create_default_nodes
from src.env.task_generator import Task


def make_task() -> Task:
    return Task(
        task_id=0,
        data_size_mb=2.0,
        cpu_cycles=1.0,
        deadline_ms=300.0,
        priority=1,
        privacy_level=0,
        bandwidth_mbps=50.0,
        local_queue_len=1,
        edge_queue_len=2,
        npu_queue_len=1,
        cloud_rtt_ms=40.0,
    )


def test_compute_node_outputs_are_valid() -> None:
    task = make_task()
    nodes = create_default_nodes()

    for action, node in nodes.items():
        result = node.execute(task)
        assert result.action == action
        assert result.latency_ms >= 0.0
        assert result.energy_j >= 0.0
        assert isinstance(result.success, bool)
        assert isinstance(result.deadline_missed, bool)
        assert isinstance(result.privacy_violation, bool)


def test_edge_npu_is_faster_than_edge_cpu_for_compute_heavy_task() -> None:
    task = make_task()
    nodes = create_default_nodes()

    edge_cpu = nodes[OffloadingAction.EDGE_CPU].execute(task)
    edge_npu = nodes[OffloadingAction.EDGE_NPU].execute(task)

    assert edge_npu.latency_ms < edge_cpu.latency_ms
