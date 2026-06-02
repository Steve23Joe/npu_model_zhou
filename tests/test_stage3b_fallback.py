from src.algorithms.dqn_with_fallback import DQNWithFallback
from src.env.compute_nodes import ComputeNode, OffloadingAction, create_default_nodes
from src.env.offloading_env import OffloadingEnv
from src.env.task_generator import Task
from src.utils.metrics import summarize_offloading_results


class FixedPolicy:
    name = "fixed"

    def __init__(self, action: OffloadingAction) -> None:
        self.action = action

    def select_action(
        self,
        task: Task,
        nodes: dict[OffloadingAction, ComputeNode],
    ) -> OffloadingAction:
        _ = task, nodes
        return self.action


def make_task(privacy_level: int = 0, npu_queue_len: int = 0) -> Task:
    return Task(
        task_id=0,
        data_size_mb=1.0,
        cpu_cycles=3.0,
        deadline_ms=500.0,
        priority=2,
        privacy_level=privacy_level,
        bandwidth_mbps=120.0,
        local_queue_len=0,
        edge_queue_len=1,
        npu_queue_len=npu_queue_len,
        cloud_rtt_ms=60.0,
    )


def test_fallback_prevents_high_privacy_cloud_action() -> None:
    nodes = create_default_nodes()
    policy = DQNWithFallback(FixedPolicy(OffloadingAction.CLOUD))

    action = policy.select_action(make_task(privacy_level=2), nodes)

    assert action == OffloadingAction.LOCAL_CPU


def test_fallback_avoids_npu_when_queue_is_overloaded() -> None:
    nodes = create_default_nodes()
    policy = DQNWithFallback(FixedPolicy(OffloadingAction.EDGE_NPU))

    action = policy.select_action(make_task(npu_queue_len=8), nodes)

    assert action != OffloadingAction.EDGE_NPU


def test_metrics_output_contains_required_stage3b_fields() -> None:
    rows = [
        {
            "action": "EDGE_NPU",
            "latency_ms": 100.0,
            "energy_j": 0.5,
            "success": True,
            "deadline_missed": False,
            "privacy_violation": False,
        }
    ]
    summary = summarize_offloading_results(rows)

    required = {
        "avg_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "success_rate",
        "deadline_miss_rate",
        "privacy_violation_rate",
        "avg_energy_j",
        "npu_selection_rate",
        "cloud_selection_rate",
    }
    assert required.issubset(summary)


def test_calibrated_environment_returns_valid_step_outputs() -> None:
    env = OffloadingEnv(seed=5, max_steps=1)
    observation, info = env.reset(seed=5)
    next_observation, reward, terminated, truncated, step_info = env.step(0)

    assert env.observation_space.contains(observation)
    assert env.observation_space.contains(next_observation)
    assert isinstance(info["task_id"], int)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is True
    assert "latency_ms" in step_info
