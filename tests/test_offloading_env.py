import importlib

import numpy as np

from src.env.offloading_env import OffloadingEnv


MODULES = [
    "src.env.task_generator",
    "src.env.compute_nodes",
    "src.env.offloading_env",
    "src.env.reward",
    "src.algorithms.local_only",
    "src.algorithms.cloud_only",
    "src.algorithms.greedy_latency",
    "src.algorithms.greedy_energy",
    "src.algorithms.threshold_policy",
    "src.algorithms.rl_policy",
    "src.models.policy_mlp",
    "src.deployment.export_onnx",
    "src.deployment.quantize_onnx",
    "src.deployment.benchmark_onnx_cpu",
    "src.deployment.simulated_npu_runtime",
    "src.utils.metrics",
    "src.utils.logger",
    "src.utils.seed",
]


def test_offloading_env_reset_and_step_behavior() -> None:
    env = OffloadingEnv(seed=1, max_steps=2)
    observation, reset_info = env.reset(seed=1)

    assert isinstance(observation, np.ndarray)
    assert observation.shape == env.observation_space.shape
    assert env.observation_space.contains(observation)
    assert reset_info["task_id"] == 0

    next_observation, reward, terminated, truncated, info = env.step(0)

    assert env.observation_space.contains(next_observation)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert info["action"] == "LOCAL_CPU"
    assert "latency_ms" in info
    assert "energy_j" in info


def test_offloading_env_truncates_at_max_steps() -> None:
    env = OffloadingEnv(seed=1, max_steps=1)
    env.reset(seed=1)

    _, _, terminated, truncated, _ = env.step(0)

    assert terminated is False
    assert truncated is True


def test_all_stage_modules_import() -> None:
    for module_name in MODULES:
        importlib.import_module(module_name)
