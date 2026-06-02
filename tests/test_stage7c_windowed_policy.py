import torch

from src.models.windowed_offloading_policy import (
    WindowedPolicyConfig,
    WindowedTransformerOffloadingPolicy,
    count_parameters,
)


def test_windowed_policy_output_shape() -> None:
    config = WindowedPolicyConfig(
        feature_dim=24,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
    )
    model = WindowedTransformerOffloadingPolicy(config)
    task_window = torch.randn(2, 8, 24)

    output = model(task_window)

    assert output.shape == (2, 8, 4)


def test_windowed_policy_parameter_count_is_positive() -> None:
    model = WindowedTransformerOffloadingPolicy(
        WindowedPolicyConfig(d_model=32, nhead=4, num_layers=1, dim_feedforward=64)
    )

    assert count_parameters(model) > 0
