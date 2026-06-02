from src.env.task_generator import Task, TaskGenerator


def test_task_generation_reproducibility() -> None:
    first = TaskGenerator(seed=123).sample_batch(5)
    second = TaskGenerator(seed=123).sample_batch(5)

    assert first == second
    assert first[0].task_id == 0
    assert first[-1].task_id == 4


def test_task_fields_are_in_expected_ranges() -> None:
    task = TaskGenerator(seed=1).sample()

    assert isinstance(task, Task)
    assert 0.05 <= task.data_size_mb <= 6.0
    assert 0.1 <= task.cpu_cycles <= 6.0
    assert 110.0 <= task.deadline_ms <= 680.0
    assert 1 <= task.priority <= 3
    assert 0 <= task.privacy_level <= 2
    assert task.bandwidth_mbps > 0
