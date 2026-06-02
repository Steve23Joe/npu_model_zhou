from src.deployment.simulated_npu_runtime import SimulatedNPURuntime


def test_simulated_npu_runtime_reports_valid_metrics() -> None:
    runtime = SimulatedNPURuntime(
        base_latency_ms=1.0,
        queue_limit=4,
        batch_size=2,
        batch_speedup=2.0,
        power_watt=5.0,
        failure_rate=0.0,
        seed=1,
    )

    summary = runtime.run_workload([1, 2, 8])

    assert summary["mean_latency_ms"] > 0.0
    assert summary["mean_energy_j"] > 0.0
    assert 0.0 <= summary["success_rate"] <= 1.0
    assert summary["overload_rate"] > 0.0
    assert summary["final_queue_depth"] >= 0.0
