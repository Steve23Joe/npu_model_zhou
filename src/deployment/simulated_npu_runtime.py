"""Queue-aware simulated NPU runtime for edge inference experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimulatedNPUResult:
    """Result of one simulated NPU inference request."""

    latency_ms: float
    energy_j: float
    success: bool
    queue_depth: int
    utilization: float
    overloaded: bool


class SimulatedNPURuntime:
    """Simulate NPU latency, queue congestion, utilization, and failures."""

    def __init__(
        self,
        base_latency_ms: float = 0.35,
        queue_limit: int = 16,
        batch_size: int = 4,
        batch_speedup: float = 2.5,
        power_watt: float = 6.0,
        failure_rate: float = 0.01,
        seed: int = 42,
    ) -> None:
        if queue_limit <= 0:
            raise ValueError("queue_limit must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_speedup <= 0:
            raise ValueError("batch_speedup must be positive")
        if not 0.0 <= failure_rate <= 1.0:
            raise ValueError("failure_rate must be in [0, 1]")

        self.base_latency_ms = base_latency_ms
        self.queue_limit = queue_limit
        self.batch_size = batch_size
        self.batch_speedup = batch_speedup
        self.power_watt = power_watt
        self.failure_rate = failure_rate
        self._rng = np.random.default_rng(seed)
        self._queue_depth = 0
        self._processed = 0
        self._failures = 0
        self._utilization_samples: list[float] = []

    def infer(self, incoming_requests: int = 1) -> SimulatedNPUResult:
        """Simulate one scheduling tick with incoming inference requests."""
        if incoming_requests < 0:
            raise ValueError("incoming_requests must be non-negative")

        self._queue_depth += incoming_requests
        overloaded = self._queue_depth > self.queue_limit
        processed_now = min(self._queue_depth, self.batch_size)
        effective_batch = max(processed_now, 1)
        batch_gain = 1.0 + (effective_batch - 1) * ((self.batch_speedup - 1.0) / self.batch_size)
        latency_ms = self.base_latency_ms * effective_batch / batch_gain

        if overloaded:
            overflow = self._queue_depth - self.queue_limit
            latency_ms += overflow * self.base_latency_ms

        utilization = min(self._queue_depth / self.queue_limit, 1.0)
        random_failure = bool(self._rng.random() < self.failure_rate)
        success = not overloaded and not random_failure

        self._queue_depth = max(0, self._queue_depth - processed_now)
        self._processed += int(success)
        self._failures += int(not success)
        self._utilization_samples.append(utilization)
        energy_j = self.power_watt * (latency_ms / 1000.0)

        return SimulatedNPUResult(
            latency_ms=float(latency_ms),
            energy_j=float(energy_j),
            success=success,
            queue_depth=self._queue_depth,
            utilization=float(utilization),
            overloaded=overloaded,
        )

    def run_workload(self, request_pattern: list[int]) -> dict[str, float]:
        """Run a request pattern and return aggregate simulated NPU metrics."""
        results = [self.infer(incoming_requests=count) for count in request_pattern]
        latencies = np.array([result.latency_ms for result in results], dtype=np.float64)
        energies = np.array([result.energy_j for result in results], dtype=np.float64)
        successes = np.array([result.success for result in results], dtype=np.float64)
        overloads = np.array([result.overloaded for result in results], dtype=np.float64)
        utilizations = np.array([result.utilization for result in results], dtype=np.float64)

        return {
            "requests": float(sum(request_pattern)),
            "ticks": float(len(request_pattern)),
            "mean_latency_ms": float(np.mean(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "mean_energy_j": float(np.mean(energies)),
            "success_rate": float(np.mean(successes)),
            "failure_rate_observed": float(1.0 - np.mean(successes)),
            "overload_rate": float(np.mean(overloads)),
            "mean_utilization": float(np.mean(utilizations)),
            "final_queue_depth": float(self._queue_depth),
        }
