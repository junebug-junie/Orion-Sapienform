"""Wall-clock budget for Cortex-governed Mind LLM phases."""

from __future__ import annotations

import time


class MindRunBudget:
    """Tracks remaining wall time for a single MindRun and caps per-phase timeouts."""

    def __init__(self, wall_time_ms_max: float, *, safety_ms: float = 50.0) -> None:
        self._t0 = time.perf_counter()
        self._wall_ms = float(wall_time_ms_max)
        self._safety_ms = float(safety_ms)

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def remaining_ms(self) -> float:
        return max(0.0, self._wall_ms - self.elapsed_ms())

    def can_run_phase(self, *, min_ms: float = 100.0) -> bool:
        return self.remaining_ms() > (self._safety_ms + min_ms)

    def phase_timeout_sec(self, configured_sec: float) -> float:
        remaining_sec = self.remaining_ms() / 1000.0
        if remaining_sec <= self._safety_ms / 1000.0:
            return 0.0
        return max(0.0, min(float(configured_sec), remaining_sec - self._safety_ms / 1000.0))

    def over_budget(self) -> bool:
        return self.elapsed_ms() > self._wall_ms
