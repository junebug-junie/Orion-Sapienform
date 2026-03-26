from __future__ import annotations

from collections import Counter
from threading import RLock


class WorkflowScheduleMetrics:
    """Tiny in-process counter surface for schedule hardening signals."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._counts: Counter[str] = Counter()

    def incr_attention(self, transition: str) -> None:
        if transition == "entered":
            key = "workflow_schedule_attention_entered_total"
        elif transition == "reminder":
            key = "workflow_schedule_attention_reminder_total"
        elif transition == "recovered":
            key = "workflow_schedule_attention_recovered_total"
        else:
            return
        with self._lock:
            self._counts[key] += 1

    def incr_error(self, error_code: str | None) -> None:
        code = str(error_code or "unknown").strip() or "unknown"
        with self._lock:
            self._counts[f"workflow_schedule_error_total|error_code={code}"] += 1

    def get(self, key: str) -> int:
        with self._lock:
            return int(self._counts.get(key, 0))

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)
