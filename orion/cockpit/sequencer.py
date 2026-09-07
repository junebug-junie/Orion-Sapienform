from __future__ import annotations

_COUNTERS: dict[str, int] = {}


def reset_seq(correlation_id: str) -> None:
    _COUNTERS[correlation_id] = 0


def next_seq(correlation_id: str) -> int:
    seq = _COUNTERS.get(correlation_id, 0)
    _COUNTERS[correlation_id] = seq + 1
    return seq


def advance_seq(correlation_id: str, n: int) -> None:
    """Skip past *n* manually assigned seq slots after ``next_seq``."""
    if n <= 0:
        return
    current = _COUNTERS.get(correlation_id, 0)
    _COUNTERS[correlation_id] = current + n - 1
