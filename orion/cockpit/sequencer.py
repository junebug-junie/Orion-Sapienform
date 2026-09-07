from __future__ import annotations

_COUNTERS: dict[str, int] = {}


def reset_seq(correlation_id: str) -> None:
    _COUNTERS[correlation_id] = 0


def next_seq(correlation_id: str) -> int:
    seq = _COUNTERS.get(correlation_id, 0)
    _COUNTERS[correlation_id] = seq + 1
    return seq
