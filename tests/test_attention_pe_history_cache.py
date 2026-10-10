from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.attention.pe_history_cache import NodePeHistoryCache

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)


def test_seed_then_incremental_with_overlap_dedupes() -> None:
    rows = [("n", NOW - timedelta(minutes=m), float(m)) for m in range(0, 300, 5)]
    calls = []

    def fetch(since):
        calls.append(since)
        return [r for r in rows if r[1] >= since]

    cache = NodePeHistoryCache()
    assert cache.refresh(fetch, now=NOW) == len(rows)
    assert calls[0] == NOW - timedelta(days=7)
    assert cache.refresh(fetch, now=NOW) == 0  # overlap re-read, nothing double counted
    assert calls[1] == NOW - timedelta(minutes=10)
    rows.append(("n", NOW + timedelta(seconds=30), 9.0))
    assert cache.refresh(fetch, now=NOW + timedelta(minutes=1)) == 1
    assert cache.current()["n"] == (9.0, NOW + timedelta(seconds=30))


def test_late_row_is_inserted_in_order_and_old_rows_trimmed() -> None:
    cache = NodePeHistoryCache()
    cache.refresh(lambda s: [("n", NOW - timedelta(days=8), 1.0), ("n", NOW, 2.0)], now=NOW)
    assert cache.current()["n"][0] == 2.0
    cache.refresh(lambda s: [("n", NOW - timedelta(minutes=3), 5.0)], now=NOW)
    assert cache.current()["n"][0] == 2.0  # a late older row is not "current"
    mags = cache.magnitudes(now=NOW)
    assert mags["n"][0].n_readings_7d == 2  # the 8-day-old row was trimmed


def test_read_error_propagates_to_the_caller() -> None:
    def boom(since):
        raise RuntimeError("db down")

    with pytest.raises(RuntimeError):
        NodePeHistoryCache().refresh(boom, now=NOW)
