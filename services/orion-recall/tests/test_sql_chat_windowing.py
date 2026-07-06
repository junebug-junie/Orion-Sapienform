"""Post-fetch SQL chat windowing drops stale sql_timeline/sql_chat rows."""

from __future__ import annotations

import asyncio

from app import worker


def _run_window(candidates, since_minutes, ts_map):
    async def _fake_fetch(turn_ids, since):
        return {tid: ts_map[tid] for tid in turn_ids if tid in ts_map}

    original = worker.fetch_chat_turn_timestamps
    worker.fetch_chat_turn_timestamps = _fake_fetch  # type: ignore[assignment]
    try:
        return asyncio.run(
            worker._window_sql_chat_candidates(candidates, since_minutes=since_minutes)
        )
    finally:
        worker.fetch_chat_turn_timestamps = original  # type: ignore[assignment]


def test_sql_windowing_drops_out_of_window_uuid_rows() -> None:
    in_window = "497229f6-12da-4ffa-9f92-95aa950db58f"
    out_window = "181c2c85-da7a-4de8-a475-4291ae290e0e"
    candidates = [
        {"id": in_window, "source": "sql_timeline", "tags": ["anchor_exact"], "ts": 0.0, "text": "recent"},
        {"id": out_window, "source": "sql_chat", "tags": ["sql"], "ts": 0.0, "text": "months old"},
        {"id": "card-1", "source": "cards", "tags": ["card"], "ts": 5.0, "text": "card"},
    ]
    kept, dropped = _run_window(candidates, 1440, {in_window: 1_783_300_000.0})

    assert dropped == 1
    assert [c["source"] for c in kept] == ["sql_timeline", "cards"]
    assert kept[0]["ts"] == 1_783_300_000.0


def test_sql_windowing_keeps_recent_rows_with_real_ts_when_id_not_uuid() -> None:
    import time

    recent_ts = time.time() - 3600
    candidates = [
        {"id": "chat_123", "source": "sql_timeline", "tags": ["chat_timeline"], "ts": recent_ts, "text": "ok"},
    ]
    kept, dropped = _run_window(candidates, 1440, {})
    assert dropped == 0
    assert kept == candidates
