"""Post-fetch windowing covers falkor_chat candidates too (see worker.py's
_SQL_CHAT_SOURCES / _LOCAL_TS_ONLY_SOURCES / _window_sql_chat_candidates
docstring)."""

from __future__ import annotations

import asyncio
import time

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


def test_falkor_chat_windowing_never_hits_postgres_even_when_id_is_uuid() -> None:
    """falkor_chat is in _LOCAL_TS_ONLY_SOURCES: even though its id is
    UUID-shaped and WOULD resolve through the same Postgres path sql_chat
    rows use, it deliberately skips that lookup -- fetch_falkor_chatturn_fragments
    already did its own Postgres join for text and stamped an accurate ts
    from Falkor's ChatTurn.ts, so re-resolving it here would be a redundant
    round-trip. Proven by seeding a ts_map entry that, if consulted, would
    produce a different (wrong) ts than the fragment's own -- it isn't."""
    turn_id = "497229f6-12da-4ffa-9f92-95aa950db58f"
    own_ts = time.time() - 60
    candidates = [
        {"id": turn_id, "source": "falkor_chat", "tags": ["falkor", "chat", "chatturn"], "ts": own_ts, "text": "hi"},
    ]
    kept, dropped = _run_window(candidates, 1440, {turn_id: 1_783_300_000.0})
    assert dropped == 0
    assert kept[0]["ts"] == own_ts


def test_falkor_chat_windowing_falls_back_to_local_ts_when_postgres_lookup_empty() -> None:
    """Falkor's own ts is already trustworthy (unlike RDF's, which needs the
    Postgres round-trip for any real timestamp at all) -- if the id doesn't
    resolve via chat_history_log, the fragment's own ts still governs."""
    recent_ts = time.time() - 3600
    candidates = [
        {"id": "unresolvable-id", "source": "falkor_chat", "tags": ["falkor", "chat"], "ts": recent_ts, "text": "ok"},
    ]
    kept, dropped = _run_window(candidates, 1440, {})
    assert dropped == 0
    assert kept == candidates


def test_falkor_chat_windowing_drops_stale_local_ts() -> None:
    old_ts = time.time() - (60 * 60 * 24 * 365)  # a year ago
    candidates = [
        {"id": "unresolvable-id", "source": "falkor_chat", "tags": ["falkor", "chat"], "ts": old_ts, "text": "old"},
    ]
    kept, dropped = _run_window(candidates, 1440, {})
    assert dropped == 1
    assert kept == []


def test_sql_chat_still_resolves_via_postgres_unaffected_by_falkor_change() -> None:
    """_LOCAL_TS_ONLY_SOURCES only opts falkor_chat out -- sql_chat/sql_timeline
    keep their original Postgres-resolution behavior."""
    turn_id = "497229f6-12da-4ffa-9f92-95aa950db58f"
    candidates = [
        {"id": turn_id, "source": "sql_chat", "tags": ["sql"], "ts": 0.0, "text": "hi"},
    ]
    kept, dropped = _run_window(candidates, 1440, {turn_id: 1_783_300_000.0})
    assert dropped == 0
    assert kept[0]["ts"] == 1_783_300_000.0
