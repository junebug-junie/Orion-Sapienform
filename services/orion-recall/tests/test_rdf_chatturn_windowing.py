"""RDF chat-turn recall must honor the profile time window.

The graph stores no usable ChatTurn timestamp, so these rails otherwise leak months-old
turns into reflective recall. worker resolves each turn's created_at from chat_history_log
and drops turns outside the window. Memory cards / SQL / RDF-claim candidates are untouched.
"""

from __future__ import annotations

import asyncio

from app import worker


def test_chatturn_id_from_fragment_reverses_sanitized_uuid() -> None:
    frag = {"id": "http://conjourney.net/orion/chatTurn/497229f6_12da_4ffa_9f92_95aa950db58f"}
    assert worker._chatturn_id_from_fragment(frag) == "497229f6-12da-4ffa-9f92-95aa950db58f"


def test_chatturn_id_from_fragment_rejects_non_chatturn_and_non_uuid() -> None:
    assert worker._chatturn_id_from_fragment({"id": "http://conjourney.net/orion/claim/abc"}) is None
    assert worker._chatturn_id_from_fragment({"id": "http://conjourney.net/orion/chatTurn/not_a_uuid"}) is None
    assert worker._chatturn_id_from_fragment({}) is None


def _run_window(candidates, since_minutes, ts_map):
    async def _fake_fetch(turn_ids, since):
        # Only ids the caller asked for, filtered to what the "DB" returns for the window.
        return {tid: ts_map[tid] for tid in turn_ids if tid in ts_map}

    original = worker.fetch_chat_turn_timestamps
    worker.fetch_chat_turn_timestamps = _fake_fetch  # type: ignore[assignment]
    try:
        return asyncio.run(
            worker._window_rdf_chatturn_candidates(candidates, since_minutes=since_minutes)
        )
    finally:
        worker.fetch_chat_turn_timestamps = original  # type: ignore[assignment]


def test_windowing_drops_out_of_window_and_stamps_kept() -> None:
    in_window = "497229f6-12da-4ffa-9f92-95aa950db58f"
    out_window = "181c2c85-da7a-4de8-a475-4291ae290e0e"
    candidates = [
        {"id": f"http://conjourney.net/orion/chatTurn/{in_window.replace('-', '_')}",
         "source": "rdf_chat", "tags": ["rdf", "chat", "chatturn"], "ts": 0.0, "text": "recent"},
        {"id": f"http://conjourney.net/orion/chatTurn/{out_window.replace('-', '_')}",
         "source": "rdf_chat", "tags": ["rdf", "chat", "chatturn"], "ts": 0.0, "text": "months old"},
        {"id": "chat_history_log:abc", "source": "sql_chat", "tags": ["sql"], "ts": 1783220298.0, "text": "sql row"},
    ]
    # Only the in-window turn is returned by the DB (out-of-window filtered by the SQL query).
    kept, dropped = _run_window(candidates, 1440, {in_window: 1783300000.0})

    assert dropped == 1
    sources = [c["source"] for c in kept]
    assert sources == ["rdf_chat", "sql_chat"]  # out-of-window rdf_chat gone, sql untouched
    kept_rdf = next(c for c in kept if c["source"] == "rdf_chat")
    assert kept_rdf["ts"] == 1783300000.0  # real timestamp stamped in
    kept_sql = next(c for c in kept if c["source"] == "sql_chat")
    assert kept_sql["ts"] == 1783220298.0  # non chat-turn candidate untouched


def test_windowing_noop_when_since_minutes_non_positive() -> None:
    candidates = [
        {"id": "http://conjourney.net/orion/chatTurn/497229f6_12da_4ffa_9f92_95aa950db58f",
         "source": "rdf_chat", "tags": ["chatturn"], "ts": 0.0},
    ]
    kept, dropped = _run_window(candidates, 0, {})
    assert dropped == 0
    assert kept == candidates


def test_windowing_skips_db_when_no_chatturn_candidates() -> None:
    candidates = [{"id": "x", "source": "cards", "tags": ["card"], "ts": 5.0}]

    async def _boom(*_a, **_k):  # must not be called
        raise AssertionError("DB lookup should be skipped when no chat-turn candidates")

    original = worker.fetch_chat_turn_timestamps
    worker.fetch_chat_turn_timestamps = _boom  # type: ignore[assignment]
    try:
        kept, dropped = asyncio.run(
            worker._window_rdf_chatturn_candidates(candidates, since_minutes=1440)
        )
    finally:
        worker.fetch_chat_turn_timestamps = original  # type: ignore[assignment]

    assert dropped == 0
    assert kept == candidates
