"""fetch_falkor_chatturn_fragments: Falkor turn discovery + Postgres text join.

Same return-fragment contract as storage/rdf_adapter.py::fetch_rdf_chatturn_fragments
(id/source/source_ref/uri/text/ts/tags/score/meta) -- fusion.py has no
per-backend adapter layer, so shape drift here would silently break scoring.
"""

from __future__ import annotations

import asyncio

import pytest

from app.storage import falkor_chat_adapter


class _FakeFalkorClient:
    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    def graph_query(self, cypher, params=None):
        self.calls.append((cypher, params))
        return self._rows


def _run(coro):
    return asyncio.run(coro)


def test_fetch_falkor_chatturn_fragments_full_shape(monkeypatch) -> None:
    rows = [
        {"turn_id": "turn-1", "ts": "2026-07-18T07:46:34+00:00", "correlation_id": "turn-1"},
    ]
    fake_client = _FakeFalkorClient(rows)
    monkeypatch.setattr(falkor_chat_adapter, "get_recall_falkor_client", lambda: fake_client)

    async def _fake_text_join(turn_ids):
        assert turn_ids == ["turn-1"]
        return {"turn-1": ("hi Circe", "hello")}

    monkeypatch.setattr(falkor_chat_adapter, "fetch_chat_turns_by_id", _fake_text_join)

    out = _run(
        falkor_chat_adapter.fetch_falkor_chatturn_fragments(
            query_text="hi", session_id="sess-1", max_items=20
        )
    )

    assert len(out) == 1
    frag = out[0]
    assert frag["id"] == "turn-1"
    assert frag["source"] == "falkor_chat"
    assert frag["source_ref"] == "falkordb"
    assert frag["uri"] == "turn-1"
    assert frag["text"] == 'ExactUserText: "hi Circe"\nOrionResponse: "hello"'
    assert frag["ts"] > 0
    assert frag["tags"] == ["falkor", "chat", "chatturn"]
    assert frag["score"] == 0.50
    assert frag["meta"] == {}

    # LIMIT is bounded (not passed through raw) -- confirms the max(1, min(...,100)) clamp.
    cypher, params = fake_client.calls[0]
    assert "ORDER BY t.ts DESC" in cypher
    # Regression: source_kind filter is load-bearing (see the Cypher's own
    # comment) -- without it, social.turn.stored.v1 turns (which never
    # resolve against chat_history_log) crowd real chat.history turns out
    # of the top-N window.
    assert "source_kind: 'chat.history'" in cypher
    assert params == {"max_items": 20}


def test_fetch_falkor_chatturn_fragments_drops_turns_with_no_postgres_row(monkeypatch) -> None:
    """A ChatTurn node with no matching chat_history_log row has nothing to
    quote -- must be dropped, not emitted with empty text."""
    rows = [
        {"turn_id": "turn-orphan", "ts": "2026-07-18T00:00:00+00:00", "correlation_id": None},
    ]
    monkeypatch.setattr(
        falkor_chat_adapter, "get_recall_falkor_client", lambda: _FakeFalkorClient(rows)
    )

    async def _empty_join(turn_ids):
        return {}

    monkeypatch.setattr(falkor_chat_adapter, "fetch_chat_turns_by_id", _empty_join)

    out = _run(
        falkor_chat_adapter.fetch_falkor_chatturn_fragments(
            query_text="hi", session_id=None, max_items=20
        )
    )
    assert out == []


def test_fetch_falkor_chatturn_fragments_empty_query_text_short_circuits(monkeypatch) -> None:
    called = False

    def _client():
        nonlocal called
        called = True
        return _FakeFalkorClient([])

    monkeypatch.setattr(falkor_chat_adapter, "get_recall_falkor_client", _client)
    out = _run(
        falkor_chat_adapter.fetch_falkor_chatturn_fragments(
            query_text="", session_id=None, max_items=20
        )
    )
    assert out == []
    assert called is False


def test_fetch_falkor_chatturn_fragments_none_client_returns_empty(monkeypatch) -> None:
    monkeypatch.setattr(falkor_chat_adapter, "get_recall_falkor_client", lambda: None)
    out = _run(
        falkor_chat_adapter.fetch_falkor_chatturn_fragments(
            query_text="hi", session_id=None, max_items=20
        )
    )
    assert out == []


def test_fetch_falkor_chatturn_fragments_falkor_failure_degrades_to_empty(monkeypatch) -> None:
    class _RaisingClient:
        def graph_query(self, cypher, params=None):
            raise RuntimeError("falkor down")

    monkeypatch.setattr(falkor_chat_adapter, "get_recall_falkor_client", lambda: _RaisingClient())
    out = _run(
        falkor_chat_adapter.fetch_falkor_chatturn_fragments(
            query_text="hi", session_id=None, max_items=20
        )
    )
    assert out == []


def test_fetch_falkor_chatturn_fragments_postgres_failure_degrades_to_empty(monkeypatch) -> None:
    rows = [{"turn_id": "turn-1", "ts": "2026-07-18T00:00:00+00:00", "correlation_id": "turn-1"}]
    monkeypatch.setattr(
        falkor_chat_adapter, "get_recall_falkor_client", lambda: _FakeFalkorClient(rows)
    )

    async def _raising_join(turn_ids):
        raise RuntimeError("postgres down")

    monkeypatch.setattr(falkor_chat_adapter, "fetch_chat_turns_by_id", _raising_join)
    out = _run(
        falkor_chat_adapter.fetch_falkor_chatturn_fragments(
            query_text="hi", session_id=None, max_items=20
        )
    )
    assert out == []
