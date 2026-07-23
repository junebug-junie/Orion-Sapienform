"""fetch_falkor_neighborhood_fragments: keyword -> Entity match -> ChatTurn
traversal -> Postgres text join. Replaces storage/rdf_adapter.py's
_fetch_rdf_neighborhood_fragments (the last live Fuseki read path in this
service, confirmed via live Fuseki container log read 2026-07-22).

Same return-fragment contract as the other Falkor adapters
(id/source/source_ref/uri/text/ts/tags/score/meta) -- fusion.py has no
per-backend adapter layer, so shape drift here would silently break scoring.
"""

from __future__ import annotations

import asyncio

from app.storage import falkor_neighborhood_adapter as neigh


class _FakeFalkorClient:
    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    def graph_query(self, cypher, params=None):
        self.calls.append((cypher, params))
        return self._rows


def _run(coro):
    return asyncio.run(coro)


def test_fetch_falkor_neighborhood_fragments_full_shape(monkeypatch) -> None:
    fake_client = _FakeFalkorClient([{"name": "Circe"}])
    monkeypatch.setattr(neigh, "get_recall_falkor_client", lambda: fake_client)

    async def _fake_turns(*, target_names, max_results):
        assert target_names == ["Circe"]
        return [{"turn_id": "turn-1", "ts": "2026-07-18T07:46:34+00:00"}]

    monkeypatch.setattr(neigh, "fetch_turns_mentioning_entities", _fake_turns)

    async def _fake_text_join(turn_ids):
        assert turn_ids == ["turn-1"]
        return {"turn-1": ("tell me about Circe", "Circe is a node in the mesh")}

    monkeypatch.setattr(neigh, "fetch_chat_turns_by_id", _fake_text_join)

    out = _run(
        neigh.fetch_falkor_neighborhood_fragments(query_text="tell me about Circe", max_items=8)
    )

    assert len(out) == 1
    frag = out[0]
    assert frag["id"] == "turn-1"
    assert frag["source"] == "falkor_neighborhood"
    assert frag["source_ref"] == "falkordb"
    assert frag["uri"] == "turn-1"
    assert frag["text"] == 'ExactUserText: "tell me about Circe"\nOrionResponse: "Circe is a node in the mesh"'
    assert frag["ts"] > 0
    assert frag["tags"] == ["falkor", "neighborhood"]
    assert frag["score"] == 0.5
    assert frag["meta"] == {"matched_entities": ["Circe"]}

    # Entity match query is keyword-driven, case-insensitive, bounded.
    cypher, params = fake_client.calls[0]
    assert "MATCH (e:Entity)" in cypher
    assert "toLower(e.name) CONTAINS kw" in cypher
    assert "circe" in params["keywords"]


def test_extract_keywords_filters_stopwords() -> None:
    """Regression for a real live finding (2026-07-22): unfiltered "and" alone
    false-positive-matched 'sandra bullock', 'nelson mandela', 'england',
    'landing pad', 'amanda', 'grand canyon skywalk' as CONTAINS substrings
    against the real orion_recall Entity set. Stopwords must never survive
    into the keyword list this module uses for entity matching."""
    keywords = neigh._extract_keywords("tell me about Orion and reverie")
    assert keywords == ["orion", "reverie"]
    assert "and" not in keywords
    assert "about" not in keywords
    assert "tell" not in keywords


def test_no_keywords_short_circuits_without_calling_falkor(monkeypatch) -> None:
    called = False

    def _client():
        nonlocal called
        called = True
        return _FakeFalkorClient([])

    monkeypatch.setattr(neigh, "get_recall_falkor_client", _client)
    # "it a" tokenizes to nothing (3+ char token regex) -- no keywords to match on.
    out = _run(neigh.fetch_falkor_neighborhood_fragments(query_text="it a", max_items=8))
    assert out == []
    assert called is False


def test_empty_query_text_short_circuits(monkeypatch) -> None:
    called = False

    def _client():
        nonlocal called
        called = True
        return _FakeFalkorClient([])

    monkeypatch.setattr(neigh, "get_recall_falkor_client", _client)
    out = _run(neigh.fetch_falkor_neighborhood_fragments(query_text="", max_items=8))
    assert out == []
    assert called is False


def test_none_client_returns_empty(monkeypatch) -> None:
    monkeypatch.setattr(neigh, "get_recall_falkor_client", lambda: None)
    out = _run(neigh.fetch_falkor_neighborhood_fragments(query_text="tell me about Circe", max_items=8))
    assert out == []


def test_no_entity_match_returns_empty_not_noisy_fallback(monkeypatch) -> None:
    """A keyword that matches no real Entity node (e.g. a generic word) must
    return [] -- the whole point of swapping off the old blind-triple-scan
    behavior, not something to silently degrade back toward."""
    fake_client = _FakeFalkorClient([])  # no entity rows match
    monkeypatch.setattr(neigh, "get_recall_falkor_client", lambda: fake_client)

    async def _should_not_be_called(**kwargs):
        raise AssertionError("fetch_turns_mentioning_entities should not run with no matched entities")

    monkeypatch.setattr(neigh, "fetch_turns_mentioning_entities", _should_not_be_called)

    out = _run(neigh.fetch_falkor_neighborhood_fragments(query_text="what's the weather today", max_items=8))
    assert out == []


def test_falkor_entity_match_failure_degrades_to_empty(monkeypatch) -> None:
    class _RaisingClient:
        def graph_query(self, cypher, params=None):
            raise RuntimeError("falkor down")

    monkeypatch.setattr(neigh, "get_recall_falkor_client", lambda: _RaisingClient())
    out = _run(neigh.fetch_falkor_neighborhood_fragments(query_text="tell me about Circe", max_items=8))
    assert out == []


def test_turns_mentioning_entities_failure_degrades_to_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        neigh, "get_recall_falkor_client", lambda: _FakeFalkorClient([{"name": "Circe"}])
    )

    async def _raise(**kwargs):
        raise RuntimeError("falkor down")

    monkeypatch.setattr(neigh, "fetch_turns_mentioning_entities", _raise)
    out = _run(neigh.fetch_falkor_neighborhood_fragments(query_text="tell me about Circe", max_items=8))
    assert out == []


def test_postgres_join_failure_degrades_to_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        neigh, "get_recall_falkor_client", lambda: _FakeFalkorClient([{"name": "Circe"}])
    )

    async def _fake_turns(*, target_names, max_results):
        return [{"turn_id": "turn-1", "ts": "2026-07-18T00:00:00+00:00"}]

    monkeypatch.setattr(neigh, "fetch_turns_mentioning_entities", _fake_turns)

    async def _raise(turn_ids):
        raise RuntimeError("postgres down")

    monkeypatch.setattr(neigh, "fetch_chat_turns_by_id", _raise)
    out = _run(neigh.fetch_falkor_neighborhood_fragments(query_text="tell me about Circe", max_items=8))
    assert out == []


def test_drops_turns_with_no_postgres_row(monkeypatch) -> None:
    monkeypatch.setattr(
        neigh, "get_recall_falkor_client", lambda: _FakeFalkorClient([{"name": "Circe"}])
    )

    async def _fake_turns(*, target_names, max_results):
        return [{"turn_id": "turn-orphan", "ts": "2026-07-18T00:00:00+00:00"}]

    monkeypatch.setattr(neigh, "fetch_turns_mentioning_entities", _fake_turns)

    async def _empty_join(turn_ids):
        return {}

    monkeypatch.setattr(neigh, "fetch_chat_turns_by_id", _empty_join)
    out = _run(neigh.fetch_falkor_neighborhood_fragments(query_text="tell me about Circe", max_items=8))
    assert out == []
