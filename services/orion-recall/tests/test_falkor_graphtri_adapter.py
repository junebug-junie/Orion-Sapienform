"""fetch_falkor_graphtri_fragments / fetch_falkor_graphtri_anchors: entity-
mention fragments standing in for the old Claim-based graphtri lane.

Same return-fragment contract as storage/rdf_adapter.py::fetch_rdf_graphtri_fragments
(id/source/source_ref/uri/text/ts/tags/score/meta), including the literal
"claim" tag render.py's is_graphtri bucketing keys off of.
"""

from __future__ import annotations

import asyncio

from app.storage import falkor_graphtri_adapter


class _FakeFalkorClient:
    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    def graph_query(self, cypher, params=None):
        self.calls.append((cypher, params))
        return self._rows


def _run(coro):
    return asyncio.run(coro)


def test_fetch_falkor_graphtri_fragments_full_shape(monkeypatch) -> None:
    rows = [
        {"turn_id": "turn-1", "entity": "circe", "ts": "2026-07-18T07:46:34+00:00"},
    ]
    fake_client = _FakeFalkorClient(rows)
    monkeypatch.setattr(falkor_graphtri_adapter, "get_recall_falkor_client", lambda: fake_client)

    out = _run(
        falkor_graphtri_adapter.fetch_falkor_graphtri_fragments(
            query_text="tell me about circe", session_id=None, keywords=["circe"], max_items=8
        )
    )

    assert len(out) == 1
    frag = out[0]
    assert frag["id"] == "turn-1:circe"
    assert frag["source"] == "falkor_graphtri"
    assert frag["source_ref"] == "falkordb"
    assert frag["uri"] == "turn-1"
    assert frag["text"] == 'Claim: mentions circe | evidence=turn-1'
    assert frag["ts"] > 0
    assert frag["tags"] == ["falkor", "graphtri", "claim"]  # "claim" required for render.py bucketing
    assert frag["score"] == 0.6
    assert frag["meta"] == {"subject": "turn-1", "entity": "circe"}

    cypher, params = fake_client.calls[0]
    assert "source_kind: 'chat.history'" in cypher
    assert "MENTIONS_ENTITY" in cypher
    assert params["keywords"] == ["circe"]


def test_fetch_falkor_graphtri_fragments_no_keywords_returns_empty(monkeypatch) -> None:
    called = False

    def _client():
        nonlocal called
        called = True
        return _FakeFalkorClient([])

    monkeypatch.setattr(falkor_graphtri_adapter, "get_recall_falkor_client", _client)
    out = _run(
        falkor_graphtri_adapter.fetch_falkor_graphtri_fragments(
            query_text="hi", session_id=None, keywords=[], max_items=8
        )
    )
    assert out == []
    assert called is False  # short-circuits before touching Falkor


def test_fetch_falkor_graphtri_fragments_none_client_returns_empty(monkeypatch) -> None:
    monkeypatch.setattr(falkor_graphtri_adapter, "get_recall_falkor_client", lambda: None)
    out = _run(
        falkor_graphtri_adapter.fetch_falkor_graphtri_fragments(
            query_text="hi", session_id=None, keywords=["hi"], max_items=8
        )
    )
    assert out == []


def test_fetch_falkor_graphtri_fragments_failure_degrades_to_empty(monkeypatch) -> None:
    class _RaisingClient:
        def graph_query(self, cypher, params=None):
            raise RuntimeError("falkor down")

    monkeypatch.setattr(falkor_graphtri_adapter, "get_recall_falkor_client", lambda: _RaisingClient())
    out = _run(
        falkor_graphtri_adapter.fetch_falkor_graphtri_fragments(
            query_text="hi", session_id=None, keywords=["hi"], max_items=8
        )
    )
    assert out == []


def test_fetch_falkor_graphtri_anchors_filtered_result(monkeypatch) -> None:
    rows = [{"name": "circe"}, {"name": "orion"}]
    fake_client = _FakeFalkorClient(rows)
    monkeypatch.setattr(falkor_graphtri_adapter, "get_recall_falkor_client", lambda: fake_client)

    out = _run(
        falkor_graphtri_adapter.fetch_falkor_graphtri_anchors(
            session_id=None, query_terms=["circe"], max_terms=8
        )
    )
    assert out["entities_terms"] == ["circe", "orion"]
    assert out["tags_terms"] == []  # honest: Falkor has no :Tag data
    assert out["claim_objs"] == ["circe", "orion"]  # mirrors entities_terms, see module docstring
    assert out["related_terms"] == ["circe", "orion"]
    # Only 1 call: the filtered query returned >= 3... wait it returned 2, so
    # the fallback unfiltered query should have run too.
    assert len(fake_client.calls) == 2


def test_fetch_falkor_graphtri_anchors_thin_result_falls_back_to_unfiltered(monkeypatch) -> None:
    call_count = 0

    class _SequencedClient:
        def graph_query(self, cypher, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [{"name": "circe"}]  # thin (< 3), triggers fallback
            return [{"name": "circe"}, {"name": "orion"}, {"name": "juniper"}]

    monkeypatch.setattr(falkor_graphtri_adapter, "get_recall_falkor_client", lambda: _SequencedClient())
    out = _run(
        falkor_graphtri_adapter.fetch_falkor_graphtri_anchors(
            session_id=None, query_terms=["circe"], max_terms=8
        )
    )
    assert call_count == 2
    assert out["entities_terms"] == ["circe", "orion", "juniper"]


def test_fetch_falkor_graphtri_anchors_none_client_returns_empty_dict(monkeypatch) -> None:
    monkeypatch.setattr(falkor_graphtri_adapter, "get_recall_falkor_client", lambda: None)
    out = _run(
        falkor_graphtri_adapter.fetch_falkor_graphtri_anchors(
            session_id=None, query_terms=["circe"], max_terms=8
        )
    )
    assert out == {"entities_terms": [], "tags_terms": [], "claim_objs": [], "related_terms": []}
