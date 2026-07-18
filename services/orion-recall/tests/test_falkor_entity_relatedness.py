"""fetch_related_entities / fetch_bridging_turns / fetch_entity_mention_timeline:
Phase 1 entity-graph reasoning primitives over the MENTIONS_ENTITY
co-occurrence graph (Phase 0 data-quality cleanup: PR #1203)."""

from __future__ import annotations

import asyncio

from app.storage import falkor_entity_relatedness


class _FakeFalkorClient:
    def __init__(self, rows_by_call=None, rows=None):
        # rows_by_call: list of row-lists returned in call order, for tests
        # that need different results per graph_query invocation (e.g.
        # bridging's direct-then-fallback-to-2-hop sequence). rows: a single
        # fixed return value for every call, for simpler tests.
        self._rows_by_call = list(rows_by_call) if rows_by_call is not None else None
        self._rows = rows
        self.calls = []

    def graph_query(self, cypher, params=None):
        self.calls.append((cypher, params))
        if self._rows_by_call is not None:
            return self._rows_by_call.pop(0) if self._rows_by_call else []
        return self._rows if self._rows is not None else []


def _run(coro):
    return asyncio.run(coro)


def test_fetch_related_entities_ranks_by_jaccard_not_raw_count(monkeypatch) -> None:
    """Live-verified shape: 'athena' has a higher raw shared-turn count than
    'tesla' against 'nvidia' but a much higher overall degree, so Jaccard
    ranks tesla above it -- this is the whole point of not ranking by raw
    count."""
    rows = [
        {"name": "tesla", "shared": 4, "degree1": 23, "degree2": 7, "jaccard": 0.153846},
        {"name": "athena", "shared": 4, "degree1": 23, "degree2": 32, "jaccard": 0.078431},
    ]
    fake_client = _FakeFalkorClient(rows=rows)
    monkeypatch.setattr(falkor_entity_relatedness, "get_recall_falkor_client", lambda: fake_client)

    out = _run(falkor_entity_relatedness.fetch_related_entities(name="nvidia", max_results=10))

    assert [r["name"] for r in out] == ["tesla", "athena"]
    assert out[0]["shared_turns"] == 4
    assert out[0]["jaccard"] > out[1]["jaccard"]

    cypher, params = fake_client.calls[0]
    assert "MENTIONS_ENTITY" in cypher
    assert params["name"] == "nvidia"
    assert params["max_results"] == 10


def test_fetch_related_entities_clamps_max_results(monkeypatch) -> None:
    fake_client = _FakeFalkorClient(rows=[])
    monkeypatch.setattr(falkor_entity_relatedness, "get_recall_falkor_client", lambda: fake_client)

    _run(falkor_entity_relatedness.fetch_related_entities(name="nvidia", max_results=9999))

    _, params = fake_client.calls[0]
    assert params["max_results"] == 50


def test_fetch_related_entities_none_client_returns_empty(monkeypatch) -> None:
    monkeypatch.setattr(falkor_entity_relatedness, "get_recall_falkor_client", lambda: None)
    out = _run(falkor_entity_relatedness.fetch_related_entities(name="nvidia"))
    assert out == []


def test_fetch_related_entities_failure_degrades_to_empty(monkeypatch) -> None:
    class _RaisingClient:
        def graph_query(self, *a, **k):
            raise RuntimeError("falkor down")

    monkeypatch.setattr(falkor_entity_relatedness, "get_recall_falkor_client", lambda: _RaisingClient())
    out = _run(falkor_entity_relatedness.fetch_related_entities(name="nvidia"))
    assert out == []


def test_fetch_related_entities_skips_unparseable_row_not_whole_request(monkeypatch) -> None:
    """Regression: numeric coercion used to happen outside the try/except
    wrapping the Falkor round-trip, so a malformed shared/jaccard value would
    raise uncaught through the /debug/entity-graph/related endpoint as an
    unhandled 500. Now a bad row is skipped, not the whole request."""
    rows = [
        {"name": "tesla", "shared": "not-a-number", "degree1": 23, "degree2": 7, "jaccard": 0.15},
        {"name": "atlas", "shared": 3, "degree1": 23, "degree2": 17, "jaccard": 0.08},
    ]
    fake_client = _FakeFalkorClient(rows=rows)
    monkeypatch.setattr(falkor_entity_relatedness, "get_recall_falkor_client", lambda: fake_client)

    out = _run(falkor_entity_relatedness.fetch_related_entities(name="nvidia"))

    assert [r["name"] for r in out] == ["atlas"]


def test_fetch_bridging_turns_prefers_direct_comention(monkeypatch) -> None:
    direct_rows = [{"turn_id": "t1", "ts": "2026-07-19T00:00:00"}]
    # Second call (the 2-hop fallback) should never actually run once direct
    # comes back non-empty -- rows_by_call would raise IndexError on pop if
    # it did, since only one entry is provided.
    fake_client = _FakeFalkorClient(rows_by_call=[direct_rows])
    monkeypatch.setattr(falkor_entity_relatedness, "get_recall_falkor_client", lambda: fake_client)

    out = _run(falkor_entity_relatedness.fetch_bridging_turns(entity_a="nvidia", entity_b="atlas"))

    assert out["mode"] == "direct"
    assert out["entity_a"] == "nvidia"
    assert out["entity_b"] == "atlas"
    assert out["results"] == [{"turn_id": "t1", "ts": "2026-07-19T00:00:00"}]
    assert len(fake_client.calls) == 1


def test_fetch_bridging_turns_falls_back_to_2hop_bridge(monkeypatch) -> None:
    bridge_rows = [{"bridge": "athena", "turn1": "t1", "turn2": "t2"}]
    fake_client = _FakeFalkorClient(rows_by_call=[[], bridge_rows])
    monkeypatch.setattr(falkor_entity_relatedness, "get_recall_falkor_client", lambda: fake_client)

    out = _run(falkor_entity_relatedness.fetch_bridging_turns(entity_a="nvidia", entity_b="atlas"))

    assert out["mode"] == "bridge"
    assert out["results"] == [{"bridge_entity": "athena", "turn1": "t1", "turn2": "t2"}]
    assert len(fake_client.calls) == 2


def test_fetch_bridging_turns_no_connection_returns_none_mode(monkeypatch) -> None:
    fake_client = _FakeFalkorClient(rows_by_call=[[], []])
    monkeypatch.setattr(falkor_entity_relatedness, "get_recall_falkor_client", lambda: fake_client)

    out = _run(falkor_entity_relatedness.fetch_bridging_turns(entity_a="nvidia", entity_b="unrelated"))

    assert out == {"mode": "none", "entity_a": "nvidia", "entity_b": "unrelated", "results": []}

    # The 2-hop Cypher must exclude t1==t2 -- otherwise a single ChatTurn
    # that happens to mention a, mid, and b together would be mislabeled as
    # a 2-hop bridge instead of what it actually is (a direct triple
    # co-mention, which the direct-check above would already have caught).
    second_cypher, _ = fake_client.calls[1]
    assert "t1 <> t2" in second_cypher


def test_fetch_bridging_turns_none_client_returns_none_mode(monkeypatch) -> None:
    monkeypatch.setattr(falkor_entity_relatedness, "get_recall_falkor_client", lambda: None)
    out = _run(falkor_entity_relatedness.fetch_bridging_turns(entity_a="a", entity_b="b"))
    assert out == {"mode": "none", "entity_a": "a", "entity_b": "b", "results": []}


def test_fetch_entity_mention_timeline_full_shape(monkeypatch) -> None:
    rows = [
        {"turn_id": "t2", "ts": "2026-07-19T00:00:00"},
        {"turn_id": "t1", "ts": "2026-07-18T00:00:00"},
    ]
    fake_client = _FakeFalkorClient(rows=rows)
    monkeypatch.setattr(falkor_entity_relatedness, "get_recall_falkor_client", lambda: fake_client)

    out = _run(falkor_entity_relatedness.fetch_entity_mention_timeline(name="nvidia"))

    assert out == [
        {"turn_id": "t2", "ts": "2026-07-19T00:00:00"},
        {"turn_id": "t1", "ts": "2026-07-18T00:00:00"},
    ]


def test_fetch_entity_mention_timeline_none_client_returns_empty(monkeypatch) -> None:
    monkeypatch.setattr(falkor_entity_relatedness, "get_recall_falkor_client", lambda: None)
    out = _run(falkor_entity_relatedness.fetch_entity_mention_timeline(name="nvidia"))
    assert out == []


def test_fetch_entity_matches_for_turns_full_shape(monkeypatch) -> None:
    rows = [
        {"tid": "turn-1", "matched": ["nvidia", "tesla"]},
        {"tid": "turn-2", "matched": []},  # no matches -- must be dropped, not kept as empty
    ]
    fake_client = _FakeFalkorClient(rows=rows)
    monkeypatch.setattr(falkor_entity_relatedness, "get_recall_falkor_client", lambda: fake_client)

    out = _run(
        falkor_entity_relatedness.fetch_entity_matches_for_turns(
            turn_ids=["turn-1", "turn-2"], target_names=["nvidia", "tesla"]
        )
    )

    assert out == {"turn-1": ["nvidia", "tesla"]}
    assert "turn-2" not in out

    _, params = fake_client.calls[0]
    assert params["turn_ids"] == ["turn-1", "turn-2"]
    assert params["target_names"] == ["nvidia", "tesla"]


def test_fetch_entity_matches_for_turns_empty_inputs_returns_empty(monkeypatch) -> None:
    fake_client = _FakeFalkorClient(rows=[])
    monkeypatch.setattr(falkor_entity_relatedness, "get_recall_falkor_client", lambda: fake_client)

    assert _run(falkor_entity_relatedness.fetch_entity_matches_for_turns(turn_ids=[], target_names=["nvidia"])) == {}
    assert _run(falkor_entity_relatedness.fetch_entity_matches_for_turns(turn_ids=["t1"], target_names=[])) == {}
    assert fake_client.calls == []


def test_fetch_entity_matches_for_turns_none_client_returns_empty(monkeypatch) -> None:
    monkeypatch.setattr(falkor_entity_relatedness, "get_recall_falkor_client", lambda: None)
    out = _run(
        falkor_entity_relatedness.fetch_entity_matches_for_turns(turn_ids=["t1"], target_names=["nvidia"])
    )
    assert out == {}
