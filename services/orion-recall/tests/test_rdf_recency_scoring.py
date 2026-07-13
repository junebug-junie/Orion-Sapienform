"""RDF-backed recall fragments must score by real recency, not a frozen UUID sort.

fetch_rdf_chatturn_fragments / fetch_rdf_graphtri_fragments used to order candidates with
``ORDER BY DESC(STR(?turn))`` (a lexical sort on the ChatTurn URI) and hardcode ``ts: 0.0`` on
every returned fragment. That meant the same fixed set of historical turns always won, and
scoring._compute_recency_factor always fell back to its neutral 0.5 weight regardless of age.

rdf_builder.py (services/orion-rdf-writer/app/rdf_builder.py:377, :608) already writes a real
``ORION.timestamp`` literal on ChatTurn/Claim nodes; these tests lock in that the adapter now
selects and orders on it, and that the parsed ``ts`` feeds real recency decay.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app import scoring
from app.storage import rdf_adapter


def _mock_post(monkeypatch: pytest.MonkeyPatch, bindings: list[dict], captured: dict | None = None):
    def _fake_post(url, data=None, **kwargs):
        if captured is not None:
            captured["sparql"] = data
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"results": {"bindings": bindings}}
        return response

    monkeypatch.setattr(rdf_adapter.requests, "post", _fake_post)


def test_rdf_chatturn_fragments_score_by_recency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rdf_adapter.settings, "RECALL_RDF_ENDPOINT_URL", "http://orion-athena-fuseki:3030/orion/query"
    )

    old_ts = "2020-01-01T00:00:00"
    new_ts = "2026-07-13T00:00:00"
    bindings = [
        {
            "turn": {"value": "http://conjourney.net/orion/chatTurn/old"},
            "prompt": {"value": "old prompt"},
            "response": {"value": "old response"},
            "ts": {"value": old_ts},
        },
        {
            "turn": {"value": "http://conjourney.net/orion/chatTurn/new"},
            "prompt": {"value": "new prompt"},
            "response": {"value": "new response"},
            "ts": {"value": new_ts},
        },
    ]
    _mock_post(monkeypatch, bindings)

    frags = rdf_adapter.fetch_rdf_chatturn_fragments(query_text="hello", session_id=None, max_items=10)
    assert len(frags) == 2

    old_frag = next(f for f in frags if "old" in f["id"])
    new_frag = next(f for f in frags if "new" in f["id"])

    # ts must be parsed into real, distinct epoch floats -- not the hardcoded 0.0 placeholder.
    assert old_frag["ts"] > 0.0
    assert new_frag["ts"] > 0.0
    assert new_frag["ts"] > old_frag["ts"]

    for mode in ("short_term", "hybrid"):
        old_score = scoring._compute_recency_factor(old_frag["ts"], mode)
        new_score = scoring._compute_recency_factor(new_frag["ts"], mode)
        assert old_score < new_score, f"mode={mode} old={old_score} new={new_score}"


def test_rdf_graphtri_fragments_score_by_recency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rdf_adapter.settings, "RECALL_RDF_ENDPOINT_URL", "http://orion-athena-fuseki:3030/orion/query"
    )

    old_ts = "2020-01-01T00:00:00"
    new_ts = "2026-07-13T00:00:00"
    bindings = [
        {
            "turn": {"value": "http://conjourney.net/orion/chatTurn/old"},
            "claim": {"value": "http://conjourney.net/orion/claim/old"},
            "pred": {"value": "http://conjourney.net/orion#likes"},
            "obj": {"value": "coffee"},
            "ts": {"value": old_ts},
        },
        {
            "turn": {"value": "http://conjourney.net/orion/chatTurn/new"},
            "claim": {"value": "http://conjourney.net/orion/claim/new"},
            "pred": {"value": "http://conjourney.net/orion#likes"},
            "obj": {"value": "tea"},
            "ts": {"value": new_ts},
        },
    ]
    _mock_post(monkeypatch, bindings)

    frags = rdf_adapter.fetch_rdf_graphtri_fragments(query_text="hello", session_id="s1", max_items=10)
    assert len(frags) == 2

    old_frag = next(f for f in frags if "old" in f["id"])
    new_frag = next(f for f in frags if "new" in f["id"])

    assert old_frag["ts"] > 0.0
    assert new_frag["ts"] > old_frag["ts"]

    for mode in ("short_term", "hybrid"):
        old_score = scoring._compute_recency_factor(old_frag["ts"], mode)
        new_score = scoring._compute_recency_factor(new_frag["ts"], mode)
        assert old_score < new_score, f"mode={mode} old={old_score} new={new_score}"


def test_rdf_fragment_ordering_uses_timestamp_not_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rdf_adapter.settings, "RECALL_RDF_ENDPOINT_URL", "http://orion-athena-fuseki:3030/orion/query"
    )

    captured: dict = {}
    _mock_post(monkeypatch, [], captured)
    rdf_adapter.fetch_rdf_chatturn_fragments(query_text="hello", session_id=None, max_items=5)
    chatturn_sparql = captured["sparql"]
    assert "ORDER BY DESC(?ts)" in chatturn_sparql
    assert "STR(?turn)" not in chatturn_sparql

    captured.clear()
    _mock_post(monkeypatch, [], captured)
    rdf_adapter.fetch_rdf_graphtri_fragments(query_text="hello", session_id="s1", max_items=5)
    graphtri_sparql = captured["sparql"]
    assert "ORDER BY DESC(?ts)" in graphtri_sparql
    assert "STR(?turn)" not in graphtri_sparql


def test_parse_rdf_timestamp_handles_epoch_iso_space_and_junk() -> None:
    assert rdf_adapter._parse_rdf_timestamp(None) == 0.0
    assert rdf_adapter._parse_rdf_timestamp("") == 0.0
    assert rdf_adapter._parse_rdf_timestamp("not-a-timestamp") == 0.0
    assert rdf_adapter._parse_rdf_timestamp("1783300000.0") == 1783300000.0
    # space-separated (chat_history.py normalizes these before writing) and "Z"-suffixed variants.
    iso_t = rdf_adapter._parse_rdf_timestamp("2026-07-13T00:00:00")
    iso_space = rdf_adapter._parse_rdf_timestamp("2026-07-13 00:00:00")
    iso_z = rdf_adapter._parse_rdf_timestamp("2026-07-13T00:00:00Z")
    assert iso_t == iso_space
    assert iso_t > 0.0
    assert iso_z > 0.0
