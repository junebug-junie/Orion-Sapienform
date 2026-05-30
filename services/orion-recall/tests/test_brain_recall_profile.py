"""brain.recall.v1 profile aliasing, RDF enablement, and vector amputation."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

_REPO = Path(__file__).resolve().parents[3]
_RECALL_ROOT = _REPO / "services" / "orion-recall"
if str(_RECALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECALL_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app import worker
from app.profiles import get_profile, load_profiles


def test_recall_v1_alias_resolves_to_brain_recall_v1() -> None:
    load_profiles.cache_clear()
    prof = get_profile("recall.v1")
    assert prof["profile"] == "brain.recall.v1"
    assert prof.get("enable_vector") is False
    assert int(prof.get("vector_top_k", -1)) == 0
    assert prof.get("enable_rdf") is True
    assert prof.get("enable_rdf_expansion") is True
    assert prof.get("rdf_graphtri_mode") is True
    assert int(prof.get("rdf_top_k", 0)) > 0
    assert int(prof.get("rdf_expansion_top_k", 0)) > 0
    assert int(prof.get("rdf_connected_chat_top_k", 0)) > 0


def test_brain_profile_rdf_expansion_and_graphtri_flags() -> None:
    load_profiles.cache_clear()
    prof = get_profile("brain.recall.v1")
    assert prof.get("enable_rdf") is True
    assert prof.get("enable_rdf_expansion") is True
    assert prof.get("rdf_graphtri_mode") is True
    assert worker._rdf_expansion_enabled(prof) is True
    assert worker._rdf_graphtri_mode(prof) is True


def test_rdf_enabled_for_brain_profile_when_global_rdf_off(monkeypatch) -> None:
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_RDF", False)
    assert worker._rdf_enabled(
        {"profile": "brain.recall.v1", "enable_rdf": True, "rdf_top_k": 8}
    )


def test_rdf_disabled_for_chat_general_without_profile_flag(monkeypatch) -> None:
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_RDF", False)
    assert not worker._rdf_enabled(
        {"profile": "chat.general.v1", "enable_rdf": False, "rdf_top_k": 4}
    )


def test_brain_recall_query_backends_never_emit_vector(monkeypatch) -> None:
    load_profiles.cache_clear()
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_VECTOR", True)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_CHAT", False)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_TIMELINE", False)
    monkeypatch.setattr(worker.settings, "RECALL_RDF_ENDPOINT_URL", "")

    profile = get_profile("brain.recall.v1")
    candidates, counts = asyncio.run(
        worker._query_backends(
            "hello",
            profile,
            session_id=None,
            node_id=None,
            entities=[],
            diagnostic=True,
            exclusion={},
        )
    )
    assert counts.get("vector", 0) == 0
    assert not any(c.get("source") == "vector" for c in candidates)


def test_vector_amputation_grep_regression() -> None:
    app_dir = _RECALL_ROOT / "app"
    text = "\n".join(p.read_text(encoding="utf-8") for p in app_dir.rglob("*.py"))
    assert "fetch_vector" not in text
    assert "storage.vector_adapter" not in text


def test_brain_recall_query_backends_rdf_expansion(monkeypatch) -> None:
    load_profiles.cache_clear()
    monkeypatch.setattr(worker.settings, "RECALL_RDF_ENDPOINT_URL", "http://fake-rdf/sparql")
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_CHAT", False)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_TIMELINE", False)

    anchors_calls: list[dict] = []
    connected_calls: list[dict] = []
    graphtri_calls: list[dict] = []

    def _fake_anchors(**kwargs):
        anchors_calls.append(kwargs)
        return {
            "entities_terms": [],
            "tags_terms": [],
            "claim_objs": [],
            "related_terms": ["orion", "substrate"],
        }

    def _fake_connected(**kwargs):
        connected_calls.append(kwargs)
        return [
            {
                "id": "rdf-chat-hop-1",
                "source": "rdf_chat",
                "source_ref": "urn:chat:1",
                "text": "connected turn",
                "ts": None,
                "tags": ["graph_hop:1"],
                "score": 0.7,
            }
        ]

    def _fake_graphtri(**kwargs):
        graphtri_calls.append(kwargs)
        return [
            {
                "id": "rdf-claim-1",
                "source": "rdf",
                "source_ref": "urn:claim:1",
                "text": "claim fragment",
                "ts": None,
                "tags": ["claim"],
                "score": 0.6,
            }
        ]

    monkeypatch.setattr(worker, "fetch_rdf_chatturn_fragments", lambda **kw: [])
    monkeypatch.setattr(worker, "fetch_graphtri_anchors", _fake_anchors)
    monkeypatch.setattr(worker, "fetch_rdf_connected_chatturns", _fake_connected)
    monkeypatch.setattr(worker, "fetch_rdf_graphtri_fragments", _fake_graphtri)
    monkeypatch.setattr(
        worker,
        "fetch_rdf_fragments",
        lambda **kw: (_ for _ in ()).throw(AssertionError("generic rdf should not run when graphtri returns rows")),
    )

    profile = get_profile("brain.recall.v1")
    candidates, counts = asyncio.run(
        worker._query_backends(
            "orion substrate recall",
            profile,
            session_id=None,
            node_id=None,
            entities=[],
            diagnostic=True,
            exclusion={},
        )
    )

    assert len(anchors_calls) == 1
    assert anchors_calls[0]["session_id"] is None
    assert len(connected_calls) == 1
    assert connected_calls[0]["terms"] == ["orion", "substrate"]
    assert len(graphtri_calls) == 1

    assert counts.get("rdf_anchor_terms") == 2
    assert counts.get("rdf_connected_chat") == 1
    assert counts.get("rdf") == 1
    assert counts.get("vector", 0) == 0

    assert any("graph_hop:1" in (c.get("tags") or []) for c in candidates)
    assert any("claim" in (c.get("tags") or []) for c in candidates)
    assert not any(c.get("source") == "vector" for c in candidates)
