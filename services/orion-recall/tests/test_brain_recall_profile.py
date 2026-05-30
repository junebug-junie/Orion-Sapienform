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
    assert int(prof.get("rdf_top_k", 0)) > 0


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
