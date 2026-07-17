"""Tests for concept_region collector wiring into the live purposeful-recall
turn pipeline (Gap 1b of the concept-graph-pipeline design).

`services/orion-recall/tests/test_concept_region_collector.py` already covers
`fetch_concept_region_fragment()` in isolation against a real store. This file
covers the wiring itself: `worker.process_recall()` invoking the collector at
the right point, respecting `RECALL_CONCEPT_REGION_ENABLED` /
`pcr_backend_plan`, degrading gracefully on collector failure, and the
`get_substrate_store()` singleton's never-raise contract.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

_REPO = Path(__file__).resolve().parents[3]
_RECALL_ROOT = _REPO / "services" / "orion-recall"
if str(_RECALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECALL_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app import worker
from app.profiles import get_profile
from app.substrate_store import get_substrate_store
from orion.core.contracts.recall import MemoryBundleStatsV1, MemoryBundleV1, RecallQueryV1


def _base_worker_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker.settings, "RECALL_PCR_ENABLED", True)
    monkeypatch.setattr(worker.settings, "RECALL_ACTIVE_PACKET_ENABLED", False)
    monkeypatch.setattr(worker.settings, "RECALL_BELIEF_RENDER_BUDGET", 128)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_CHAT", False)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_TIMELINE", False)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_RDF", False)
    monkeypatch.setattr(worker.settings, "RECALL_INTENT_ROUTING_ENABLED", False)


def _wire_common_mocks(monkeypatch: pytest.MonkeyPatch, *, belief_fuse_called: list) -> None:
    belief_profile = get_profile("chat.belief.semantic.v1")
    monkeypatch.setattr(worker, "get_profile", lambda _name: dict(belief_profile))

    async def _anchor(**kwargs):
        return [], {}

    def _mock_fuse(**kwargs):
        return MemoryBundleV1(rendered="fuse", stats=MemoryBundleStatsV1()), []

    def _mock_belief_fuse(**kwargs):
        belief_fuse_called.append(kwargs)
        return MemoryBundleV1(rendered="belief ok", stats=MemoryBundleStatsV1()), []

    monkeypatch.setattr(worker, "_fetch_anchor_candidates", _anchor)
    monkeypatch.setattr(worker, "_query_backends", AsyncMock(return_value=([], {})))
    monkeypatch.setattr(worker, "fuse_candidates", _mock_fuse)
    monkeypatch.setattr(worker, "pcr_fuse_belief_candidates", _mock_belief_fuse)


def _purposeful_query() -> RecallQueryV1:
    return RecallQueryV1(
        fragment="tell me about continuity and self-modeling",
        profile="chat.belief.semantic.v1",
        recall_phase="purposeful",
        retrieval_intent="semantic",
        task_hints={"rule_id": "topic_shift"},
        session_id="sess-concept-region",
    )


def test_worker_invokes_concept_region_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _base_worker_settings(monkeypatch)
    monkeypatch.setattr(worker.settings, "RECALL_CONCEPT_REGION_ENABLED", True)

    belief_fuse_called: list = []
    _wire_common_mocks(monkeypatch, belief_fuse_called=belief_fuse_called)

    concept_region_called: list = []

    def _mock_concept_region(query, *, store):
        concept_region_called.append((query, store))
        return [{"id": "cr-1", "source": "concept_region", "snippet": "Orion: continuity", "score": 0.7}]

    monkeypatch.setattr(worker, "fetch_concept_region_fragment", _mock_concept_region)
    monkeypatch.setattr(worker, "get_substrate_store", lambda: "fake-store-handle")

    bundle, decision = asyncio.run(worker.process_recall(_purposeful_query(), corr_id="corr-cr-1", diagnostic=True))

    assert concept_region_called
    assert concept_region_called[0][1] == "fake-store-handle"
    assert belief_fuse_called
    passed_candidates = belief_fuse_called[0]["candidates"]
    assert any(c.get("source") == "concept_region" for c in passed_candidates)
    assert bundle.rendered == "belief ok"
    assert decision.recall_debug["pcr"]["backend_plan"]
    assert "concept_region" in decision.recall_debug["pcr"]["backend_plan"]


def test_worker_skips_concept_region_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _base_worker_settings(monkeypatch)
    monkeypatch.setattr(worker.settings, "RECALL_CONCEPT_REGION_ENABLED", False)

    belief_fuse_called: list = []
    _wire_common_mocks(monkeypatch, belief_fuse_called=belief_fuse_called)

    concept_region_called: list = []

    def _mock_concept_region(query, *, store):
        concept_region_called.append((query, store))
        return [{"id": "cr-1", "source": "concept_region", "snippet": "should not appear", "score": 0.7}]

    monkeypatch.setattr(worker, "fetch_concept_region_fragment", _mock_concept_region)

    asyncio.run(worker.process_recall(_purposeful_query(), corr_id="corr-cr-2", diagnostic=True))

    assert concept_region_called == []


def test_worker_concept_region_failure_degrades_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    """A collector-level exception must not crash the turn -- the route must
    still render a bundle, just without concept_region's fragments."""
    _base_worker_settings(monkeypatch)
    monkeypatch.setattr(worker.settings, "RECALL_CONCEPT_REGION_ENABLED", True)

    belief_fuse_called: list = []
    _wire_common_mocks(monkeypatch, belief_fuse_called=belief_fuse_called)

    def _raising_concept_region(query, *, store):
        raise RuntimeError("store unreachable")

    monkeypatch.setattr(worker, "fetch_concept_region_fragment", _raising_concept_region)
    monkeypatch.setattr(worker, "get_substrate_store", lambda: None)

    bundle, decision = asyncio.run(worker.process_recall(_purposeful_query(), corr_id="corr-cr-3", diagnostic=True))

    assert bundle.rendered == "belief ok"
    passed_candidates = belief_fuse_called[0]["candidates"]
    assert not any(c.get("source") == "concept_region" for c in passed_candidates)


def test_get_substrate_store_never_raises_on_init_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.substrate_store as substrate_store_mod

    monkeypatch.setattr(substrate_store_mod, "_STORE", None)

    def _boom():
        raise RuntimeError("backend unreachable")

    monkeypatch.setattr(substrate_store_mod, "build_substrate_store_from_env", _boom)

    assert substrate_store_mod.get_substrate_store() is None


def test_get_substrate_store_caches_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.substrate_store as substrate_store_mod

    monkeypatch.setattr(substrate_store_mod, "_STORE", None)

    calls: list[int] = []

    def _fake_build():
        calls.append(1)
        return object()

    monkeypatch.setattr(substrate_store_mod, "build_substrate_store_from_env", _fake_build)

    first = get_substrate_store()
    second = substrate_store_mod.get_substrate_store()

    assert first is second
    assert len(calls) == 1
