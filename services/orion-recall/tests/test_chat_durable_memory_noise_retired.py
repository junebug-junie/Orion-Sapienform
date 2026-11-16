"""Issue #3461: the Juniper chat prompt's DURABLE MEMORY must not carry
AI Town chatter, bus-latency anomalies, active_packet digests, or
topic-foundry concepts. Each test pins one cut at the seam that makes it.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

_REPO = Path(__file__).resolve().parents[3]
_RECALL_ROOT = _REPO / "services" / "orion-recall"
if str(_RECALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECALL_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app import sql_chat, worker
from app.collectors import concept_region as cr
from app.fusion import pcr_fuse_belief_candidates, render_continuity_bundle
from app.pcr_collectors import apply_chat_memory_exclusions
from app.profiles import get_profile
from orion.core.contracts.recall import MemoryBundleStatsV1, MemoryBundleV1, RecallQueryV1
from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    NodeRefV1,
    SubstrateEdgeV1,
    SubstrateProvenanceV1,
    SubstrateTemporalWindowV1,
)
from orion.substrate.store import InMemorySubstrateGraphStore


# ── AI Town: sql_chat union ──────────────────────────────────────────────────


def _capture_sql(monkeypatch) -> Dict[str, str]:
    captured: Dict[str, str] = {}

    class _Conn:
        async def fetch(self, query):
            captured["query"] = query
            return []

        async def close(self):
            return None

    async def _connect(_dsn):
        return _Conn()

    monkeypatch.setattr(sql_chat, "asyncpg", type("_AsyncPg", (), {"connect": _connect}))
    return captured


def test_sql_chat_without_aitown_scans_the_hub_table_only(monkeypatch):
    captured = _capture_sql(monkeypatch)
    asyncio.run(sql_chat.fetch_chat_history_pairs(limit=6, since_minutes=120, include_aitown=False))
    q = captured["query"]
    assert sql_chat.settings.RECALL_SQL_AITOWN_CHAT_TABLE not in q
    assert "UNION" not in q
    assert f"FROM {sql_chat.settings.RECALL_SQL_CHAT_TABLE}" in q
    assert "ORDER BY created_at DESC" in q and "LIMIT 6" in q


def test_sql_chat_default_still_unions_aitown_for_other_profiles(monkeypatch):
    captured = _capture_sql(monkeypatch)
    asyncio.run(sql_chat.fetch_chat_history_pairs(limit=6, since_minutes=120))
    assert sql_chat.settings.RECALL_SQL_AITOWN_CHAT_TABLE in captured["query"]
    assert "UNION ALL" in captured["query"]


def test_chat_exclusions_turn_off_bus_anomaly_and_aitown_without_mutating_input():
    base = get_profile("chat.continuity.v1")
    out = apply_chat_memory_exclusions(base)
    assert out["enable_bus_synaptic_anomaly"] is False
    assert out["sql_chat_include_aitown"] is False
    assert "enable_bus_synaptic_anomaly" not in base


# ── worker: both PCR chat phases, and a non-chat profile ────────────────────


def _wire_worker(monkeypatch, profile_name: str) -> Dict[str, List[Any]]:
    s = worker.settings
    monkeypatch.setattr(s, "RECALL_PCR_ENABLED", True)
    monkeypatch.setattr(s, "RECALL_BUS_SYNAPTIC_ANOMALY_IN_CHAT", True)
    monkeypatch.setattr(s, "RECALL_ENABLE_SQL_CHAT", True)
    monkeypatch.setattr(s, "RECALL_ENABLE_SQL_TIMELINE", False)
    monkeypatch.setattr(s, "RECALL_ENABLE_RDF", False)
    monkeypatch.setattr(s, "RECALL_FALKOR_IN_CHAT", False)
    monkeypatch.setattr(s, "RECALL_CONCEPT_REGION_ENABLED", False)
    monkeypatch.setattr(s, "RECALL_ENABLE_CARDS", False)
    monkeypatch.setattr(s, "RECALL_INTENT_ROUTING_ENABLED", False)

    profile = dict(get_profile(profile_name))
    profile["sql_chat_top_k"] = 6
    profile.pop("enable_sql_chat", None)
    monkeypatch.setattr(worker, "get_profile", lambda _name: dict(profile))

    calls: Dict[str, List[Any]] = {"bus": [], "pairs": []}

    async def _bus(**kwargs):
        calls["bus"].append(kwargs)
        return [{"id": "bus-1", "source": "bus_synaptic_anomaly", "text": "hop latency |z|=58.9", "score": 0.9}]

    async def _pairs(**kwargs):
        calls["pairs"].append(kwargs)
        return []

    async def _anchor(**kwargs):
        return [], {}

    monkeypatch.setattr(worker, "fetch_bus_synaptic_anomaly_fragments", _bus)
    monkeypatch.setattr(worker, "fetch_chat_history_pairs", _pairs)
    monkeypatch.setattr(worker, "fetch_chat_messages", AsyncMock(return_value=[]))
    monkeypatch.setattr(worker, "_fetch_anchor_candidates", _anchor)
    monkeypatch.setattr(
        worker,
        "fuse_candidates",
        lambda **kw: (MemoryBundleV1(rendered="", stats=MemoryBundleStatsV1()), []),
    )
    return calls


@pytest.mark.parametrize(
    "phase,profile_name,intent",
    [("continuity", "chat.continuity.v1", "continuity"), ("purposeful", "chat.belief.open_loop.v1", "open_loop")],
)
def test_pcr_chat_phases_never_fetch_bus_anomaly_or_aitown(monkeypatch, phase, profile_name, intent):
    calls = _wire_worker(monkeypatch, profile_name)
    q = RecallQueryV1(
        fragment="what's happening out on the street",
        profile=profile_name,
        recall_phase=phase,
        retrieval_intent=intent,
        session_id="s-3461",
    )
    bundle, decision = asyncio.run(worker.process_recall(q, corr_id=f"c-3461-{phase}"))
    assert calls["bus"] == []
    assert "bus_synaptic_anomaly" not in bundle.rendered
    assert "bus_synaptic_anomaly" not in (decision.backend_counts or {})
    if phase == "continuity":  # purposeful disables sql_chat in its collector plan
        assert calls["pairs"], "sql_chat feed was never reached"
    for kw in calls["pairs"]:
        assert kw["include_aitown"] is False


def test_non_chat_profile_keeps_bus_anomaly_and_aitown(monkeypatch):
    """Other recall consumers (reflect, social rooms, journals) are untouched."""
    calls = _wire_worker(monkeypatch, "reflect.v1")
    q = RecallQueryV1(fragment="what's happening out on the street", profile="reflect.v1", session_id="s-3461")
    asyncio.run(worker.process_recall(q, corr_id="c-3461-reflect"))
    assert calls["bus"], "reflect.v1 should still fetch the bus anomaly feed"
    assert calls["pairs"] and all(kw["include_aitown"] is True for kw in calls["pairs"])


# ── topic-foundry concepts ──────────────────────────────────────────────────


def _prov(key: str) -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="human_verified",
        source_kind="test_fixture",
        source_channel="test:3461",
        producer="test_chat_durable_memory_noise_retired",
        evidence_refs=[f"fixture#{key}"],
    )


def _node(node_id: str, label: str) -> ConceptNodeV1:
    return ConceptNodeV1(
        node_id=node_id,
        anchor_scope="orion",
        subject_ref=node_id,
        promotion_state="canonical",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc)),
        provenance=_prov(node_id),
        label=label,
        definition=f"Definition of {label}",
    )


def _edge(a: str, b: str) -> SubstrateEdgeV1:
    return SubstrateEdgeV1(
        edge_id=f"edge-{a}-{b}",
        source=NodeRefV1(node_id=a, node_kind="concept"),
        target=NodeRefV1(node_id=b, node_kind="concept"),
        predicate="associated_with",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc)),
        provenance=_prov(f"{a}-{b}"),
    )


def test_concept_region_drops_topic_foundry_nodes_and_their_edges():
    store = InMemorySubstrateGraphStore()
    seed = "sub-concept-seed-orion"
    tf = "sub-concept-topicfoundry-12ccd29e-82d0-4e1f-9adb-1209e1e419fe-4"
    store.upsert_node(identity_key="concept|orion|seed", node=_node(seed, "Orion"))
    store.upsert_node(identity_key="concept|tf|github", node=_node(tf, "github"))
    store.upsert_edge(identity_key=f"{tf}|assoc|{seed}", edge=_edge(tf, seed))

    frags = cr.fetch_concept_region_fragment("Orion, look at the github PR", store=store)

    text = " ".join(f["snippet"] for f in frags)
    assert "Orion" in text  # the seed concept still surfaces
    assert "topicfoundry" not in text
    assert all("github" not in f["snippet"] for f in frags)


def test_topic_foundry_producer_still_uses_the_filtered_prefix():
    """If the producer's id scheme drifts, the filter silently stops working."""
    from orion.substrate.adapters.topic_foundry import map_topic_foundry_run_to_substrate

    out = map_topic_foundry_run_to_substrate(
        run_id="run-3461",
        topics=[{"topic_id": 3, "count": 50, "outlier_pct": 0.1, "label": "github"}],
        keywords_by_topic={3: ["github", "pr"]},
    )
    concept_ids = [n.node_id for n in out.nodes if n.node_kind == "concept"]
    assert concept_ids
    assert all(cr._is_topic_foundry_id(nid) for nid in concept_ids)


# ── empty digests render nothing, not a bare header ─────────────────────────


def test_empty_belief_digest_renders_no_header():
    bundle, _ = pcr_fuse_belief_candidates(
        candidates=[], profile=dict(get_profile("chat.belief.open_loop.v1")), retrieval_intent="open_loop"
    )
    assert bundle.rendered == ""


def test_empty_continuity_digest_renders_no_header():
    bundle, _ = render_continuity_bundle(candidates=[], profile=dict(get_profile("chat.continuity.v1")))
    assert bundle.rendered == ""
