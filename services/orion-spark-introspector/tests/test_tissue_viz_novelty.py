from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

import app.worker as worker
from app.substrate_reads import (
    ExecutionTrajectorySnapshot,
    GrammarTruthSnapshot,
    ReasoningActivitySnapshot,
)
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1

_NOW = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)


def _dim(name: str, score: float) -> SelfStateDimensionV1:
    return SelfStateDimensionV1(dimension_id=name, score=score, confidence=1.0)


def _self_state_payload(*, coherence: float = 1.0) -> dict:
    """Reproduces the live incident's exact dimension readings.

    The `uncertainty` dimension score here is set directly to 0.0, matching
    what orion-self-state-runtime actually publishes: its own
    uncertainty_score(overall_salience, coherence) = salience * (1 -
    coherence) formula already computed to exactly 0 upstream (that service's
    internal coherence was saturated at 1.0). _phi_from_self_state() in THIS
    service just consumes that already-zero raw score directly -- it does not
    re-derive uncertainty from coherence itself. Its own local `coherence`
    variable (field_coh * (1-continuity_pressure) * (1-reliability_pressure)
    * transport, all ** 0.25) is a separate recomputation used only as a
    0.3-1.0 scaling multiplier on (0.6*uncertainty + 0.4*introspection) --
    it cannot zero novelty on its own (floor 0.3), so it's not the
    mechanism under test here; introspection_pressure=0.0 is the second
    always-dead input (no channel ever feeds it, see
    docs/superpowers -- structurally sparse, not wired to anything)."""
    dims = {
        name: _dim(name, score)
        for name, score in (
            ("coherence", coherence),
            ("field_intensity", 0.6),
            ("agency_readiness", 0.5),
            ("execution_pressure", 0.2),
            ("reasoning_pressure", 0.1),
            ("resource_pressure", 0.3),
            ("reliability_pressure", 0.0),
            ("continuity_pressure", 0.0),
            ("social_pressure", 0.0),
            ("introspection_pressure", 0.0),
            ("uncertainty", 0.0),
            ("policy_pressure", 0.0),
            ("transport_integrity", 1.0),
        )
    }
    return SelfStateV1(
        self_state_id="self.state:tick_novelty_test:policy.v1",
        generated_at=_NOW,
        source_field_tick_id="tick_novelty_test",
        source_field_generated_at=_NOW,
        source_attention_frame_id="frame_novelty_test",
        source_attention_generated_at=_NOW,
        overall_intensity=0.5,
        overall_confidence=0.8,
        overall_condition="steady",
        dimensions=dims,
        dimension_trajectory={},
    ).model_dump(mode="json")


def _mock_healthy_substrate_reads(monkeypatch) -> None:
    monkeypatch.setattr(worker, "_SUBSTRATE_CACHE", None, raising=False)
    monkeypatch.setattr(
        worker,
        "fetch_grammar_truth",
        AsyncMock(
            return_value=GrammarTruthSnapshot(
                degraded=False,
                degraded_reasons=[],
                enabled_reducers={"execution_trajectory": True},
                reducer_health_by_name={"execution_trajectory": {"classification": "healthy"}},
            )
        ),
    )
    monkeypatch.setattr(
        worker,
        "fetch_execution_trajectory",
        AsyncMock(return_value=ExecutionTrajectorySnapshot(ok=True, projection=None)),
    )
    monkeypatch.setattr(
        worker,
        "fetch_reasoning_activity",
        AsyncMock(return_value=ReasoningActivitySnapshot(ok=False, projection=None)),
    )


def setup_function(_fn) -> None:
    worker._LAST_EMBEDDING_NOVELTY = None


def teardown_function(_fn) -> None:
    worker._LAST_EMBEDDING_NOVELTY = None


def test_novelty_stat_defaults_to_zero_when_never_set() -> None:
    assert worker._LAST_EMBEDDING_NOVELTY is None
    assert worker._novelty_stat() == 0.0


def test_novelty_stat_reflects_last_embedding_value() -> None:
    worker._LAST_EMBEDDING_NOVELTY = 0.42
    assert worker._novelty_stat() == pytest.approx(0.42)


def test_phi_from_self_state_novelty_is_zero_with_live_incident_dimensions() -> None:
    """Confirms the actual bug mechanism directly, with the exact dimension
    readings observed in the live incident: _phi_from_self_state()["novelty"]
    = (0.6*uncertainty + 0.4*introspection_pressure) * (0.3 + 0.7*coherence).
    With uncertainty=0.0 and introspection_pressure=0.0 (both real upstream
    readings, not missing/defaulted -- see _self_state_payload's docstring),
    the weighted-sum term is exactly 0 regardless of the coherence multiplier,
    which is why _phi_from_self_state()["novelty"] must never be used for the
    live tissue-viz display."""
    ss = SelfStateV1.model_validate(_self_state_payload(coherence=1.0))
    phi = worker._phi_from_self_state(ss)
    assert phi["novelty"] == 0.0


@pytest.mark.asyncio
async def test_handle_semantic_upsert_updates_last_embedding_novelty(monkeypatch) -> None:
    monkeypatch.setattr(worker, "_EXPECTED_EMB", {}, raising=False)
    monkeypatch.setattr(worker, "_SEEN_DOC", {}, raising=False)
    monkeypatch.setattr(worker, "_pub_bus", None, raising=False)
    broadcasts = []

    async def _capture_broadcast(payload):
        broadcasts.append(payload)

    monkeypatch.setattr(worker.manager, "broadcast", _capture_broadcast, raising=False)

    env = BaseEnvelope(
        kind="vector.upsert.v1",
        source=ServiceRef(name="orion-vector-host", node="n1"),
        correlation_id=uuid4(),
        payload={
            "doc_id": "turn-novelty-1",
            "collection": "orion_chat_turns",
            "embedding": [1.0, 0.0, 0.0, 0.0],
            "embedding_kind": "semantic",
            "text": "hello world",
            "meta": {},
        },
    )

    assert worker._LAST_EMBEDDING_NOVELTY is None
    await worker.handle_semantic_upsert(env)

    # First embedding on a fresh channel_key is the cold-start branch:
    # novelty=1.0 (no prior expectation to compare against).
    assert worker._LAST_EMBEDDING_NOVELTY == pytest.approx(1.0)
    tissue = [b for b in broadcasts if b.get("type") == "tissue.update"]
    assert tissue, "expected a tissue.update broadcast"
    assert tissue[-1]["stats"]["novelty"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_handle_self_state_tissue_update_uses_embedding_novelty_not_uncertainty(
    monkeypatch,
) -> None:
    """Regression for the live incident: with coherence saturated at 1.0 (so
    the SelfStateV1-derived novelty is structurally 0), the self-state-tick
    tissue.update broadcast must show the last real chat-triggered embedding
    novelty instead -- not 0."""
    monkeypatch.setattr(worker, "_SUBSTRATE_CACHE", None, raising=False)
    _mock_healthy_substrate_reads(monkeypatch)
    monkeypatch.setattr(worker, "_LAST_EMBEDDING_NOVELTY", 0.37, raising=False)

    broadcasts = []

    async def _capture_broadcast(payload):
        broadcasts.append(payload)

    class _Bus:
        enabled = True

        async def publish(self, channel, env):
            pass

    monkeypatch.setattr(worker, "_pub_bus", _Bus(), raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", _capture_broadcast, raising=False)
    monkeypatch.setattr(worker, "_INNER_SCALER", worker._new_inner_scaler(), raising=False)
    monkeypatch.setattr(worker, "_INNER_PREV_FELT", None, raising=False)
    monkeypatch.setattr(worker, "_INNER_PREV_HEADLINE", None, raising=False)
    monkeypatch.setattr(worker, "_INNER_DEGENERATE_STREAK", 0, raising=False)
    monkeypatch.setattr(worker, "_INNER_SINK", AsyncMock(append=lambda *_a, **_k: None), raising=False)

    env = BaseEnvelope(
        kind="substrate.self_state.v1",
        source=ServiceRef(name="substrate-runtime", node="athena"),
        payload=_self_state_payload(coherence=1.0),
    )
    await worker.handle_self_state(env)

    tissue = [b for b in broadcasts if b.get("type") == "tissue.update"]
    assert tissue, "expected a tissue.update broadcast"
    assert tissue[-1]["stats"]["novelty"] == pytest.approx(0.37)
