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

_NOW = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)


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
async def test_inner_state_tick_tissue_update_uses_embedding_novelty(
    monkeypatch,
) -> None:
    """2026-07-22 (SelfStateV1 burn): was
    test_handle_self_state_tissue_update_uses_embedding_novelty_not_uncertainty,
    a regression for a live incident where a SelfStateV1-derived novelty
    formula structurally pinned at 0. That formula (_phi_from_self_state) no
    longer exists -- run_inner_state_tick() uses TISSUE.phi() as its
    heuristic baseline now. The surviving assertion worth keeping: the
    tissue.update broadcast's novelty must come from the last real
    chat-triggered embedding novelty (_LAST_EMBEDDING_NOVELTY), not whatever
    phi_stats["novelty"] happens to read."""
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

    await worker.run_inner_state_tick()

    tissue = [b for b in broadcasts if b.get("type") == "tissue.update"]
    assert tissue, "expected a tissue.update broadcast"
    assert tissue[-1]["stats"]["novelty"] == pytest.approx(0.37)
