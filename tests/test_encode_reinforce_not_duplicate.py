from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orion.memory.crystallization.intake_pipeline import process_consolidation_crystallization
from orion.memory.crystallization.projector import ProjectionConfig
from orion.memory.crystallization.proposer import propose
from orion.memory.crystallization.schemas import (
    CrystallizationEvidenceRefV1,
    MemoryCrystallizationProposeRequestV1,
    new_crystallization_id,
)


class _Settings:
    MEMORY_FORMATION_AUTO_ACTIVATE_ENABLED = False
    MEMORY_FORMATION_AUTO_ENCODE_ACTIVATION_RATIO = 0.4
    SERVICE_NAME = "orion-memory-consolidation"
    SERVICE_VERSION = "0.1.0"
    NODE_NAME = "test"


def _semantic_proposed():
    req = MemoryCrystallizationProposeRequestV1(
        kind="semantic",
        subject="Deploy plan",
        summary="We chose k3s for staging cluster rollout",
        scope=["project:orion"],
        evidence=[CrystallizationEvidenceRefV1(source_kind="chat_turn", source_id="corr-a")],
        proposed_by="test",
    )
    return propose(req)


@pytest.mark.asyncio
async def test_reinforce_existing_skips_insert(monkeypatch):
    existing = _semantic_proposed()
    existing.crystallization_id = new_crystallization_id()
    existing.status = "active"

    candidate = existing.model_copy(deep=True)
    candidate.crystallization_id = new_crystallization_id()
    candidate.evidence.append(
        CrystallizationEvidenceRefV1(source_kind="chat_turn", source_id="corr-2", strength=0.75)
    )

    pool = MagicMock()
    bus = MagicMock(enabled=True)
    bus.publish = AsyncMock()

    list_mock = AsyncMock(return_value=[existing])
    update_mock = AsyncMock()
    insert_mock = AsyncMock()
    emit_mock = AsyncMock(return_value=True)

    monkeypatch.setattr(
        "orion.memory.crystallization.intake_pipeline.list_crystallizations",
        list_mock,
    )
    monkeypatch.setattr(
        "orion.memory.crystallization.intake_pipeline.update_crystallization",
        update_mock,
    )
    monkeypatch.setattr(
        "orion.memory.crystallization.intake_pipeline.insert_crystallization",
        insert_mock,
    )
    monkeypatch.setattr(
        "orion.memory.crystallization.intake_pipeline.emit_crystallization_lifecycle",
        emit_mock,
    )

    cid, final_row, outcome = await process_consolidation_crystallization(
        pool,
        bus,
        crystallization=candidate,
        settings=_Settings(),
        project_config=ProjectionConfig(),
    )

    assert outcome == "reinforced"
    assert cid == existing.crystallization_id
    assert final_row.dynamics.reinforcement_count == 1
    assert any(ev.source_id == "corr-2" for ev in final_row.evidence)
    insert_mock.assert_not_called()
    update_mock.assert_called_once()
    list_mock.assert_called_once_with(pool, status=None, limit=200)
    emit_mock.assert_called_once()
    assert emit_mock.call_args.kwargs["lifecycle"] == "reinforced"


@pytest.mark.asyncio
async def test_non_duplicate_inserts_proposed(monkeypatch):
    existing = _semantic_proposed()
    existing.crystallization_id = new_crystallization_id()
    existing.status = "active"

    candidate = _semantic_proposed()
    candidate.summary = "We chose Nomad for staging cluster rollout"
    candidate.subject = candidate.summary

    pool = MagicMock()
    bus = MagicMock(enabled=True)

    list_mock = AsyncMock(return_value=[existing])
    update_mock = AsyncMock()
    insert_mock = AsyncMock(return_value=candidate.crystallization_id)
    emit_mock = AsyncMock(return_value=True)

    monkeypatch.setattr(
        "orion.memory.crystallization.intake_pipeline.list_crystallizations",
        list_mock,
    )
    monkeypatch.setattr(
        "orion.memory.crystallization.intake_pipeline.update_crystallization",
        update_mock,
    )
    monkeypatch.setattr(
        "orion.memory.crystallization.intake_pipeline.insert_crystallization",
        insert_mock,
    )
    monkeypatch.setattr(
        "orion.memory.crystallization.intake_pipeline.emit_crystallization_lifecycle",
        emit_mock,
    )

    cid, final_row, outcome = await process_consolidation_crystallization(
        pool,
        bus,
        crystallization=candidate,
        settings=_Settings(),
        project_config=ProjectionConfig(),
    )

    assert outcome == "proposed"
    assert cid == candidate.crystallization_id
    assert final_row.dynamics.reinforcement_count == 0
    insert_mock.assert_called_once()
    update_mock.assert_not_called()
    emit_mock.assert_called_once()
    assert emit_mock.call_args.kwargs["lifecycle"] == "proposed"
