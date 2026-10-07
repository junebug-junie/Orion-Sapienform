"""Regression: every intake crystallization carries a unique `memory_window:<id>` scope,
which made the duplicate check's scope-overlap rule impossible to satisfy, so identical
chat turns were saved again and again ("Run github compactor." x8, live 2026-10-05)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orion.memory.crystallization.detection import detect_duplicates, scopes_overlap
from orion.memory.crystallization.intake_pipeline import process_consolidation_crystallization
from orion.memory.crystallization.projector import ProjectionConfig
from orion.memory.crystallization.proposer import propose
from orion.memory.crystallization.schemas import (
    CrystallizationEvidenceRefV1,
    MemoryCrystallizationProposeRequestV1,
    new_crystallization_id,
)


def _row(window: str, summary: str = "Run github compactor.", scope: list[str] | None = None):
    row = propose(
        MemoryCrystallizationProposeRequestV1(
            kind="semantic",
            subject=summary,
            summary=summary,
            scope=scope if scope is not None else [f"memory_window:{window}"],
            evidence=[CrystallizationEvidenceRefV1(source_kind="chat_turn", source_id=window)],
            proposed_by="test",
        )
    )
    row.crystallization_id = new_crystallization_id()
    return row


def test_window_scopes_do_not_block_duplicates():
    old, new = _row("w1"), _row("w2")
    assert detect_duplicates(new, [old]).duplicates == [old.crystallization_id]


def test_real_topic_scopes_still_gate():
    a = _row("w1", scope=["project:orion", "memory_window:w1"])
    b = _row("w2", scope=["project:other", "memory_window:w2"])
    assert not scopes_overlap(a.scope, b.scope)
    assert detect_duplicates(b, [a]).duplicates == []


def test_different_text_is_not_a_duplicate():
    assert detect_duplicates(_row("w2", "Plan the staging rollout for k3s"), [_row("w1")]).duplicates == []


class _Settings:
    MEMORY_FORMATION_AUTO_ACTIVATE_ENABLED = True
    MEMORY_FORMATION_AUTO_ENCODE_ACTIVATION_RATIO = 0.4
    SERVICE_NAME = "orion-memory-consolidation"
    SERVICE_VERSION = "0.1.0"
    NODE_NAME = "test"


@pytest.mark.asyncio
async def test_copy_outside_top_window_reinforces_not_inserts(monkeypatch):
    """The salience-ranked window misses the old copy; the exact-text lookup finds it."""
    old = _row("w1")
    old.status = "active"
    new = _row("w2")
    mod = "orion.memory.crystallization.intake_pipeline."
    insert, update = AsyncMock(), AsyncMock()
    monkeypatch.setattr(mod + "list_crystallizations", AsyncMock(return_value=[]))
    monkeypatch.setattr(mod + "find_exact_duplicates", AsyncMock(return_value=[old]))
    monkeypatch.setattr(mod + "insert_crystallization", insert)
    monkeypatch.setattr(mod + "update_crystallization", update)
    monkeypatch.setattr(mod + "emit_crystallization_lifecycle", AsyncMock(return_value=True))

    cid, _row_out, outcome = await process_consolidation_crystallization(
        MagicMock(),
        MagicMock(enabled=True),
        crystallization=new,
        settings=_Settings(),
        project_config=ProjectionConfig(),
    )
    assert outcome == "reinforced" and cid == old.crystallization_id
    insert.assert_not_called()
    update.assert_awaited_once()


async def _run(monkeypatch, old, new):
    mod = "orion.memory.crystallization.intake_pipeline."
    insert, update = AsyncMock(return_value="new-id"), AsyncMock()
    monkeypatch.setattr(mod + "list_crystallizations", AsyncMock(return_value=[]))
    monkeypatch.setattr(mod + "find_exact_duplicates", AsyncMock(return_value=[old]))
    monkeypatch.setattr(mod + "insert_crystallization", insert)
    monkeypatch.setattr(mod + "update_crystallization", update)
    monkeypatch.setattr(mod + "emit_crystallization_lifecycle", AsyncMock(return_value=True))
    _, _, outcome = await process_consolidation_crystallization(
        MagicMock(), MagicMock(enabled=True), crystallization=new,
        settings=_Settings(), project_config=ProjectionConfig(),
    )
    return outcome, insert, update


@pytest.mark.asyncio
async def test_intimate_window_never_reinforces_an_active_row(monkeypatch):
    old = _row("w1")
    old.status = "active"
    new = _row("w2")
    new.governance.sensitivity = "intimate"
    outcome, _insert, update = await _run(monkeypatch, old, new)
    assert outcome == "proposed"
    update.assert_not_called()


@pytest.mark.asyncio
async def test_identity_scoped_window_never_reinforces_an_active_row(monkeypatch):
    old = _row("w1")
    old.status = "active"
    new = _row("w2", scope=["identity:juniper", "memory_window:w2"])
    outcome, _insert, update = await _run(monkeypatch, old, new)
    assert outcome == "proposed"
    update.assert_not_called()


@pytest.mark.asyncio
async def test_short_exact_copy_reinforces_despite_zero_token_overlap(monkeypatch):
    old = _row("w1", "ok")
    old.status = "active"
    outcome, insert, _update = await _run(monkeypatch, old, _row("w2", "ok"))
    assert outcome == "reinforced"
    insert.assert_not_called()
