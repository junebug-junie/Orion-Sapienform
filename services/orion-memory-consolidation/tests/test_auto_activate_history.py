"""Stage 0A: every auto-activated memory gets an `auto_activate` history row.

Before this, `intake_pipeline` threw away the history dict that
`formation_executor.auto_activate` builds, so 350 auto-saved rows had no audit
trail at all and later surfaces described them as approved.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orion.memory.crystallization.intake_pipeline import process_consolidation_crystallization
from orion.memory.crystallization.projector import ProjectionConfig, ProjectionResult
from orion.memory.crystallization.proposer import propose
from orion.memory.crystallization.schemas import (
    CrystallizationEvidenceRefV1,
    MemoryCrystallizationProposeRequestV1,
)

_P = "orion.memory.crystallization.intake_pipeline."


class _Settings:
    MEMORY_FORMATION_AUTO_ACTIVATE_ENABLED = True
    MEMORY_FORMATION_AUTO_ENCODE_ACTIVATION_RATIO = 0.4
    SERVICE_NAME = "orion-memory-consolidation"
    SERVICE_VERSION = "0.1.0"
    NODE_NAME = "test"


def _semantic():
    return propose(
        MemoryCrystallizationProposeRequestV1(
            kind="semantic",
            subject="Headed to Austin",
            summary="Thanks. Headed to Austin and will fly back on Wednesday.",
            scope=["project:orion"],
            evidence=[CrystallizationEvidenceRefV1(source_kind="chat_turn", source_id="corr-a")],
            proposed_by="test",
        )
    )


def _patch_pipeline(monkeypatch, *, history_mock):
    async def _project(pool, bus, row, **kw):
        return row, ProjectionResult()

    monkeypatch.setattr(_P + "list_crystallizations", AsyncMock(return_value=[]))
    monkeypatch.setattr(_P + "insert_crystallization", AsyncMock(return_value="cid-123"))
    monkeypatch.setattr(_P + "update_crystallization", AsyncMock())
    monkeypatch.setattr(_P + "emit_crystallization_lifecycle", AsyncMock(return_value=True))
    monkeypatch.setattr(_P + "project_crystallization", _project)
    monkeypatch.setattr(_P + "insert_history", history_mock)


@pytest.mark.asyncio
async def test_auto_activated_row_gets_auto_activate_history(monkeypatch):
    history_mock = AsyncMock()
    _patch_pipeline(monkeypatch, history_mock=history_mock)

    cid, row, outcome = await process_consolidation_crystallization(
        MagicMock(), None, crystallization=_semantic(), settings=_Settings(),
        project_config=ProjectionConfig(),
    )

    assert outcome == "auto_activated"
    assert row.status == "active"
    history_mock.assert_awaited_once()
    kw = history_mock.await_args.kwargs
    assert kw["crystallization_id"] == "cid-123"
    assert kw["op"] == "auto_activate"
    assert kw["actor"] == "system:formation_policy"
    assert kw["after"]["status"] == "active"
    assert kw["before"]["status"] == "proposed"
    # Never 'approve': that op means a deliberate decision.
    assert kw["op"] != "approve"


@pytest.mark.asyncio
async def test_history_failure_is_logged_not_swallowed_silently(monkeypatch, caplog):
    history_mock = AsyncMock(side_effect=RuntimeError("db down"))
    _patch_pipeline(monkeypatch, history_mock=history_mock)

    with caplog.at_level("ERROR"):
        _cid, _row, outcome = await process_consolidation_crystallization(
            MagicMock(), None, crystallization=_semantic(), settings=_Settings(),
            project_config=ProjectionConfig(),
        )

    assert outcome == "auto_activated"
    assert "crystallization_auto_activate_history_failed" in caplog.text


@pytest.mark.asyncio
async def test_governor_path_rows_write_no_auto_history(monkeypatch):
    history_mock = AsyncMock()
    _patch_pipeline(monkeypatch, history_mock=history_mock)
    stance = _semantic()
    stance.kind = "stance"  # formation_policy routes stance to the governor queue

    _cid, row, outcome = await process_consolidation_crystallization(
        MagicMock(), None, crystallization=stance, settings=_Settings(),
        project_config=ProjectionConfig(),
    )

    assert outcome == "proposed"
    history_mock.assert_not_called()
