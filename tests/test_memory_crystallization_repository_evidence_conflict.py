"""insert_crystallization() must not 500 (or write a duplicate row) if a
crystallization's evidence list itself carries two entries for the same
(source_kind, source_id) -- the DB-level backstop behind the two application
dedup fixes (WindowStore.append_turn, build_crystallization_from_window) for
the live 2026-08-20 duplicate-evidence bug. Targets the
idx_mcr_sources_unique index added in
orion/core/storage/sql/memory_crystallizations.sql.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from orion.memory.crystallization.repository import insert_crystallization
from orion.memory.crystallization.schemas import (
    CrystallizationEvidenceRefV1,
    CrystallizationGovernanceV1,
    MemoryCrystallizationV1,
    new_crystallization_id,
)


def _pool_with_conn() -> tuple[AsyncMock, AsyncMock]:
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value={"crystallization_id": "11111111-1111-1111-1111-111111111111"})
    conn.execute = AsyncMock()
    conn.transaction = MagicMock()
    conn.transaction.return_value.__aenter__ = AsyncMock(return_value=None)
    conn.transaction.return_value.__aexit__ = AsyncMock(return_value=None)

    pool = AsyncMock()
    pool.acquire = MagicMock()
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire.return_value = acquire_cm
    return pool, conn


@pytest.mark.asyncio
async def test_evidence_insert_targets_the_unique_index_with_do_nothing():
    pool, conn = _pool_with_conn()
    now = datetime(2026, 8, 20, tzinfo=timezone.utc)
    crystallization = MemoryCrystallizationV1(
        crystallization_id=new_crystallization_id(),
        kind="episode",
        subject="s",
        summary="s",
        governance=CrystallizationGovernanceV1(proposed_by="test"),
        created_at=now,
        updated_at=now,
        evidence=[
            CrystallizationEvidenceRefV1(source_kind="chat_turn", source_id="corr-1", strength=0.5),
            # Same (source_kind, source_id) twice -- exactly the shape produced
            # by the live bug before the upstream dedup fixes.
            CrystallizationEvidenceRefV1(source_kind="chat_turn", source_id="corr-1", strength=0.5),
        ],
    )

    await insert_crystallization(pool, crystallization)

    evidence_calls = [c for c in conn.execute.await_args_list if "memory_crystallization_sources" in c.args[0]]
    assert len(evidence_calls) == 2, "both rows are still attempted -- DO NOTHING is a DB-level no-op, not a skip"
    for call in evidence_calls:
        sql = call.args[0]
        assert "ON CONFLICT (crystallization_id, source_kind, source_id) DO NOTHING" in sql
