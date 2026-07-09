from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from orion.core.contracts.memory_cards import MemoryCardCreateV1
from orion.core.storage import memory_cards as mc_dal
from orion.cognition.github_compactor.constants import REPO_DEV_SNAPSHOT_SLOT


@pytest.mark.asyncio
async def test_supersede_and_insert_compactor_card_supersedes_existing() -> None:
    old_id = uuid4()
    new_id = uuid4()

    conn = AsyncMock()
    conn.fetchrow = AsyncMock(
        side_effect=[
            {"card_id": old_id},  # FOR UPDATE lookup
            {"j": {"card_id": str(old_id), "status": "active"}},  # before snapshot
            {"j": {"card_id": str(old_id), "status": "superseded"}},  # after snapshot
        ]
    )
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

    card = MemoryCardCreateV1(
        types=["fact"],
        anchor_class="project",
        status="active",
        priority="high_recall",
        provenance="repo_compactor",
        project="acme/widgets",
        title="Recent repo development",
        summary="Built compactor.",
        tags=[REPO_DEV_SNAPSHOT_SLOT],
        subschema={"compactor_slot": REPO_DEV_SNAPSHOT_SLOT},
    )

    async def _fake_insert(_conn, _card, *, actor, op="create"):
        return new_id

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mc_dal, "_insert_card_on_conn", _fake_insert)
    try:
        result_id = await mc_dal.supersede_and_insert_compactor_card(
            pool,
            slot=REPO_DEV_SNAPSHOT_SLOT,
            card=card,
            actor="github_compactor_pass",
        )
    finally:
        monkeypatch.undo()

    assert result_id == new_id
    assert conn.execute.await_count >= 1
    executed_sql = conn.execute.await_args_list[0].args[0]
    assert "status = 'superseded'" in executed_sql.lower() or "superseded" in executed_sql.lower()
