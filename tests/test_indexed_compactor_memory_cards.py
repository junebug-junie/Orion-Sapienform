from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from orion.core.contracts.memory_cards import MemoryCardCreateV1
from orion.core.storage import memory_cards as mc_dal


INDEX = "chat_compactor:day:2026-07-08"


@pytest.mark.asyncio
async def test_upsert_indexed_compactor_card_updates_existing() -> None:
    existing_id = uuid4()
    conn = AsyncMock()
    # FOR UPDATE lookup (with before snapshot) → UPDATE RETURNING after snapshot
    conn.fetchrow = AsyncMock(
        side_effect=[
            {
                "card_id": existing_id,
                "j": {"card_id": str(existing_id), "summary": "old"},
            },
            {"j": {"card_id": str(existing_id), "summary": "Updated summary"}},
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
        anchor_class="event",
        status="active",
        priority="high_recall",
        provenance="chat_compactor",
        title="Chat digest (2026-07-08)",
        summary="Updated summary",
        tags=["chat_dev_digest"],
        subschema={"compactor_index": INDEX, "compactor_kind": "chat_history_log"},
    )

    result_id = await mc_dal.upsert_indexed_compactor_card(
        pool,
        index=INDEX,
        card=card,
        actor="chat_history_compactor_pass",
    )
    assert result_id == existing_id
    # Must not supersede
    for call in conn.execute.await_args_list:
        sql = str(call.args[0]).lower()
        assert "superseded" not in sql
    # FOR UPDATE lookup must key on compactor_index
    lookup_sql = str(conn.fetchrow.await_args_list[0].args[0]).lower()
    assert "compactor_index" in lookup_sql
    assert "for update" in lookup_sql


@pytest.mark.asyncio
async def test_upsert_indexed_compactor_card_inserts_when_missing() -> None:
    new_id = uuid4()
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(side_effect=[None])  # no existing
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
        anchor_class="event",
        status="active",
        priority="high_recall",
        provenance="chat_compactor",
        title="Chat digest",
        summary="Fresh digest",
        tags=["chat_dev_digest"],
        subschema={"compactor_index": INDEX},
    )

    async def _fake_insert(_conn, _card, *, actor, op="create"):
        assert op == "create"
        return new_id

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mc_dal, "_insert_card_on_conn", _fake_insert)
    try:
        result_id = await mc_dal.upsert_indexed_compactor_card(
            pool,
            index=INDEX,
            card=card,
            actor="chat_history_compactor_pass",
        )
    finally:
        monkeypatch.undo()

    assert result_id == new_id
