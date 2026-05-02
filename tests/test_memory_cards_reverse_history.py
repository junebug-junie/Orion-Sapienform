"""Integration tests for memory card history undo (requires RECALL_PG_DSN + asyncpg + psycopg2)."""

from __future__ import annotations

import asyncio
import os
from uuid import UUID, uuid4

import pytest


@pytest.mark.skipif(not os.environ.get("RECALL_PG_DSN"), reason="RECALL_PG_DSN not set")
def test_reverse_history_unknown_op_raises_and_leaves_no_audit_row() -> None:
    pytest.importorskip("asyncpg")
    pytest.importorskip("psycopg2")
    import asyncpg

    from orion.core.storage.memory_cards import apply_memory_cards_schema, reverse_history

    dsn = os.environ["RECALL_PG_DSN"]
    apply_memory_cards_schema(dsn)

    async def _body() -> None:
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=1)
        assert pool is not None
        try:
            hid = await pool.fetchval(
                """
                INSERT INTO memory_card_history (history_id, card_id, edge_id, op, actor, before, after)
                VALUES (gen_random_uuid(), NULL, NULL, $1, 'pytest', NULL, NULL)
                RETURNING history_id
                """,
                "bogus_op_for_undo_test",
            )
            cnt_before = int(await pool.fetchval("SELECT COUNT(*)::int FROM memory_card_history"))
            with pytest.raises(ValueError, match="not reversible"):
                await reverse_history(pool, str(hid), actor="pytest")
            cnt_after = int(await pool.fetchval("SELECT COUNT(*)::int FROM memory_card_history"))
            assert cnt_after == cnt_before
            await pool.execute("DELETE FROM memory_card_history WHERE history_id = $1", hid)
        finally:
            await pool.close()

    asyncio.run(_body())


@pytest.mark.skipif(not os.environ.get("RECALL_PG_DSN"), reason="RECALL_PG_DSN not set")
def test_reverse_history_edge_remove_restores_edge() -> None:
    pytest.importorskip("asyncpg")
    pytest.importorskip("psycopg2")
    import asyncpg

    from orion.core.contracts.memory_cards import MemoryCardEdgeV1, MemoryCardV1
    from orion.core.storage.memory_cards import (
        add_edge,
        apply_memory_cards_schema,
        insert_card,
        list_edges,
        list_history,
        remove_edge,
        reverse_history,
    )

    dsn = os.environ["RECALL_PG_DSN"]
    apply_memory_cards_schema(dsn)
    uid = uuid4().hex[:12]

    async def _body() -> None:
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
        assert pool is not None
        try:
            c1 = MemoryCardV1(
                slug=f"undo-a-{uid}",
                types=["fact"],
                provenance="pytest",
                title="A",
                summary="A",
            )
            c2 = MemoryCardV1(
                slug=f"undo-b-{uid}",
                types=["fact"],
                provenance="pytest",
                title="B",
                summary="B",
            )
            cid1 = await insert_card(pool, c1, actor="pytest")
            cid2 = await insert_card(pool, c2, actor="pytest")
            assert isinstance(cid1, UUID) and isinstance(cid2, UUID)
            eid = await add_edge(
                pool,
                MemoryCardEdgeV1(from_card_id=cid1, to_card_id=cid2, edge_type="relates_to"),
                actor="pytest",
            )
            out_before = await list_edges(pool, card_id=str(cid1), direction="out")
            assert len(out_before) == 1
            await remove_edge(pool, str(eid), actor="pytest")
            out_mid = await list_edges(pool, card_id=str(cid1), direction="out")
            assert len(out_mid) == 0
            hist = await list_history(pool, edge_id=str(eid), limit=20)
            remove_rows = [h for h in hist if h.op == "edge_remove"]
            assert len(remove_rows) == 1
            await reverse_history(pool, str(remove_rows[0].history_id), actor="pytest")
            out_after = await list_edges(pool, card_id=str(cid1), direction="out")
            assert len(out_after) == 1
            assert out_after[0].edge_id == eid
        finally:
            await pool.execute(
                "DELETE FROM memory_cards WHERE slug = ANY($1::text[])",
                [f"undo-a-{uid}", f"undo-b-{uid}"],
            )
            await pool.close()

    asyncio.run(_body())
