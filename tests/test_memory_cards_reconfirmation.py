"""Integration tests for Stage 2 auto-promotion phase 1: reconfirmation
tracking (requires RECALL_PG_DSN + asyncpg + psycopg2), following the same
real-Postgres pattern as test_memory_cards_reverse_history.py."""

from __future__ import annotations

import asyncio
import os

import pytest


@pytest.mark.skipif(not os.environ.get("RECALL_PG_DSN"), reason="RECALL_PG_DSN not set")
def test_fetch_card_id_by_fingerprint_and_record_reconfirmation_roundtrip() -> None:
    pytest.importorskip("asyncpg")
    pytest.importorskip("psycopg2")
    import asyncpg

    from orion.core.contracts.memory_cards import MemoryCardCreateV1
    from orion.core.storage.memory_cards import (
        apply_memory_cards_schema,
        count_distinct_reconfirmation_sessions,
        fetch_card_id_by_fingerprint,
        insert_card,
        record_reconfirmation,
    )

    dsn = os.environ["RECALL_PG_DSN"]
    apply_memory_cards_schema(dsn)

    async def _body() -> None:
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=1)
        assert pool is not None
        fp = "test-reconfirmation-fingerprint-8f2c1a"
        try:
            # No card yet -- must return None, not raise.
            assert await fetch_card_id_by_fingerprint(pool, fp) is None

            card = MemoryCardCreateV1(
                types=["fact"],
                title="Test reconfirmation card",
                summary="A fact used only by this test.",
                provenance="auto_extractor",
                status="pending_review",
                subschema={"auto_extractor_fingerprint": fp, "auto_extractor_session_id": "session-a"},
            )
            card_id = await insert_card(pool, card, actor="pytest", op="create")

            found_id = await fetch_card_id_by_fingerprint(pool, fp)
            assert found_id == card_id

            # Created in session-a only -- 1 distinct session so far.
            assert await count_distinct_reconfirmation_sessions(pool, card_id) == 1

            # Reconfirmed from the SAME session -- still 1 distinct session
            # (this is the echo-guard case: repetition within one session
            # must not look like independent confirmation).
            await record_reconfirmation(pool, card_id=card_id, session_id="session-a", actor="pytest")
            assert await count_distinct_reconfirmation_sessions(pool, card_id) == 1

            # Reconfirmed from a DIFFERENT session -- now 2 distinct sessions.
            await record_reconfirmation(pool, card_id=card_id, session_id="session-b", actor="pytest")
            assert await count_distinct_reconfirmation_sessions(pool, card_id) == 2

            # A history row with op='reconfirmed' must actually exist and be
            # attributed to the right card -- not just inferred from the count.
            rows = await pool.fetch(
                "SELECT op, actor, \"after\" FROM memory_card_history WHERE card_id = $1 AND op = 'reconfirmed' ORDER BY created_at",
                card_id,
            )
            assert len(rows) == 2
            assert rows[0]["actor"] == "pytest"
            import json

            after_0 = rows[0]["after"] if isinstance(rows[0]["after"], dict) else json.loads(rows[0]["after"])
            after_1 = rows[1]["after"] if isinstance(rows[1]["after"], dict) else json.loads(rows[1]["after"])
            assert after_0["session_id"] == "session-a"
            assert after_1["session_id"] == "session-b"
        finally:
            await pool.execute("DELETE FROM memory_card_history WHERE card_id IN (SELECT card_id FROM memory_cards WHERE subschema ->> 'auto_extractor_fingerprint' = $1)", fp)
            await pool.execute("DELETE FROM memory_cards WHERE subschema ->> 'auto_extractor_fingerprint' = $1", fp)
            await pool.close()

    asyncio.run(_body())


@pytest.mark.skipif(not os.environ.get("RECALL_PG_DSN"), reason="RECALL_PG_DSN not set")
def test_record_reconfirmation_with_null_session_id_does_not_crash() -> None:
    """Not every turn carries a session_id (ChatHistoryTurnV1.session_id is
    Optional) -- recording a reconfirmation must handle None gracefully,
    not error out and skip recording entirely."""
    pytest.importorskip("asyncpg")
    pytest.importorskip("psycopg2")
    import asyncpg

    from orion.core.contracts.memory_cards import MemoryCardCreateV1
    from orion.core.storage.memory_cards import apply_memory_cards_schema, insert_card, record_reconfirmation

    dsn = os.environ["RECALL_PG_DSN"]
    apply_memory_cards_schema(dsn)

    async def _body() -> None:
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=1)
        assert pool is not None
        fp = "test-reconfirmation-null-session-4d9e2b"
        try:
            card = MemoryCardCreateV1(
                types=["fact"],
                title="Null session test card",
                summary="A fact used only by this test.",
                provenance="auto_extractor",
                status="pending_review",
                subschema={"auto_extractor_fingerprint": fp},
            )
            card_id = await insert_card(pool, card, actor="pytest", op="create")
            history_id = await record_reconfirmation(pool, card_id=card_id, session_id=None, actor="pytest")
            assert history_id is not None
        finally:
            await pool.execute("DELETE FROM memory_card_history WHERE card_id IN (SELECT card_id FROM memory_cards WHERE subschema ->> 'auto_extractor_fingerprint' = $1)", fp)
            await pool.execute("DELETE FROM memory_cards WHERE subschema ->> 'auto_extractor_fingerprint' = $1", fp)
            await pool.close()

    asyncio.run(_body())
