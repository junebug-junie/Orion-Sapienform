"""Orion plays Zelda, section F, against a real Postgres: the migration, the session row, the real
memory_cards writer, and the link between them. Skipped without ORION_TEST_PG_URI (a throwaway
database: every run creates and drops its own schema)."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

DSN = os.getenv("ORION_TEST_PG_URI")
pytestmark = pytest.mark.skipif(not DSN, reason="ORION_TEST_PG_URI (throwaway database) required")

REPO = Path(__file__).resolve().parents[3]
MIGRATION = REPO / "services/orion-sql-db/manual_migration_retro_play_sessions_v1.sql"
CARDS_DDL = REPO / "orion/core/storage/sql/memory_cards.sql"
T0 = datetime(2026, 10, 22, 20, 2, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_export_writes_the_row_and_one_retro_play_card_linked_to_it(monkeypatch) -> None:
    asyncpg = pytest.importorskip("asyncpg")
    from scripts import retro_memory_export as rx
    from scripts.settings import settings

    monkeypatch.setattr(settings, "HUB_RETRO_MEMORY_EXPORT_ENABLED", True, raising=False)
    monkeypatch.setattr(rx, "_table_ready", False)
    schema = "retro_export_test_" + uuid4().hex[:10]
    admin = await asyncpg.connect(DSN)
    await admin.execute(f'CREATE SCHEMA "{schema}"')
    try:
        pool = await asyncpg.create_pool(DSN, min_size=1, max_size=2, server_settings={"search_path": schema})
        try:
            async with pool.acquire() as conn:
                await conn.execute(CARDS_DDL.read_text())
                await conn.execute(MIGRATION.read_text())   # the migration first, then the lazy DDL is a no-op
            journal = [{"kind": "session_started", "at": T0.isoformat()},
                       {"kind": "control", "port": 0, "controller": "human", "frames": 600},
                       {"kind": "control", "port": 0, "controller": "orion", "frames": 400},
                       {"kind": "goal_finished", "decision_id": "d1", "verdict": "met"},
                       {"kind": "session_ended", "at": (T0 + timedelta(minutes=9)).isoformat()}]
            out = await rx.export_session(pool=pool, bus=None, session_id="sess-pg", journal=journal,
                                          atlas_delta={"rooms_new": ["0:77", "0:78"]}, kind="co_play",
                                          game_id="zelda-prg0", lineage_id="lin-pg")
            assert out.state == "exported" and out.card_id
            async with pool.acquire() as conn:
                row = await conn.fetchrow("SELECT * FROM retro_play_sessions WHERE session_id = 'sess-pg'")
                card = await conn.fetchrow("SELECT * FROM memory_cards WHERE card_id = $1", row["memory_card_id"])
            assert row["played_together"] is True and row["human_share"] == pytest.approx(0.6)
            assert row["summary"] == ("Played The Legend of Zelda together for 9 min (a human drove 60% of frames), "
                                      "2 new rooms, 1 of 1 predictions met.")
            assert (card["provenance"], card["trust_source"], card["status"]) == ("retro_play", "retro_play", "active")
            assert card["summary"] == row["summary"] and "retro_session:sess-pg" in card["anchors"]
            again = await rx.export_session(pool=pool, bus=None, session_id="sess-pg", journal=journal,
                                            atlas_delta={}, kind="co_play", game_id="zelda-prg0")
            assert again.state == "duplicate"
            async with pool.acquire() as conn:
                assert await conn.fetchval("SELECT count(*) FROM memory_cards") == 1
        finally:
            await pool.close()
    finally:
        await admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await admin.close()
