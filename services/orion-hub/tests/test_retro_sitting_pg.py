"""The sitting record against a real Postgres: the migration (twice), the sitting row, the human_play
row, and the Temporal Self source query that reads it. Skipped without ORION_TEST_PG_URI (a throwaway
database: the run creates and drops its own schema)."""
from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

DSN = os.getenv("ORION_TEST_PG_URI")
pytestmark = pytest.mark.skipif(not DSN, reason="ORION_TEST_PG_URI (throwaway database) required")

REPO = Path(__file__).resolve().parents[3]
PLAY_MIGRATION = REPO / "services/orion-sql-db/manual_migration_retro_play_sessions_v1.sql"
WATCH_MIGRATION = REPO / "services/orion-sql-db/manual_migration_retro_watch_v1.sql"
SID = "a" * 36


def msg(mask):
    import json
    return json.dumps({"version": 1, "session_id": SID, "port": 0, "epoch": 1, "seq": 1, "mask": mask})


@pytest.mark.asyncio
async def test_a_sitting_is_stored_and_read_back_as_a_play_arc_that_credits_orion_with_nothing(monkeypatch) -> None:
    asyncpg = pytest.importorskip("asyncpg")
    from orion.temporal_self import sources as S
    from scripts import retro_sitting

    monkeypatch.setattr(retro_sitting, "TICK_SEC", 3600)
    schema = "retro_sitting_test_" + uuid4().hex[:10]
    admin = await asyncpg.connect(DSN)
    await admin.execute(f'CREATE SCHEMA "{schema}"')
    try:
        pool = await asyncpg.create_pool(DSN, min_size=1, max_size=2, server_settings={"search_path": schema})
        try:
            async with pool.acquire() as conn:
                await conn.execute(PLAY_MIGRATION.read_text())
                await conn.execute(WATCH_MIGRATION.read_text())
                await conn.execute(WATCH_MIGRATION.read_text())          # a second apply changes nothing

            async def resolve(sid): return "a-link-to-the-past", "The Legend of Zelda: A Link to the Past"
            clock = {"t": 1_800_000_000.0}
            cfg = SimpleNamespace(HUB_RETRO_SITTING_ENABLED=True, HUB_RETRO_SITTING_GAP_SEC=300, HUB_RETRO_SITTING_MIN_SEC=120)
            svc = retro_sitting.SittingService(cfg, pool_provider=lambda: pool, resolve_game=resolve, clock=lambda: clock["t"])
            svc.note_input(SID, msg(128))
            while svc._jobs:
                import asyncio
                await asyncio.gather(*list(svc._jobs))
            async with pool.acquire() as conn:
                open_row = await conn.fetchrow("SELECT * FROM retro_sitting")
            assert open_row["ended_at"] is None and open_row["game_title"].endswith("A Link to the Past")
            assert open_row["backfilled"] is False and open_row["mask_changes"] == 1
            assert await svc.state_player("juniper") is True
            for i in range(1, 9):                                          # eight more changes, 100 s apart
                clock["t"] += 100.0
                svc.note_input(SID, msg(0 if i % 2 else 128))
            svc.note_action(SID, "save")
            await svc.close()                                              # the gateway's shutdown path

            async with pool.acquire() as conn:
                sitting = await conn.fetchrow("SELECT * FROM retro_sitting")
                play = await conn.fetchrow("SELECT * FROM retro_play_sessions")
                assert await conn.fetchval("SELECT count(*) FROM retro_sitting") == 1
                assert await conn.fetchval("SELECT count(*) FROM retro_play_sessions") == 1
                # the Temporal Self chronicle's own query (services/orion-durable-runs), with its two bounds
                src = (REPO / "services/orion-durable-runs/app/temporal_self_sources.py").read_text()
                query = re.search(r'SourceQuery\("retro_play", """(.*?)"""', src, re.S).group(1)
                rows = await conn.fetch(query.replace("%(lo)s", "$1").replace("%(hi)s", "$2"),
                                        play["ended_at"], play["ended_at"].replace(year=play["ended_at"].year + 1))
            assert sitting["ended_at"] is not None and sitting["end_reason"] == "shutdown"
            assert (sitting["mask_changes"], sitting["saves"], sitting["player_stated"]) == (9, 1, "juniper")
            assert play["kind"] == "human_play" and play["session_id"] == "sitting:" + sitting["sitting_id"]
            assert (play["orion_frames"], play["human_frames"], play["human_share"]) == (0, 0, None)
            assert play["control_basis"] == "human_input" and play["player_stated"] == "juniper"
            assert play["duration_sec"] == 800.0 and play["memory_card_id"] is None
            (event,) = S.adapt("retro_play", [dict(r) for r in rows])
            assert event.label == "Juniper played The Legend of Zelda: A Link to the Past for 13 min; Orion did not play"
            assert event.subject_ref == "retro_sitting:" + sitting["sitting_id"]
            assert event.payload["duration_sec"] == 800.0
            # the same sitting recorded again (a restart replaying its close) adds nothing
            again = retro_sitting.SittingService(cfg, pool_provider=lambda: pool, clock=lambda: clock["t"])
            from orion.retro.sitting import Sitting
            await again._finish(Sitting(sitting_id=sitting["sitting_id"], retro_session_id=SID, started_at=clock["t"] - 800,
                                        last_input_at=clock["t"], ended_at=clock["t"], end_reason="idle",
                                        game_id="a-link-to-the-past", game_title="A Link to the Past"))
            async with pool.acquire() as conn:
                assert await conn.fetchval("SELECT count(*) FROM retro_play_sessions") == 1
        finally:
            await pool.close()
    finally:
        await admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await admin.close()
