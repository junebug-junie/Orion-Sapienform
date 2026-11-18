"""The line's rows against a real Postgres (game company patch 3): the migration (twice), the hard order of a yield
and a delivery (a yielded row can never be claimed, and the finish never overwrites it), the settled looks, the
reply within 2 minutes and the thumb (only a delivered line; a guest's is stored as a guest's). Skipped without
ORION_TEST_PG_URI (a throwaway database: the run creates and drops its own schema). CI runs it in
.github/workflows/retro-memory-tests.yml."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

DSN = os.getenv("ORION_TEST_PG_URI")
pytestmark = pytest.mark.skipif(not DSN, reason="ORION_TEST_PG_URI (throwaway database) required")

REPO = Path(__file__).resolve().parents[3]
MIGRATIONS = [REPO / "services/orion-sql-db" / name for name in (
    "manual_migration_retro_play_sessions_v1.sql", "manual_migration_retro_watch_v1.sql",
    "manual_migration_retro_watch_v2.sql", "manual_migration_retro_lines_v1.sql")]
NOW = datetime(2026, 11, 18, 20, 0, tzinfo=timezone.utc)


async def _insert(conn, rc, line_id, sid="s1"):
    await conn.execute(rc.INSERT_SQL, line_id, sid, "alttp", line_id, line_id, NOW, None, ["system", "system"])


async def _finish(conn, rc, line_id, outcome, text=None):
    return await conn.fetchval(rc.FINISH_SQL, line_id, outcome, NOW, None, text, ["look-1"], ["h1"], None, [],
                               1, 1, 0, 11.0, 41.0, 17.0, 70.0, "hold-1", ["system", "system"], None)


@pytest.mark.asyncio
async def test_line_rows_keep_the_hard_order_and_the_player_signals() -> None:
    asyncpg = pytest.importorskip("asyncpg")
    from scripts import retro_commentary as rc

    schema = "retro_line_test_" + uuid4().hex[:10]
    admin = await asyncpg.connect(DSN)
    await admin.execute(f'CREATE SCHEMA "{schema}"')
    try:
        conn = await asyncpg.connect(DSN, server_settings={"search_path": schema})
        try:
            for m in MIGRATIONS:
                await conn.execute(m.read_text())
            await conn.execute(MIGRATIONS[-1].read_text())                 # a second apply changes nothing

            # Yielded first: the claim returns nothing, and the finish keeps 'yielded'.
            await _insert(conn, rc, "L1")
            assert await conn.fetchval(rc.YIELD_SQL, "L1", "juniper_spoke") == "L1"
            assert await conn.fetchval(rc.CLAIM_SQL, "L1") is None
            assert await _finish(conn, rc, "L1", "delivered", "too late") == "yielded"
            assert await conn.fetchval("SELECT text FROM retro_line WHERE line_id = 'L1'") == "too late"

            # Claimed first: a later yield finds nothing to mark.
            await _insert(conn, rc, "L2")
            assert await conn.fetchval(rc.CLAIM_SQL, "L2") == "L2"
            assert await conn.fetchval(rc.YIELD_SQL, "L2", "juniper_spoke") is None
            assert await _finish(conn, rc, "L2", "failed") == "delivered"
            row = await conn.fetchrow("SELECT stance_sec, motor_sec, finalize_sec, lease_priorities FROM retro_line "
                                      "WHERE line_id = 'L2'")
            assert (row["stance_sec"], row["motor_sec"], row["finalize_sec"]) == (11.0, 41.0, 17.0)
            assert row["lease_priorities"] == ["system", "system"]

            # A pass and a no_frames line finish with their own outcome.
            await _insert(conn, rc, "L3")
            assert await _finish(conn, rc, "L3", "passed") == "passed"
            await _insert(conn, rc, "L4")
            assert await _finish(conn, rc, "L4", "no_frames") == "no_frames"

            # Thumbs: only a delivered line; a guest's is stored as a guest's.
            assert await conn.fetchval(rc.THUMB_SQL, "L2", "up", "guest") == "guest"
            assert await conn.fetchval(rc.THUMB_SQL, "L3", "up", "juniper") is None
            assert await conn.fetchval(rc.THUMB_SQL, "nope", "down", "juniper") is None

            # A reply within 2 minutes of a delivered line marks it (once).
            await conn.execute("UPDATE retro_line SET finished_at = now() - interval '30 seconds' WHERE line_id = 'L2'")
            assert await conn.execute(rc.REPLIED_SQL, "s1", "juniper", 120.0) == "UPDATE 1"
            assert await conn.execute(rc.REPLIED_SQL, "s1", "juniper", 120.0) == "UPDATE 0"
            await _insert(conn, rc, "L5")
            await conn.fetchval(rc.CLAIM_SQL, "L5")
            await conn.execute("UPDATE retro_line SET finished_at = now() - interval '10 minutes' WHERE line_id = 'L5'")
            assert await conn.execute(rc.REPLIED_SQL, "s1", "guest", 120.0) == "UPDATE 0"

            # Settled looks: the bucket is written once.
            await conn.execute(
                "INSERT INTO retro_watch_look (look_id, sitting_id, looked_at, outcome, want_id, note) "
                "VALUES ('look-1', 's1', $1, 'stored', 'w1', 'Link walks'), ('look-2', 's1', $1, 'stored', 'w2', 'x')",
                NOW - timedelta(minutes=1))
            await conn.executemany(rc.SETTLE_SQL, [("look-1", True, "checked_holds", NOW, "L2"),
                                                   ("look-2", False, "unchecked", NOW, "L2")])
            await conn.executemany(rc.SETTLE_SQL, [("look-1", True, "contradicted", NOW, "L5")])     # judged: kept
            await conn.executemany(rc.SETTLE_SQL, [("look-2", False, "unchecked", NOW, "L5")])       # unchecked: kept
            rows = {r["look_id"]: r for r in await conn.fetch(
                "SELECT look_id, checkable, verdict, settled_by_line FROM retro_watch_look")}
            assert rows["look-1"]["verdict"] == "checked_holds" and rows["look-1"]["settled_by_line"] == "L2"
            assert rows["look-2"]["checkable"] is False and rows["look-2"]["verdict"] == "unchecked"
            assert rows["look-2"]["settled_by_line"] == "L2"
            # A note a passed turn left unchecked may still be judged by a later turn that could check it.
            await conn.executemany(rc.SETTLE_SQL, [("look-2", True, "checked_holds", NOW, "L5")])
            assert await conn.fetchval("SELECT verdict FROM retro_watch_look WHERE look_id = 'look-2'") == "checked_holds"

            # The gate's reads.
            assert await conn.fetchval(rc.NEWEST_NOTE_TEXT_SQL, "s1") in ("Link walks", "x")
            notes = await conn.fetch(rc.NOTES_SQL, "s1", NOW - timedelta(hours=1))
            assert {r["look_id"] for r in notes} == {"look-1", "look-2"}
            assert await conn.fetch(rc.PREVIOUS_LINES_SQL, "s1") == []      # no delivered line has text yet

            # The outcome check refuses an unknown outcome.
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute("UPDATE retro_line SET outcome = 'said' WHERE line_id = 'L3'")
        finally:
            await conn.close()
    finally:
        await admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await admin.close()
