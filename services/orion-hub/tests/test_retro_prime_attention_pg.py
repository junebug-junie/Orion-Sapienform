"""Game company patch 4 against a real Postgres: the SQL behind world:play (orion/attention/play_read.py, run by both
attention contests) and the finished sitting's event-want history (scripts/retro_sitting.py EVENT_HISTORY_SQL), on the
tables the retro migrations create. Skipped without ORION_TEST_PG_URI (a throwaway database: the run creates and drops
its own schema). CI runs it in .github/workflows/retro-memory-tests.yml."""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

DSN = os.getenv("ORION_TEST_PG_URI")
pytestmark = pytest.mark.skipif(not DSN, reason="ORION_TEST_PG_URI (throwaway database) required")

REPO = Path(__file__).resolve().parents[3]
MIGRATIONS = [REPO / "services/orion-sql-db" / f for f in (
    "manual_migration_retro_play_sessions_v1.sql", "manual_migration_retro_watch_v1.sql",
    "manual_migration_retro_watch_v2.sql", "manual_migration_retro_lines_v1.sql")]


def test_play_inputs_and_event_history_read_the_real_tables():
    asyncpg = pytest.importorskip("asyncpg")
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from orion.attention.play_read import read_play_inputs
    from orion.attention.world_first import judge_candidate, play_candidate
    from orion.retro.sitting import Sitting
    from scripts import retro_sitting

    schema = "retro_p4_test_" + uuid4().hex[:10]
    now = datetime.now(timezone.utc).replace(microsecond=0)

    async def setup():
        admin = await asyncpg.connect(DSN)
        await admin.execute(f'CREATE SCHEMA "{schema}"')
        await admin.close()
        conn = await asyncpg.connect(DSN, server_settings={"search_path": schema})
        for m in MIGRATIONS:
            await conn.execute(m.read_text())
        rows = [(f"b{d}", now - timedelta(days=d, hours=3), now - timedelta(days=d, hours=2), True, None)
                for d in range(1, 26)]
        rows.append(("S1", now - timedelta(minutes=25), now - timedelta(seconds=20), False, None))
        for sid, start, last, backfilled, ended in rows:
            await conn.execute(
                "INSERT INTO retro_sitting (sitting_id, retro_session_id, game_id, game_title, started_at, last_input_at,"
                " ended_at, backfilled) VALUES ($1, 'r', 'crystalis', 'Crystalis', $2, $3, $4, $5)",
                sid, start, last, ended if ended else (None if sid == "S1" else last), backfilled)
        await conn.execute(
            "INSERT INTO retro_watch_preference (pref_id, scope, scope_id, set_by, mode, why, set_at) VALUES "
            "('p1', 'game', 'crystalis', 'orion', 'off', 'not tonight', $1)", now - timedelta(hours=1))
        await conn.execute(
            "INSERT INTO retro_watch_look (look_id, sitting_id, game_id, looked_at, frame_hashes, note, outcome, want_id)"
            " VALUES ('l1', 'S1', 'crystalis', $1, ARRAY['h'], 'A hero walks east.', 'stored', 'w1'),"
            "('l2', 'S1', 'crystalis', $2, ARRAY['h'], 'unclear', 'unclear', 'w2')",
            now - timedelta(seconds=60), now - timedelta(seconds=10))
        await conn.close()

    async def teardown():
        admin = await asyncpg.connect(DSN)
        await admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await admin.close()

    asyncio.run(setup())
    try:
        url = DSN.replace("postgresql://", "postgresql+psycopg2://", 1)
        engine = sqlalchemy.create_engine(url, connect_args={"options": f"-csearch_path={schema}"})
        sittings, watch, absent = read_play_inputs(engine, now=now)
        # Only the sittings that can still speak into the 7-day grid (6 of the 25 backfilled ones, and S1).
        assert absent is None and len(sittings) == 7
        # Orion's standing "off" for the game, and the newest STORED note (not the newer unclear one).
        assert (watch.mode, watch.set_by, watch.why) == ("off", "orion", "not tonight")
        assert watch.newest_note == "A hero walks east."
        v = judge_candidate(play_candidate(sittings, now=now, watch=watch))
        assert not v.eligible and v.reason.startswith("not watching: the watch is off (set by Orion")
        engine.dispose()
        # A database before the migrations: absent with its reason, never "nobody plays".
        bare = sqlalchemy.create_engine(url, connect_args={"options": "-csearch_path=no_such_schema_p4"})
        s2, w2, why = read_play_inputs(bare, now=now)
        assert s2 is None and w2 is None and why.startswith("retro_sitting unreadable")
        bare.dispose()

        async def history():
            pool = await asyncpg.create_pool(DSN, min_size=1, max_size=1, server_settings={"search_path": schema})
            try:
                bids = []

                async def bid(want):
                    bids.append(want)
                    return SimpleNamespace(outcome="funded", reason="net_positive")

                cfg = SimpleNamespace(HUB_RETRO_SITTING_ENABLED=True, HUB_RETRO_SITTING_MIN_SEC=120,
                                      HUB_RETRO_SITTING_EVENT_WANT_ENABLED=True)
                svc = retro_sitting.SittingService(cfg, pool_provider=lambda: pool, bid=bid)
                s = Sitting(sitting_id="S1", retro_session_id="r", started_at=(now - timedelta(minutes=90)).timestamp(),
                            last_input_at=now.timestamp(), ended_at=now.timestamp(), game_id="crystalis")
                out = await svc._post_event_want(s)
                return out, bids
            finally:
                await pool.close()

        out, bids = asyncio.run(history())
        # 25 backfilled hour-long sittings in the 30 days before; S1 itself is never its own history.
        assert out == "funded" and len(bids) == 1 and bids[0].salience == 1.0
    finally:
        asyncio.run(teardown())
