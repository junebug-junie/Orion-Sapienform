"""Orion's look (game company patch 2) against a real Postgres: the migration (twice), the histogram and the
floor, the counters on the sitting row, a look row and the frame-expiry move, Orion's choices copied from the act
ledger, Juniper's switch, the play row's note count read back by the Temporal Self query, and every statement
`dig watch status` sends. Skipped without ORION_TEST_PG_URI (a throwaway database: the run creates and drops its
own schema). CI runs it in .github/workflows/retro-memory-tests.yml."""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

DSN = os.getenv("ORION_TEST_PG_URI")
pytestmark = pytest.mark.skipif(not DSN, reason="ORION_TEST_PG_URI (throwaway database) required")

REPO = Path(__file__).resolve().parents[3]
MIGRATIONS = [REPO / "services/orion-sql-db" / n for n in (
    "manual_migration_retro_play_sessions_v1.sql", "manual_migration_retro_watch_v1.sql",
    "manual_migration_retro_watch_v2.sql", "manual_migration_retro_watch_v2.sql")]   # v2 twice: changes nothing
# The columns of grammar_acts the copy of Orion's choices reads (the live table has more).
GRAMMAR_ACTS = """CREATE TABLE grammar_acts (act_id varchar PRIMARY KEY, actor varchar, verb varchar, object_ref text,
  test_before text, session_id varchar, opened_at timestamptz, outcome_note text, created_at timestamptz DEFAULT now())"""


@pytest.mark.asyncio
async def test_the_watch_tables_round_trip() -> None:
    asyncpg = pytest.importorskip("asyncpg")
    from orion.retro import watch as W
    from orion.temporal_self import sources as S
    from scripts import retro_sitting, retro_watch

    schema = "retro_watch_test_" + uuid4().hex[:10]
    admin = await asyncpg.connect(DSN)
    await admin.execute(f'CREATE SCHEMA "{schema}"')
    try:
        pool = await asyncpg.create_pool(DSN, min_size=1, max_size=2, server_settings={"search_path": schema})
        try:
            async with pool.acquire() as conn:
                for m in MIGRATIONS:
                    await conn.execute(m.read_text())
                await conn.execute(GRAMMAR_ACTS)
            now = datetime.now(timezone.utc)
            clock = {"t": now.timestamp()}
            cfg = SimpleNamespace(HUB_RETRO_SITTING_ENABLED=True, HUB_RETRO_SITTING_GAP_SEC=300, HUB_RETRO_SITTING_MIN_SEC=120,
                                  HUB_RETRO_WATCH_ENABLED=True)
            sitting = retro_sitting.SittingService(cfg, pool_provider=lambda: pool, clock=lambda: clock["t"])
            watch = retro_watch.WatchService(cfg, sitting=sitting, pool_provider=lambda: pool, clock=lambda: clock["t"])
            sitting.tracker.on_input("s" * 36, 0, 128, clock["t"] - 1800)
            for i in range(1, 9):                                           # 29 minutes of play, 200 s apart
                sitting.tracker.on_input("s" * 36, 0, 0 if i % 2 else 128, clock["t"] - 1800 + 200 * i)
            cur = sitting.tracker.current
            cur.game_id, cur.game_title = "a-link-to-the-past", "A Link to the Past"
            assert await sitting._persist(cur)

            # the histogram, added in two flushes, then the floor with the counts it came from
            g = cur.game_id
            for _ in range(2):
                for v in [0.05 + 0.0001 * i for i in range(250)]:
                    k = (g, True, W.bin_of(v)); watch.hist_pending[k] = watch.hist_pending.get(k, 0) + 1
                for v in [0.0003] * 250:
                    k = (g, False, W.bin_of(v)); watch.hist_pending[k] = watch.hist_pending.get(k, 0) + 1
                await watch.flush()
            async with pool.acquire() as conn:
                assert await conn.fetchval("SELECT sum(checks) FROM retro_watch_change_hist WHERE game_id = $1", g) == 1000
                floor = await conn.fetchrow("SELECT * FROM retro_watch_floor WHERE game_id = $1", g)
            assert floor["status"] == "set" and 0.0003 < floor["floor"] <= 0.05
            assert (floor["checks_input"], floor["checks_no_input"], floor["seconds"]) == (500, 500, 2000.0)
            import json as _json
            hist = _json.loads(floor["histogram"]) if isinstance(floor["histogram"], str) else floor["histogram"]
            assert sum(hist["input"]) == 500 and sum(hist["no_input"]) == 500
            assert watch.calibrations[g].floor == floor["floor"]

            # counters on the sitting row
            watch._state_for(cur)
            watch.state.counters.update(ring_checks=900, static_skips=600, checks_over_floor=300, notes_stored=7)
            await watch._write_counters(cur.sitting_id, dict(watch.state.counters))
            async with pool.acquire() as conn:
                row = await conn.fetchrow("SELECT * FROM retro_sitting WHERE sitting_id = $1", cur.sitting_id)
            assert (row["ring_checks"], row["static_skips"], row["checks_over_floor"], row["notes_stored"]) == (900, 600, 300, 7)

            # a look row, then a later post of one of its frames moves its expiry; a verdict defaults to unchecked
            t0 = now.replace(microsecond=0)
            hashes = ["a" * 64, "b" * 64]
            await watch._db("execute", retro_watch.LOOK_INSERT_SQL, "look-1", cur.sitting_id, g, t0, hashes,
                            t0 + timedelta(hours=1), "Link walks north.", "stored", None, 0.08, "look-1", "lease-1",
                            1200, 5.6, 1086, None)
            await watch._db("execute", retro_watch.LOOK_INSERT_SQL, "look-1", cur.sitting_id, g, t0, hashes,
                            t0, "dup", "stored", None, 0.08, "look-1", None, None, None, None, None)    # idempotent
            await watch._db("execute", retro_watch.EXPIRY_SQL, ["b" * 64, "c" * 64], t0 + timedelta(hours=2))
            await watch._db("execute", retro_watch.EXPIRY_SQL, ["a" * 64], t0 + timedelta(minutes=30))   # never earlier
            async with pool.acquire() as conn:
                look = await conn.fetchrow("SELECT * FROM retro_watch_look WHERE look_id = 'look-1'")
            assert look["note"] == "Link walks north." and look["verdict"] == "unchecked" and look["lease_id"] == "lease-1"
            assert look["frames_expire_at"] == t0 + timedelta(hours=2) and list(look["frame_hashes"]) == hashes
            with pytest.raises(asyncpg.CheckViolationError):
                await watch._db("execute", retro_watch.LOOK_INSERT_SQL, "look-2", cur.sitting_id, g, t0, [], None, None,
                                "nothing", None, None, "look-2", None, None, None, None, None)

            # Orion's choices, from the act ledger; Juniper's switch; the lower wins
            async with pool.acquire() as conn:
                await conn.execute("""INSERT INTO grammar_acts VALUES
 ('act-game', 'orion.watch_choice', 'regulate', 'retro_game:a-link-to-the-past', 'watch:off', 'turn-1', now() - interval '1 hour', 'watch off: standing', now()),
 ('act-other', 'orion.look', 'look', 'camera:cam0', NULL, NULL, now(), '', now()),
 ('act-old', 'orion.watch_choice', 'regulate', 'retro_game:tetris', 'watch:quiet', NULL, now() - interval '500 days', 'too old', now())""")
            mode = await watch.refresh_mode(cur)
            assert mode == W.Mode("off", "orion", "watch off: standing")
            async with pool.acquire() as conn:
                await conn.execute("""INSERT INTO grammar_acts VALUES ('act-sit', 'orion.watch_choice', 'regulate',
 'retro_sitting:""" + cur.sitting_id + """', 'watch:on', 'turn-2', now(), 'watch on', now())""")
            assert (await watch.refresh_mode(cur)).mode == "on"
            assert await watch.set_juniper("off") and watch.mode == W.Mode("off", "juniper", None)
            async with pool.acquire() as conn:
                prefs = await conn.fetch("SELECT pref_id, set_by, mode FROM retro_watch_preference ORDER BY set_at")
            assert [p["pref_id"] for p in prefs if p["set_by"] == "orion"] == ["act:act-game", "act:act-sit"]
            await watch.refresh_mode(cur)                                    # copies nothing twice
            async with pool.acquire() as conn:
                assert await conn.fetchval("SELECT count(*) FROM retro_watch_preference") == 3

            # the play row carries the note count, and the chronicle's own query reads it into the label
            clock["t"] += 10
            closed = sitting.tracker.close_all("idle")[0]
            await sitting._finish(closed)
            src = (REPO / "services/orion-durable-runs/app/temporal_self_sources.py").read_text()
            query = re.search(r'SourceQuery\("retro_play", """(.*?)"""', src, re.S).group(1)
            query = query.replace("%(lo)s", "$1").replace("%(hi)s", "$2")
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, now - timedelta(days=1), now + timedelta(days=1))
                # recovery reads notes_stored from the sitting row
                assert await conn.fetch(retro_sitting.RECOVER_MISSING_SQL, 120.0, 7) == []
            async with pool.acquire() as conn:
                play = await conn.fetchrow("SELECT summary, looks FROM retro_play_sessions")
            assert play["looks"] == 7 and play["summary"].endswith("Orion watched: 7 notes.")
            (event,) = S.adapt("retro_play", [dict(r) for r in rows])
            assert rows[0]["looks"] == 7 and event.label.endswith("Orion watched: 7 notes")

            # every statement `dig watch status` sends parses and runs on this schema
            from orion.dig import watch as wt
            sent: list[str] = []

            def rows_fn(sql: str) -> list[dict]:
                sent.append(sql)
                return [{"sitting_id": cur.sitting_id, "game_id": g, "game_title": "A Link to the Past",
                         "started_at": "x", "last_input_at": "y", "player_stated": None}] if sql == wt.OPEN_SITTING_SQL else []
            wt.status(rows=rows_fn)
            async with pool.acquire() as conn:
                await conn.execute("CREATE TABLE want_ledger (decided_at timestamptz, family text, outcome text, reason text, "
                                   "lane text, net double precision, binding_constraint text)")
                for sql in sent + [wt.GAMES_SQL]:
                    await conn.fetch(sql)
            await watch.close()
        finally:
            await pool.close()
    finally:
        await admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await admin.close()
