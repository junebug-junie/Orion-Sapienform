"""Orion's now on real Postgres (integration design 2026-11-15, Part A patch 1): the migration, the three
new source reads, salience at insert, the now's SQL, the cite check and the metric's SQL, end to end.

Lifecycle: readings and a Prime thought land on the act ledger -> the chronicle stores them as
experiences with salience and surprisal -> Prime's now shows the surprising reading (never its own
thought) and an experience_shown row is written -> Prime cites it plus an id it was never shown ->
one experience_use row is stored, the other is dropped -> the integration metric moves from 0.

Set ORION_ADMISSION_TEST_DSN to a disposable database (CI provides one). Each test makes its own schema.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT), str(TESTS), str(SERVICE_ROOT / "evals")]

DSN = os.getenv("ORION_ADMISSION_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")

from app.temporal_self_chronicle import ChronicleConfig, Chronicler, SalienceBook  # noqa: E402
from app.temporal_self_sources import SourceReader  # noqa: E402
from app.temporal_self_store import ChronicleStore  # noqa: E402
from orion.integration.organs import load_organs  # noqa: E402
from orion.temporal_self import ReducerConfig  # noqa: E402

TZ = "America/Denver"

# Live column types (information_schema, 2026-11-15), only the columns the reads use plus NOT NULLs.
SOURCE_TABLES = """
CREATE TABLE grammar_acts (act_id varchar PRIMARY KEY, trace_id varchar NOT NULL, level varchar NOT NULL,
  actor varchar NOT NULL, verb varchar NOT NULL, object_ref text, correlation_id varchar,
  opened_at timestamptz, outcome_value double precision, outcome_note text, surprise_nats double precision,
  closed_at timestamptz, created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE world_pulse_read_seed (seed_id text PRIMARY KEY, kind text NOT NULL DEFAULT 'reading',
  run_id text NOT NULL DEFAULT 'r', url text NOT NULL DEFAULT '', title text NOT NULL DEFAULT '', request_id uuid,
  created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE prediction_exceedance (exceedance_id bigserial PRIMARY KEY, series_id varchar(120) NOT NULL,
  origin_ts timestamptz NOT NULL, issued_at timestamptz NOT NULL, name varchar(60) NOT NULL,
  threshold double precision NOT NULL, window_start timestamptz NOT NULL, window_end timestamptz NOT NULL,
  probability double precision, n_paths integer NOT NULL, served_baseline boolean NOT NULL DEFAULT false,
  scored_at timestamptz, hit_raw boolean, hit_mean boolean, max_raw double precision);
CREATE TABLE prediction_joint_surprise (joint_id bigserial PRIMARY KEY, family varchar(32) NOT NULL,
  horizon_min integer NOT NULL, issued_at timestamptz NOT NULL, origin_ts timestamptz NOT NULL,
  target_ts timestamptz NOT NULL, status varchar(16) NOT NULL, scored_at timestamptz, p double precision);
"""


def with_db(fn):
    from uuid import uuid4

    from psycopg import AsyncConnection
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    import temporal_self_fixture_db as fdb

    async def go():
        schema = "exp_now_test_" + uuid4().hex[:10]
        async with await AsyncConnection.connect(DSN, autocommit=True) as conn:
            await conn.execute(f'CREATE SCHEMA "{schema}"')
        try:
            async with AsyncConnectionPool(DSN, min_size=1, max_size=4, open=False, kwargs={
                    "autocommit": True, "prepare_threshold": 0, "row_factory": dict_row,
                    "options": f"-c search_path={schema}"}) as pool:
                async with pool.connection() as conn:
                    await fdb.create_schema(conn)
                    for stmt in fdb.statements(SOURCE_TABLES):
                        await conn.execute(stmt, prepare=False)
                await fn(pool)
        finally:
            async with await AsyncConnection.connect(DSN, autocommit=True) as conn:
                await conn.execute(f'DROP SCHEMA "{schema}" CASCADE')
    asyncio.run(go())


async def seed(pool, t0: datetime) -> str:
    """25 ordinary readings with FALLING surprisal (the newest is the smallest) and one surprising reading in
    the middle of them, one Prime thought, two scored exceedances."""
    async with pool.connection() as conn:
        for i in range(26):
            surprising = i == 25
            at = t0 + timedelta(minutes=22, seconds=30) if surprising else t0 + timedelta(minutes=i)
            await conn.execute(
                "INSERT INTO grammar_acts (act_id, trace_id, level, actor, verb, object_ref, correlation_id, opened_at,"
                " outcome_value, outcome_note, surprise_nats, closed_at) VALUES (%s, 't', 'act', 'hub.world_pulse_read',"
                " 'look', %s, %s, %s, %s, %s, %s, %s)",
                (f"act_{i:024x}", f"https://example.org/{i}", f"00000000-0000-0000-0000-{i:012x}",
                 at - timedelta(minutes=2), 0 if surprising else 1,
                 "no_read_evidence" if surprising else "stage1_completed", 3.0 if surprising else 0.01 - i * 1e-4, at))
            await conn.execute("INSERT INTO world_pulse_read_seed (seed_id, url, title) VALUES (%s, %s, %s)",
                               (f"reading:00000000-0000-0000-0000-{i:012x}", f"https://example.org/{i}", f"Article {i}"))
        await conn.execute(
            "INSERT INTO grammar_acts (act_id, trace_id, level, actor, verb, object_ref, opened_at, outcome_value,"
            " outcome_note, surprise_nats, closed_at) VALUES ('act_" + "f" * 24 + "', 't', 'act', 'prime', 'infer',"
            " 'prime_thought:abc', %s, 1, 'woke: quiet | no move', 0.0004, %s)",
            (t0 + timedelta(minutes=30), t0 + timedelta(minutes=30, seconds=14)))
        for hit in (False, True):
            w0 = t0 + timedelta(minutes=5 if hit else 0)
            await conn.execute(
                "INSERT INTO prediction_exceedance (series_id, origin_ts, issued_at, name, threshold, window_start,"
                " window_end, probability, n_paths, scored_at, hit_raw, max_raw) VALUES ('athena.cabinet_temp_c', %s, %s,"
                " %s, 32, %s, %s, 0, 40, %s, %s, 33)",
                (w0, w0, "shed" if hit else "guard", w0, w0 + timedelta(minutes=30), w0 + timedelta(minutes=31), hit))
        await conn.execute(
            "INSERT INTO prediction_joint_surprise (family, horizon_min, issued_at, origin_ts, target_ts, status, p)"
            " VALUES ('family_a', 15, %s, %s, %s, 'scored', 0.4)", (t0, t0, t0 + timedelta(minutes=10)))
    return f"reading_turn:act_{25:024x}"


def test_lifecycle_on_postgres(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")  # main's settings; the pool is the test's
    from app import main

    real_now = datetime.now(timezone.utc)
    t0 = (real_now - timedelta(hours=3)).replace(second=0, microsecond=0)

    async def scenario(pool):
        surprising = await seed(pool, t0)
        clock = {"now": t0}
        store = ChronicleStore(pool)
        ch = Chronicler(store=store, reader=SourceReader(pool, TZ),
                        # backfill_days=1: the seed starts 3 h before the real clock, which is the previous LOCAL day
                        # (America/Denver) whenever the test runs 07:00-10:00 UTC. With 0 the chronicle starts at
                        # today's local midnight and never reads those rows (CI flake, 2026-11-16 10:00 UTC).
                        cfg=ChronicleConfig(reducer=ReducerConfig(tz_name=TZ), read_lag_sec=300.0, backfill_days=1),
                        now=lambda: clock["now"],
                        # The rule under test is set here: the live registry gives reading_turn no magnitude.
                        salience=SalienceBook({**load_organs().magnitude_rule, "reading_turn": "surprise_nats"},
                                              store.magnitude_history))
        # Step the chronicle forward to just before the real clock, an hour at a time.
        while clock["now"] < real_now - timedelta(minutes=1):
            clock["now"] = min(clock["now"] + timedelta(hours=1), real_now - timedelta(minutes=1))
            assert (await ch.step())["error"] is None

        async with pool.connection() as conn:
            rows = {r["event_id"]: r for r in await (await conn.execute(
                "SELECT event_id, source_kind, salience, surprise_nats, label, related_refs FROM temporal_self_event"
                " WHERE source_kind IN ('reading_turn', 'prime_thought', 'prediction_surprise')")).fetchall()}
        # 1. emitted: three new kinds stored, with salience and surprisal from the sources' own rows
        assert {r["source_kind"] for r in rows.values()} == {"reading_turn", "prime_thought", "prediction_surprise"}
        assert rows[surprising]["salience"] == 1.0 and rows[surprising]["surprise_nats"] == 3.0
        assert rows[surprising]["label"] == "no_read_evidence: Article 25"
        assert rows[f"reading_turn:act_{0:024x}"]["salience"] is None  # no history yet: NULL, never 0
        assert rows[f"reading_turn:act_{24:024x}"]["salience"] == 0.0  # newest and smallest: the rest point
        crossed = [r for r in rows.values() if r["source_kind"] == "prediction_surprise" and r["surprise_nats"] > 4]
        assert len(crossed) == 1 and crossed[0]["related_refs"] == ["prediction_joint_surprise:1"]

        # 2. Prime's now shows the surprising reading first; never Prime's own thought; one shown row.
        # The read cuts NOW_COMMIT_MARGIN_SEC below the clock: age the stored rows past it (time passing).
        async with pool.connection() as conn:
            await conn.execute("UPDATE temporal_self_event SET ingested_at = ingested_at - interval '10 minutes'")
        monkeypatch.setattr(main, "_checkpointer_cm", pool)
        since = (real_now - timedelta(hours=2)).isoformat()
        now = await main.temporal_self_now(organ="prime", since=since, run_ref="prime_thought:test")
        assert now["items"][0]["event_id"] == surprising and now["items"][0]["salience"] == 1.0
        assert all(i["organ"] != "prime" for i in now["items"])
        assert sum(1 for i in now["items"] if i["organ"] == "reading") == 2
        async with pool.connection() as conn:
            shown = await (await conn.execute("SELECT reader_organ, event_ids FROM experience_shown")).fetchall()
        assert shown == [{"reader_organ": "prime", "event_ids": [i["event_id"] for i in now["items"]]}]
        # an explicit since= never moves the cursor; the first default read starts 24 h back and sees them.
        # #3498: a read nobody closes moves nothing, so the next read offers the same rows again; once the
        # reader closes it (no cites), the next default read starts at its cut-off and sees nothing new.
        first = await main.temporal_self_now(organ="prime")
        assert first["items"] and first["items"][0]["event_id"] == surprising
        unclosed = await main.temporal_self_now(organ="prime")  # first was never closed: offered again
        assert unclosed["items"][0]["event_id"] == surprising
        # closing the newer read moves the cursor to its cut-off; the next default read sees nothing new
        closed = await main.temporal_self_now_use({"shown_id": unclosed["shown_id"],
                                                   "citing_ref": "prime_thought:t1", "cites": []})
        assert closed["uses"] == []
        again = await main.temporal_self_now(organ="prime")
        assert again["items"] == [] and again["since"] == unclosed["as_of"]
        # the older read closing late never moves the cursor backwards (max over closed reads)
        await main.temporal_self_now_use({"shown_id": first["shown_id"], "citing_ref": "prime_thought:t0",
                                          "cites": []})
        assert (await main.temporal_self_now(organ="prime"))["since"] == unclosed["as_of"]
        # coverage's split (orion/integration/coverage.py NOW_READS_SQL): closed with no cite vs never closed
        from orion.integration.coverage import NOW_READS_SQL

        async with pool.connection() as conn:
            reads = await (await conn.execute(NOW_READS_SQL, {"lo": real_now - timedelta(days=2),
                                                              "hi": real_now + timedelta(hours=1)})).fetchone()
        # 5 reads so far: the since= read (unclosed), first + unclosed (closed, no cite), again + the last (unclosed)
        assert (reads["reads"], reads["reads_no_cite"], reads["reads_unclosed"]) == (5, 2, 3)

        before = await main.temporal_self_integration(hours=6)
        assert before["rate"] == 0.0 and before["closed_organs"] == []

        # 3-4. Prime cites it and an id it was never shown: one stored, one dropped
        res = await main.temporal_self_now_use({"shown_id": now["shown_id"], "citing_ref": "prime_thought:test",
                                                "cites": [f"experience:{surprising}", "dream_cycle:x"]})
        assert [u["event_id"] for u in res["uses"]] == [surprising] and res["rejected_not_shown"] == ["dream_cycle:x"]
        async with pool.connection() as conn:
            uses = await (await conn.execute("SELECT event_id, citing_organ FROM experience_use")).fetchall()
        assert uses == [{"event_id": surprising, "citing_organ": "prime"}]
        # the 3-cite cap holds per shown row across calls (row-locked)
        others = [i["event_id"] for i in now["items"] if i["event_id"] != surprising][:3]
        more = await main.temporal_self_now_use({"shown_id": now["shown_id"], "citing_ref": "prime_thought:test",
                                                 "cites": others})
        assert len(more["uses"]) == 2 and more["rejected_over_cap"] == others[2:]

        # 5. the metric moved, by the metric's own SQL
        after = await main.temporal_self_integration(hours=6)
        assert after["used_by_other"] == 3 and "reading" in after["closed_organs"] and "prime" not in after["closed_organs"]
        assert after["rate"] == pytest.approx(3 / after["experiences"])

    with_db(scenario)
