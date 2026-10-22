"""Rung 3 follow-up against a real Postgres: the act-episode resolver and Hub's pull SQL.

Needs ORION_TEST_PG_URI (a scratch database; tables are created and dropped here). The rung-3 CI job
starts a postgres:16 service and sets ORION_TEST_PG_MUST_RUN=1, so a missing URI fails there instead
of skipping.
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from orion.autonomy import act_episodes  # noqa: E402
from orion.world_pulse_read import pull  # noqa: E402

URI = os.environ.get("ORION_TEST_PG_URI")
if not URI:
    if os.environ.get("ORION_TEST_PG_MUST_RUN") == "1":
        raise RuntimeError("ORION_TEST_PG_URI is required in CI for the rung-3 reading pull PG tests")
    pytest.skip("ORION_TEST_PG_URI not set", allow_module_level=True)

sqlalchemy = pytest.importorskip("sqlalchemy")
asyncpg = pytest.importorskip("asyncpg")
from sqlalchemy import create_engine, text  # noqa: E402

T0 = datetime(2026, 10, 21, 6, 0, tzinfo=timezone.utc)

# The #2830 table as it is live today (no object_ref / void_reason / resolved_at).
OLD_DDL = """
CREATE TABLE substrate_act_episodes (
    episode_id text PRIMARY KEY, act_label text NOT NULL, verb text NOT NULL, actor text NOT NULL,
    signal_id text NOT NULL, dispatch_kind text NOT NULL, target_id text NOT NULL, arm text NOT NULL,
    path text NOT NULL, decided_at timestamptz NOT NULL, horizon_at timestamptz NOT NULL,
    dispatch_frame_id text, expected jsonb NOT NULL DEFAULT '{}'::jsonb, outcome_value double precision,
    created_at timestamptz NOT NULL DEFAULT now())
"""
GRAMMAR_ACTS = """
CREATE TABLE grammar_acts (act_id text PRIMARY KEY, actor text, verb text, signal_id text, test_before text,
    opened_at timestamptz, closed_at timestamptz, outcome_value double precision)
"""


@pytest.fixture()
def engine():
    eng = create_engine(URI.replace("postgresql://", "postgresql+psycopg2://", 1))
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS substrate_act_episodes, grammar_acts"))
        c.execute(text(OLD_DDL))
        c.execute(text(GRAMMAR_ACTS))
    yield eng
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS substrate_act_episodes, grammar_acts"))
    eng.dispose()


def _row(eid, arm, obj, decided=T0, horizon_sec=12000, checked=None):
    return {"episode_id": eid, "act_label": "look:hub.world_pulse_read:artifact_produced", "verb": "look",
            "actor": "hub.world_pulse_read", "signal_id": "artifact_produced", "dispatch_kind": "observe",
            "target_id": "queue:world_pulse_read", "arm": arm, "path": "proven_effect", "decided_at": decided,
            "horizon_at": decided + timedelta(seconds=horizon_sec), "dispatch_frame_id": "f",
            "expected": {} if checked is None else {"checked_object": checked},
            "outcome_value": None, "object_ref": obj}


def _act(c, act_id, seed, closed, value, opened=None):
    c.execute(text("INSERT INTO grammar_acts VALUES (:i, 'hub.world_pulse_read', 'look', 'artifact_produced', "
                   ":t, :o, :c, :v)"),
              {"i": act_id, "t": f"read_seed:{seed}", "o": opened or closed - timedelta(seconds=600),
               "c": closed, "v": value})


def test_ensure_table_upgrades_the_live_2830_table_and_resolves_both_arms(engine) -> None:
    with engine.begin() as c:
        act_episodes.ensure_table(c)          # adds the three columns to the existing table
        for r in (_row("treated-read", "treated", "read_seed:a"), _row("control-unread", "control", "read_seed:b"),
                  _row("control-read-late", "control", "read_seed:c"), _row("void", "treated", "read_seed:d"),
                  _row("no-object", "treated", None), _row("read-before", "treated", "read_seed:e"),
                  _row("failed-read", "treated", "read_seed:f")):
            act_episodes.insert_decision(c, r)
        act_episodes.void_decision(c, "void", "doer_refused:no_candidate")
        _act(c, "a1", "a", T0 + timedelta(hours=1), 1.0)
        _act(c, "c1", "c", T0 + timedelta(seconds=12001), 1.0)       # after the horizon
        _act(c, "d1", "d", T0 + timedelta(hours=1), 1.0)             # void: never resolved
        _act(c, "e1", "e", T0 - timedelta(minutes=5), 1.0)           # before the decision
        _act(c, "f1", "f", T0 + timedelta(hours=1), 0.0)             # read failed
        assert act_episodes.resolve_due(c, now=T0 + timedelta(seconds=11999)) == []   # nothing due yet
        assert act_episodes.in_flight(c, actor="hub.world_pulse_read", now=T0 + timedelta(hours=1))
        got = {e: v for e, _, v in act_episodes.resolve_due(c, now=T0 + timedelta(seconds=12001))}
        assert got == {"treated-read": 1.0, "control-unread": 0.0, "control-read-late": 0.0,
                       "read-before": 0.0, "failed-read": 0.0}
        assert act_episodes.control_stats(c, act_label="look:hub.world_pulse_read:artifact_produced") == (2, 0)
        assert act_episodes.resolve_due(c, now=T0 + timedelta(days=1)) == []           # idempotent
        void = c.execute(text("SELECT void_reason, horizon_at = decided_at, outcome_value "
                              "FROM substrate_act_episodes WHERE episode_id = 'void'")).one()
        assert tuple(void) == ("doer_refused:no_candidate", True, None)


def test_a_void_decision_releases_its_in_flight_hold(engine) -> None:
    with engine.begin() as c:
        act_episodes.ensure_table(c)
        act_episodes.insert_decision(c, _row("v", "treated", "read_seed:z"))
        assert act_episodes.in_flight(c, actor="hub.world_pulse_read", now=T0 + timedelta(minutes=1))
        act_episodes.void_decision(c, "v", "doer_refused:arousal_strained")
        assert not act_episodes.in_flight(c, actor="hub.world_pulse_read", now=T0 + timedelta(minutes=1))


def test_the_migration_matches_the_lazy_columns(engine) -> None:
    sql = (REPO / "services/orion-sql-db/manual_migration_reading_pull_v1.sql").read_text()
    for col in ("object_ref", "void_reason", "resolved_at", "pull_dispatch_id", "pull_arm", "pulled_at"):
        assert f"ADD COLUMN IF NOT EXISTS {col}" in sql
        assert col in act_episodes.DDL or col in pull.ENSURE_PULL_COLUMNS_SQL


# --- Hub's pull SQL (asyncpg, as Hub runs it) ------------------------------------------------------

async def _hub_sql() -> None:
    from orion.world_pulse_read.durable import READING_DURABLE_SQL
    from orion.world_pulse_read.queue import ENSURE_TABLE_SQL, GENERAL_READING_SQL

    conn = await asyncpg.connect(URI)
    try:
        await conn.execute("DROP TABLE IF EXISTS reading_durable_turn, world_pulse_read_seed")
        await conn.execute(ENSURE_TABLE_SQL)
        await conn.execute(GENERAL_READING_SQL)
        await conn.execute(READING_DURABLE_SQL)
        await conn.execute(pull.ENSURE_PULL_COLUMNS_SQL)
        await conn.execute(pull.ENSURE_PULL_COLUMNS_SQL)               # idempotent
        ins = ("INSERT INTO world_pulse_read_seed (seed_id, kind, run_id, url, priority, status, created_at) "
               "VALUES ($1, 'digest_item', 'r', $2, $3, 'pending', $4)")
        await conn.execute(ins, "inflight", "https://x/1", 0, T0 - timedelta(hours=3))
        await conn.execute(ins, "older-low", "https://x/2", 10, T0 - timedelta(hours=2))
        await conn.execute(ins, "head", "https://x/3", 0, T0 - timedelta(hours=1))
        await conn.execute(ins, "next", "https://x/4", 0, T0)
        await conn.execute("INSERT INTO reading_durable_turn (seed_id, stage, attempt, run_id, request_json) "
                           "VALUES ('inflight', 1, 0, 'run-1', '{}'::jsonb)")
        # CLAIM_SQL's own order, minus the in-flight read: priority first, then age.
        assert (await conn.fetchrow(pull.SELECT_PULL_CANDIDATE_SQL))["seed_id"] == "head"
        row = await conn.fetchrow(pull.MARK_PULL_SQL, "head", "d:t", "treated")
        assert row["seed_id"] == "head"
        st = await conn.fetchrow("SELECT status, pull_arm, claimed_at IS NOT NULL AS c FROM world_pulse_read_seed "
                                 "WHERE seed_id='head'")
        assert tuple(st) == ("claimed", "treated", True)
        assert await conn.fetchrow(pull.MARK_PULL_SQL, "head", "d:again", "treated") is None   # once only
        assert (await conn.fetchrow(pull.SELECT_PULL_CANDIDATE_SQL))["seed_id"] == "next"
        await conn.fetchrow(pull.MARK_PULL_SQL, "next", "d:c", "control")
        st = await conn.fetchrow("SELECT status, pull_arm FROM world_pulse_read_seed WHERE seed_id='next'")
        assert tuple(st) == ("pending", "control")                     # the lane may still read it
        assert (await conn.fetchrow(pull.SELECT_PULL_CANDIDATE_SQL))["seed_id"] == "older-low"
        start, end = pull.utc_day_bounds(datetime.now(timezone.utc))
        assert await conn.fetchval(pull.COUNT_STARTED_PULLS_SQL, start, end) == 1   # control is not a read
    finally:
        await conn.execute("DROP TABLE IF EXISTS reading_durable_turn, world_pulse_read_seed")
        await conn.close()


def test_hub_pull_sql_follows_the_lane_order_and_marks_once() -> None:
    asyncio.run(_hub_sql())


def test_an_unconfirmed_decision_is_voided_and_old_painting_controls_never_count(engine) -> None:
    """Review 2026-10-22: a row the doer never confirmed (RPC error) must not resolve as an episode,
    and #2830's no-object painting control rows (0 by construction) must not enter the control rate."""
    with engine.begin() as c:
        act_episodes.ensure_table(c)
        act_episodes.insert_decision(c, _row("lost", "control", None, checked="read_seed:q"))
        act_episodes.insert_decision(c, _row("ok", "control", "read_seed:r"))
        _act(c, "q1", "q", T0 + timedelta(hours=1), 1.0)   # the lane read the unmarked seed anyway
        paint = {**_row("paint-old", "control", None), "act_label": "make:thought.painting:artifact_produced",
                 "verb": "make", "actor": "thought.painting", "outcome_value": 0.0}
        act_episodes.insert_decision(c, paint)
        got = act_episodes.resolve_due(c, now=T0 + timedelta(seconds=12001))
        assert [(e, v) for e, _, v in got] == [("ok", 0.0)]
        lost = c.execute(text("SELECT void_reason, outcome_value FROM substrate_act_episodes "
                              "WHERE episode_id = 'lost'")).one()
        assert tuple(lost) == ("unconfirmed_by_doer", None)
        assert act_episodes.control_stats(c, act_label="make:thought.painting:artifact_produced") == (0, 0)
        assert act_episodes.arm_stats(c, act_label="look:hub.world_pulse_read:artifact_produced",
                                      arm="control") == (1, 0)


def test_a_started_pull_holds_the_route_in_sql(engine) -> None:
    """Review 2026-10-22: the store's in-flight SQL sees a started pull on any path."""
    sys.path.insert(0, str(REPO / "services" / "orion-execution-dispatch-runtime"))
    from app.store import ExecutionDispatchRuntimeStore

    with engine.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS substrate_dispatch_results"))
        c.execute(text("CREATE TABLE substrate_dispatch_results (dispatch_kind text, target_id text, "
                       "status text, result_json jsonb, created_at timestamptz DEFAULT now())"))
        c.execute(text("INSERT INTO substrate_dispatch_results VALUES ('observe', 'queue:world_pulse_read', "
                       "'succeeded', '{\"settlement\": {\"state\": \"started\"}}'::jsonb, now())"))
    store = ExecutionDispatchRuntimeStore(URI.replace("postgresql://", "postgresql+psycopg2://", 1))
    kw = dict(actor="hub.world_pulse_read", dispatch_kind="observe", target_id="queue:world_pulse_read",
              include_open_acts=False)
    try:
        assert store.act_in_flight(**kw, started_within_sec=12000.0) is True
        assert store.act_in_flight(**kw, started_within_sec=None) is False
    finally:
        with engine.begin() as c:
            c.execute(text("DROP TABLE IF EXISTS substrate_dispatch_results"))
