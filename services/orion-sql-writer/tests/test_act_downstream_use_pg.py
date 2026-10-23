"""Act downstream use (app/act_downstream_use.py) against a REAL Postgres.

``ORION_TEST_PG_URI`` must be a throwaway database: these tests DROP the source tables.

    docker run -d --rm --name pg -e POSTGRES_PASSWORD=t -p 127.0.0.1:55511:5432 postgres:16-alpine
    ORION_TEST_PG_URI=postgresql://postgres:t@127.0.0.1:55511/postgres pytest tests/test_act_downstream_use_pg.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]
for p in (str(SERVICE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

PG = os.getenv("ORION_TEST_PG_URI", "")
pytestmark = pytest.mark.skipif(not PG, reason="ORION_TEST_PG_URI not set (throwaway Postgres only)")

NOW = datetime.now(timezone.utc).replace(microsecond=0)
T0 = NOW - timedelta(days=6)   # every act's 72 h window is closed
SOURCES = ("grammar_acts", "reverie_visual_chain", "chat_history_log", "retro_play_sessions", "thought_decision",
           "temporal_self_arc", "act_downstream_use")


@pytest.fixture()
def pg():
    from sqlalchemy import create_engine, text

    from app.models.act_downstream_use import ActDownstreamUseSQL

    # A non-UTC session: thought_decision.created_at and chat_history_log.created_at are naive UTC,
    # so a missing AT TIME ZONE 'UTC' would shift them by 6-7 h and fail the window tests.
    engine = create_engine(PG, connect_args={"options": "-c timezone=America/Denver"})
    with engine.begin() as conn:
        for t in SOURCES:
            conn.execute(text(f"DROP TABLE IF EXISTS {t}"))
        # Only the columns the rollup reads; the real tables are wider.
        conn.execute(text("CREATE TABLE grammar_acts (act_id varchar PRIMARY KEY, actor varchar, verb varchar, "
                          "object_ref text, test_before text, correlation_id varchar, opened_at timestamptz, "
                          "closed_at timestamptz, created_at timestamptz NOT NULL DEFAULT now())"))
        conn.execute(text("CREATE TABLE reverie_visual_chain (chain_id text PRIMARY KEY, created_at timestamptz, "
                          "chain_json jsonb)"))
        conn.execute(text("CREATE TABLE chat_history_log (id varchar PRIMARY KEY, created_at timestamp)"))
        conn.execute(text("CREATE TABLE retro_play_sessions (session_id text PRIMARY KEY, dispatch_id text, "
                          "created_at timestamptz)"))
        conn.execute(text("CREATE TABLE thought_decision (id varchar PRIMARY KEY, created_at timestamp, "
                          "day_citations jsonb)"))
        conn.execute(text("CREATE TABLE temporal_self_arc (arc_id text PRIMARY KEY, arc_json jsonb)"))
    ActDownstreamUseSQL.__table__.create(engine)
    yield engine
    engine.dispose()


def _exec(engine, sql, **kw):
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(text(sql), kw)


def act(engine, act_id, actor, *, opened, closed=None, object_ref=None, test_before=None, corr=None, verb="make"):
    _exec(engine, "INSERT INTO grammar_acts (act_id, actor, verb, object_ref, test_before, correlation_id, "
                  "opened_at, closed_at) VALUES (:a, :ac, :v, :o, :t, :c, :op, :cl)",
          a=act_id, ac=actor, v=verb, o=object_ref, t=test_before, c=corr, op=opened, cl=closed)


def arc(engine, arc_id, refs):
    _exec(engine, "INSERT INTO temporal_self_arc VALUES (:a, CAST(:j AS jsonb))",
          a=arc_id, j=json.dumps({"evidence_refs": refs}))


def chat_row(engine, row_id, at):
    _exec(engine, "INSERT INTO chat_history_log VALUES (:i, :t)", i=row_id,
          t=at.astimezone(timezone.utc).replace(tzinfo=None))


def decision(engine, dec_id, at, cited):
    # thought_decision.created_at is naive UTC, as live.
    _exec(engine, "INSERT INTO thought_decision VALUES (:i, :t, CAST(:j AS jsonb))",
          i=dec_id, t=at.astimezone(timezone.utc).replace(tzinfo=None),
          j=json.dumps({"cited_valid": cited, "label": "verified"}))


def rows(engine) -> dict:
    from sqlalchemy import text

    with engine.begin() as conn:
        return {r.act_id: r for r in conn.execute(text("SELECT * FROM act_downstream_use"))}


def test_join_paths_window_and_statuses(pg) -> None:
    from app.act_downstream_use import family_report, run_one_rollup

    h = timedelta(hours=1)
    # Painting: output chain cited by an arc that a decision cites 5 h later -> used.
    act(pg, "p1", "thought.painting", opened=T0, closed=T0 + h, test_before="visual_dispatch:d1")
    _exec(pg, "INSERT INTO reverie_visual_chain VALUES ('c1', :t, CAST(:j AS jsonb))", t=T0 + h,
          j=json.dumps({"run_request": {"dispatch_id": "d1"}}))
    # Painting that produced nothing -> no_output, used false.
    act(pg, "p2", "thought.painting", opened=T0, closed=T0 + h, test_before="visual_dispatch:d2")
    # Chat reply: its row is evidence, but the only citing decision is 80 h later -> not used.
    act(pg, "r1", "hub.chat_reply", opened=T0, closed=T0 + h, corr="turn-1", verb="say")
    for row_id, at in (("turn-1", T0 + h), ("turn-2", T0 + 11 * h), ("out-1", T0 + h)):
        chat_row(pg, row_id, at)
    # Chat reply cited BEFORE it opened (an older arc holding the same row) -> not used.
    act(pg, "r2", "hub.chat_reply", opened=T0 + 10 * h, closed=T0 + 11 * h, corr="turn-2", verb="say")
    # Retro co-play names its output directly; leisure joins on the dispatch id.
    act(pg, "g1", "hub.retro_coplay", opened=T0, closed=T0 + h, test_before="retro_play_sessions:s:1.000",
        corr="s:1.000", verb="play")
    act(pg, "g2", "hub.retro_leisure", opened=T0, closed=T0 + h, corr="disp-1", verb="play")
    _exec(pg, "INSERT INTO retro_play_sessions VALUES ('s:1.000', NULL, :t), ('lz', 'disp-1', :t)", t=T0 + h)
    # Families with no path: never "0% used".
    act(pg, "w1", "hub.world_pulse_read", opened=T0, closed=T0 + h, object_ref="https://x", verb="look")
    act(pg, "o1", "hub.outreach", opened=T0, closed=T0 + h, corr="out-1", verb="say")

    arc(pg, "A", ["reverie_visual_chain:c1", "chat_history_log:turn-1", "retro_play_sessions:s:1.000",
                  "chat_history_log:out-1"])
    arc(pg, "B", ["chat_history_log:turn-2", "retro_play_sessions:lz"])
    decision(pg, "d-early", T0 + 9 * h, ["arc:B"])           # before r2 opened
    decision(pg, "d-5h", T0 + 6 * h, ["arc:A"])
    decision(pg, "d-80h", T0 + 81 * h, ["arc:A"])            # outside the 72 h window for r1
    decision(pg, "d-uncited", T0 + 2 * h, [])
    decision(pg, "d-garbled", T0 + 2 * h, ["arc:missing", "not-an-arc"])

    # d-5h is within r1's window, so make r1's first use only the late one by removing it from A.
    _exec(pg, "UPDATE temporal_self_arc SET arc_json = CAST(:j AS jsonb) WHERE arc_id = 'A'",
          j=json.dumps({"evidence_refs": ["reverie_visual_chain:c1", "retro_play_sessions:s:1.000",
                                          "chat_history_log:out-1"]}))
    arc(pg, "C", ["chat_history_log:turn-1"])
    decision(pg, "d-80h-c", T0 + 81 * h, ["arc:C"])

    assert run_one_rollup(pg) == 8
    r = rows(pg)
    assert (r["p1"].link_status, r["p1"].used, r["p1"].site, r["p1"].decision_id, r["p1"].arc_id) == (
        "linked", True, "thought_decision.day_citations", "d-5h", "A")
    assert r["p1"].evidence_ref == "reverie_visual_chain:c1" and r["p1"].first_use_at == T0 + 6 * h
    assert (r["p2"].link_status, r["p2"].used) == ("no_output", False)
    assert (r["r1"].link_status, r["r1"].used) == ("linked", False)
    assert (r["r2"].link_status, r["r2"].used) == ("linked", False)
    assert (r["g1"].link_path, r["g1"].used) == ("direct", True)
    assert (r["g2"].link_path, r["g2"].used, r["g2"].output_ref) == ("retro_leisure", True, "retro_play_sessions:lz")
    for a in ("w1", "o1"):
        assert (r[a].link_status, r[a].used) == ("no_link_path", None)
    assert all(x.window_closed for x in r.values())

    rep = {x["actor"]: x for x in family_report(pg, days=14)}
    assert rep["hub.world_pulse_read"]["status"] == "no_link_path" and rep["hub.world_pulse_read"]["use_rate"] is None
    assert rep["hub.outreach"]["use_rate"] is None
    paint = rep["thought.painting"]
    assert (paint["acts"], paint["use_rate"], paint["use_rate_all"]) == (2, 1.0, 0.5)   # the failed act counts in _all
    assert paint["credit_closed"] == round(1 / 3, 3)   # arc A holds 3 refs: arc-granular credit
    assert rep["hub.chat_reply"]["use_rate"] == 0.0
    assert rep["hub.outreach"]["use_rate_all"] is None and rep["hub.outreach"]["credit_closed"] is None


def test_open_window_is_flagged_and_incremental_pass_flips_used(pg) -> None:
    from app.act_downstream_use import run_one_rollup

    opened = NOW - timedelta(hours=2)
    act(pg, "r", "hub.chat_reply", opened=opened, closed=opened + timedelta(minutes=1), corr="t", verb="say")
    chat_row(pg, "t", opened)
    arc(pg, "A", ["chat_history_log:t"])
    run_one_rollup(pg)
    first = rows(pg)["r"]
    assert first.used is False and first.window_closed is False   # not a final "not used"
    decision(pg, "d", NOW - timedelta(minutes=5), ["arc:A"])
    run_one_rollup(pg)   # table non-empty: incremental pass
    again = rows(pg)["r"]
    assert again.used is True and again.decision_id == "d"
    assert run_one_rollup(pg) == 1   # idempotent upsert, one row per act
    assert len(rows(pg)) == 1


def test_a_stalled_writer_is_caught_up_not_frozen(pg) -> None:
    # Review finding: a wall-clock-only cutoff froze rows the writer missed (crash loop, outage)
    # at window_closed=false forever. Any open-window row and any act with no row is recomputed.
    from sqlalchemy import text

    from app.act_downstream_use import run_one_rollup

    old = NOW - timedelta(days=8)
    act(pg, "stale", "hub.chat_reply", opened=old, closed=old, corr="ts", verb="say")
    chat_row(pg, "ts", old)
    arc(pg, "A", ["chat_history_log:ts"])
    decision(pg, "d", old + timedelta(hours=3), ["arc:A"])
    act(pg, "seed", "hub.world_pulse_read", opened=NOW, verb="look")   # makes the table non-empty
    _exec(pg, "INSERT INTO act_downstream_use (act_id, actor, verb, link_status, window_closed, refreshed_at) "
              "VALUES ('stale', 'hub.chat_reply', 'say', 'linked', false, now() - interval '7 days'), "
              "('seed', 'hub.world_pulse_read', 'look', 'no_link_path', false, now())")
    act(pg, "late", "hub.chat_reply", opened=old, closed=old, corr="ts", verb="say")   # replayed late, no row
    run_one_rollup(pg)
    r = rows(pg)
    assert (r["stale"].used, r["stale"].window_closed) == (True, True)
    assert (r["late"].used, r["late"].window_closed) == (True, True)


def test_close_before_open_painting_two_outputs_and_chain_time_guard(pg) -> None:
    from app.act_downstream_use import run_one_rollup

    h = timedelta(hours=1)
    # A close-before-open act (opened_at NULL, live has one): started_at falls back to created_at.
    _exec(pg, "INSERT INTO grammar_acts (act_id, actor, verb, test_before, created_at, closed_at) VALUES "
              "('nul', 'thought.painting', 'make', 'visual_dispatch:dn', :c, :c)", c=T0)
    # Two chains share a dispatch id: the used one ('z') is not the alphabetically first output.
    act(pg, "two", "thought.painting", opened=T0, closed=T0 + h, test_before="visual_dispatch:d2")
    # A chain from a much older run with the same dispatch id is not this act's output.
    act(pg, "old", "thought.painting", opened=T0, closed=T0 + h, test_before="visual_dispatch:dold")
    for chain, disp, at in (("cn", "dn", T0), ("a", "d2", T0 + h), ("z", "d2", T0 + h),
                            ("co", "dold", T0 - 5 * h)):
        _exec(pg, "INSERT INTO reverie_visual_chain VALUES (:c, :t, CAST(:j AS jsonb))", c=chain, t=at,
              j=json.dumps({"run_request": {"dispatch_id": disp}}))
    arc(pg, "A", ["reverie_visual_chain:z", "reverie_visual_chain:cn", "reverie_visual_chain:co"])
    decision(pg, "d", T0 + 3 * h, ["arc:A"])
    # A decision before the output row existed cannot have used it.
    act(pg, "pre", "thought.painting", opened=T0, closed=T0 + 4 * h, test_before="visual_dispatch:dp")
    _exec(pg, "INSERT INTO reverie_visual_chain VALUES ('cp', :t, CAST(:j AS jsonb))", t=T0 + 4 * h,
          j=json.dumps({"run_request": {"dispatch_id": "dp"}}))
    arc(pg, "P", ["reverie_visual_chain:cp"])
    decision(pg, "dp-early", T0 + 2 * h, ["arc:P"])
    run_one_rollup(pg)
    r = rows(pg)
    assert (r["nul"].used, r["nul"].started_at) == (True, T0)
    assert (r["two"].used, r["two"].evidence_ref, r["two"].link_path) == (True, "reverie_visual_chain:z", "painting")
    assert (r["old"].link_status, r["old"].used) == ("no_output", False)
    assert (r["pre"].link_status, r["pre"].used) == ("linked", False)
