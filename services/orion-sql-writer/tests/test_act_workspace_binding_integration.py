"""Rung 3 against real Postgres: an act is bound to the workspace winner it answers, and its close
writes Orion's own non-final `acted` verdict on that loop (attention_loop_outcome, actor='orion').

Before rung 3, actor='orion' rows came only from the shed loop (0 in the 48 h to 2026-10-21) and
no act ever carried a workspace_ref. Opt-in like the other DB tests (ORION_SQL_WRITER_TEST_DATABASE_URI,
a scratch database; everything lives in a throwaway schema).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models.cockpit_turn_sighting import CockpitTurnSightingSQL
from app.models.grammar_trace import GrammarActSQL, GrammarEventSQL, GrammarTraceSQL  # noqa: F401
from orion.grammar import act_ledger
from orion.grammar.act_ledger import resolve_due_acts
from orion.grammar.acts import act_closed_event, act_opened_event, build_act
from orion.grammar.ledger import apply_grammar_trace_batch
from test_act_ledger_integration import L10_DDL

pytestmark = pytest.mark.integration

# Live columns, \d on conjourney 2026-10-21.
RUNG3_DDL = """
CREATE TABLE substrate_attention_broadcast_log (
    log_id TEXT PRIMARY KEY, generated_at TIMESTAMPTZ NOT NULL, projection_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE attention_loop_outcome (
    outcome_id TEXT PRIMARY KEY, loop_id TEXT NOT NULL, theme_key TEXT NOT NULL, verdict TEXT NOT NULL,
    actor TEXT NOT NULL DEFAULT 'juniper', note TEXT NOT NULL DEFAULT '',
    salience_at_close DOUBLE PRECISION NOT NULL DEFAULT 0, weights_version TEXT NOT NULL DEFAULT 'seed-v1',
    features_at_close JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL,
    enqueued_at TIMESTAMPTZ NOT NULL DEFAULT now())
"""


@pytest.fixture
def session():
    import os

    uri = os.environ["ORION_SQL_WRITER_TEST_DATABASE_URI"]
    schema = f"act_ws_test_{uuid.uuid4().hex[:10]}"
    admin = create_engine(uri)
    with admin.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_engine(uri, connect_args={"options": f"-csearch_path={schema}"})
    with engine.begin() as conn:
        for stmt in (L10_DDL + ";" + RUNG3_DDL).split(";"):
            if stmt.strip():
                conn.execute(text(stmt))
    Base.metadata.create_all(bind=engine, tables=[GrammarActSQL.__table__, GrammarEventSQL.__table__,
                                                  GrammarTraceSQL.__table__, CockpitTurnSightingSQL.__table__])
    sess = sessionmaker(bind=engine)()
    act_ledger.configure_act_workspace_binding(True)
    try:
        yield sess
    finally:
        act_ledger.configure_act_workspace_binding(True)
        sess.rollback()
        sess.close()
        engine.dispose()
        with admin.begin() as conn:
            conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        admin.dispose()


def _broadcast(sess, *, at, node="world:chat", action="watch", dwell=3, loop="open-loop-chat1", salience=0.9):
    projection = {
        "selected_action_type": action, "selected_open_loop_id": loop if action != "none" else None,
        "attended_node_ids": [node] if action != "none" else [], "dwell_ticks": dwell,
        "generated_at": at.isoformat(), "frame": {"open_loops": [{"id": loop, "salience": salience}]},
    }
    sess.execute(text("INSERT INTO substrate_attention_broadcast_log (log_id, generated_at, projection_json) "
                      "VALUES (:id, :at, CAST(:p AS jsonb))"),
                 {"id": f"broadcast-{uuid.uuid4().hex[:8]}", "at": at, "p": json.dumps(projection)})


def _reply(key):
    return build_act(actor="hub.chat_reply", verb="say", key=key, signal_id="juniper_replied",
                     horizon_sec=1800.0, basis="median gap")


def _open(sess, act, at, corr):
    apply_grammar_trace_batch(sess, [act_opened_event(act, source_service="orion-hub", source_component="t",
                                                      correlation_id=corr, at=at)])


def _juniper_turn(sess, at):
    sess.execute(text("INSERT INTO chat_history_log (id, correlation_id, source, prompt, response, created_at) "
                      "VALUES (:id, :c, 'hub_orion', 'hi', 'r', :at)"),
                 {"id": uuid.uuid4().hex, "c": f"turn-{uuid.uuid4().hex[:6]}",
                  "at": at.astimezone(timezone.utc).replace(tzinfo=None)})


def _ws(sess, act_id):
    return sess.execute(text("SELECT workspace_ref FROM grammar_acts WHERE act_id = :a"), {"a": act_id}).scalar()


def _orion_verdicts(sess):
    return sess.execute(text("SELECT loop_id, verdict, actor, salience_at_close, features_at_close "
                             "FROM attention_loop_outcome WHERE actor = 'orion'")).mappings().all()


def test_reply_under_a_chat_winner_closes_that_loop_as_acted(session):
    """Fails on pre-rung-3 code: no workspace_ref was ever stamped and no orion verdict written."""
    now = datetime.now(timezone.utc)
    t0 = now - timedelta(minutes=10)
    _broadcast(session, at=t0 - timedelta(seconds=20))
    act = _reply("c1")
    _open(session, act, t0, "c1")
    assert _ws(session, act.act_id) == "open-loop-chat1"

    _juniper_turn(session, t0 + timedelta(seconds=120))
    assert resolve_due_acts(session, now=now).closed_replied == 1
    rows = _orion_verdicts(session)
    assert len(rows) == 1
    row = rows[0]
    assert (row["loop_id"], row["verdict"], row["actor"]) == ("open-loop-chat1", "acted", "orion")
    assert row["salience_at_close"] == pytest.approx(0.9)
    assert row["features_at_close"]["act_id"] == act.act_id and row["features_at_close"]["outcome_value"] == 1.0

    # Idempotent: a re-run closes nothing new and writes no second verdict.
    resolve_due_acts(session, now=now)
    assert len(_orion_verdicts(session)) == 1


def test_orion_never_writes_a_terminal_verdict(session):
    now = datetime.now(timezone.utc)
    t0 = now - timedelta(minutes=10)
    _broadcast(session, at=t0 - timedelta(seconds=5))
    act = _reply("c2")
    _open(session, act, t0, "c2")
    apply_grammar_trace_batch(session, [act_closed_event(act, produced=False, source_service="orion-hub",
                                                         source_component="t", at=t0 + timedelta(seconds=30))])
    verdicts = {r["verdict"] for r in session.execute(text("SELECT verdict FROM attention_loop_outcome")).mappings()}
    assert verdicts == {"acted"}


@pytest.mark.parametrize("kw,why", [
    (dict(node="node:substrate.biometrics"), "a winner this actor does not answer"),
    (dict(dwell=1), "a one-tick flicker"),
    (dict(action="none"), "a calm tick"),
])
def test_no_binding_without_a_bindable_winner(session, kw, why):
    now = datetime.now(timezone.utc)
    t0 = now - timedelta(minutes=10)
    _broadcast(session, at=t0 - timedelta(seconds=5), **kw)
    act = _reply(f"c-{uuid.uuid4().hex[:4]}")
    _open(session, act, t0, "cx")
    assert _ws(session, act.act_id) is None, why


def test_a_winner_older_than_90s_does_not_bind(session):
    t0 = datetime.now(timezone.utc) - timedelta(minutes=10)
    _broadcast(session, at=t0 - timedelta(seconds=120))
    act = _reply("c-stale")
    _open(session, act, t0, "c-stale")
    assert _ws(session, act.act_id) is None


def test_other_actors_are_never_bound(session):
    t0 = datetime.now(timezone.utc) - timedelta(minutes=10)
    _broadcast(session, at=t0 - timedelta(seconds=5))
    act = build_act(actor="thought.painting", verb="make", key="run-x", signal_id="artifact_produced",
                    horizon_sec=5400.0, basis="deadline")
    apply_grammar_trace_batch(session, [act_opened_event(act, source_service="orion-thought", source_component="t", at=t0)])
    assert _ws(session, act.act_id) is None


def test_kill_switch_stops_stamping_and_verdicts(session):
    now = datetime.now(timezone.utc)
    t0 = now - timedelta(minutes=10)
    _broadcast(session, at=t0 - timedelta(seconds=5))
    act_ledger.configure_act_workspace_binding(False)
    act = _reply("c-off")
    _open(session, act, t0, "c-off")
    _juniper_turn(session, t0 + timedelta(seconds=60))
    resolve_due_acts(session, now=now)
    assert _ws(session, act.act_id) is None and _orion_verdicts(session) == []


def test_missing_broadcast_table_never_blocks_the_act(session):
    session.execute(text("DROP TABLE substrate_attention_broadcast_log"))
    act = _reply("c-notable")
    _open(session, act, datetime.now(timezone.utc) - timedelta(minutes=1), "c-notable")
    assert session.execute(text("SELECT count(*) FROM grammar_acts WHERE act_id = :a"), {"a": act.act_id}).scalar() == 1
