"""Game company patch 4 against real Postgres: a game line answers world:play, and Juniper's engagement with it is graded.

1. A line said while ``world:play`` holds the workspace is stamped with that loop, and its close writes Orion's
   non-final ``acted`` verdict on it; never on a reading loop (spec section 7, "Attention and action meet").
2. A line said while the page was told "Juniper" is a ``juniper_replied`` act; the horizon resolver grades it from her
   thumb, her chat-box reply within 2 min, or her chat turn within 2 min (orion/grammar/line_engagement.py). A guest's
   thumb never counts as hers. That close is the family weight's engagement record (orion/physiology/weights.py (b)).

Opt-in like the other DB tests (ORION_SQL_WRITER_TEST_DATABASE_URI, a scratch database; a throwaway schema).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from orion.grammar.act_ledger import resolve_due_acts
from orion.grammar.acts import act_opened_event, build_act
from orion.grammar.ledger import apply_grammar_trace_batch
from orion.grammar.line_engagement import LINE_ENGAGE_HORIZON_SEC
from orion.physiology.appetite import ActRecord
from orion.physiology.weights import evidence_for
from test_act_workspace_binding_integration import _broadcast, _juniper_turn, _orion_verdicts, _ws, session  # noqa: F401

pytestmark = pytest.mark.integration

# The columns the resolver reads (services/orion-sql-db/manual_migration_retro_lines_v1.sql).
LINE_DDL = """
CREATE TABLE retro_line (line_id TEXT PRIMARY KEY, thumb TEXT, thumb_by TEXT, thumb_at TIMESTAMPTZ,
                         replied_at TIMESTAMPTZ, replied_by TEXT)
"""


def _line_act(line_id: str):
    return build_act(actor="hub.retro_commentary", verb="say", key=line_id, signal_id="juniper_replied",
                     horizon_sec=LINE_ENGAGE_HORIZON_SEC, basis="t", object_ref="retro_sitting:S1",
                     test_before=f"retro_line:{line_id}")


def _say(sess, line_id, at, **row):
    if not sess.execute(text("SELECT to_regclass('retro_line')")).scalar():
        sess.execute(text(LINE_DDL))
    sess.execute(text("INSERT INTO retro_line (line_id, thumb, thumb_by, thumb_at, replied_at, replied_by) "
                      "VALUES (:l, :t, :tb, :ta, :ra, :rb)"),
                 {"l": line_id, "t": row.get("thumb"), "tb": row.get("thumb_by"), "ta": row.get("thumb_at"),
                  "ra": row.get("replied_at"), "rb": row.get("replied_by")})
    act = _line_act(line_id)
    apply_grammar_trace_batch(sess, [act_opened_event(act, source_service="orion-hub", source_component="t",
                                                      correlation_id=line_id, at=at)])
    return act


def _closed(sess, act_id):
    return sess.execute(text("SELECT outcome_value, outcome_source, outcome_note, closed_at FROM grammar_acts "
                             "WHERE act_id = :a"), {"a": act_id}).mappings().first()


def test_a_line_under_a_play_winner_is_bound_and_juniper_thumb_grades_it(session):
    now = datetime.now(timezone.utc)
    t0 = now - timedelta(minutes=20)
    _broadcast(session, at=t0 - timedelta(seconds=20), node="world:play", loop="open-loop-play1")
    lid = str(uuid.uuid4())
    act = _say(session, lid, t0, thumb="up", thumb_by="juniper", thumb_at=t0 + timedelta(seconds=40))
    assert _ws(session, act.act_id) == "open-loop-play1"
    # Not graded before its horizon, even with a thumb in.
    resolve_due_acts(session, now=t0 + timedelta(seconds=60))
    assert _closed(session, act.act_id)["closed_at"] is None
    s = resolve_due_acts(session, now=now)
    row = _closed(session, act.act_id)
    assert (row["outcome_value"], row["outcome_source"]) == (1.0, "resolver") and "thumb up" in row["outcome_note"]
    assert s.closed_replied == 1
    [v] = _orion_verdicts(session)
    assert (v["loop_id"], v["verdict"]) == ("open-loop-play1", "acted")
    # The family weight reads it as Juniper's engagement.
    rec = ActRecord(act_id=act.act_id, family="hub.retro_commentary", opened_at=t0, closed_at=row["closed_at"],
                    outcome_value=row["outcome_value"], outcome_source=row["outcome_source"],
                    signal_id="juniper_replied")
    ev = evidence_for("hub.retro_commentary", [rec], [], since=t0 - timedelta(days=1), until=now + timedelta(days=1))
    assert ev.engagement_n == 1 and ev.engagement_rate == 1.0


def test_a_line_under_a_reading_winner_is_never_bound(session):
    now = datetime.now(timezone.utc)
    t0 = now - timedelta(minutes=20)
    _broadcast(session, at=t0 - timedelta(seconds=20), node="world:reading", loop="open-loop-read1")
    act = _say(session, str(uuid.uuid4()), t0)
    assert _ws(session, act.act_id) is None
    resolve_due_acts(session, now=now)
    assert _orion_verdicts(session) == []


@pytest.mark.parametrize("row,turn_after,value,note", [
    (dict(thumb="up", thumb_by="guest", thumb_at=10), None, 0.0, "no thumb from Juniper"),
    (dict(thumb="down", thumb_by="juniper", thumb_at=30, replied_at=20, replied_by="juniper"), None, 0.0,
     "thumb down"),
    (dict(replied_at=90, replied_by="juniper"), None, 1.0, "game chat box"),
    (dict(replied_at=90, replied_by="guest"), None, 0.0, "no thumb from Juniper"),
    (dict(replied_at=200, replied_by="juniper"), None, 0.0, "no thumb from Juniper"),
    ({}, 60, 1.0, "chat turn started within 2 min"),
    ({}, 200, 0.0, "no thumb from Juniper"),
])
def test_juniper_engagement_grades(session, row, turn_after, value, note):
    now = datetime.now(timezone.utc)
    t0 = now - timedelta(minutes=20)
    vals = {k: (t0 + timedelta(seconds=v) if k.endswith("_at") else v) for k, v in row.items()}
    act = _say(session, str(uuid.uuid4()), t0, **vals)
    if turn_after is not None:
        _juniper_turn(session, t0 + timedelta(seconds=turn_after))
    resolve_due_acts(session, now=now)
    got = _closed(session, act.act_id)
    assert (got["outcome_value"], got["outcome_source"]) == (value, "resolver") and note in got["outcome_note"]


def test_a_missing_line_table_leaves_the_act_open_for_the_next_tick(session):
    now = datetime.now(timezone.utc)
    t0 = now - timedelta(minutes=20)
    act = _line_act(str(uuid.uuid4()))
    apply_grammar_trace_batch(session, [act_opened_event(act, source_service="orion-hub", source_component="t",
                                                         correlation_id="x", at=t0)])
    resolve_due_acts(session, now=now)
    assert _closed(session, act.act_id)["closed_at"] is None
