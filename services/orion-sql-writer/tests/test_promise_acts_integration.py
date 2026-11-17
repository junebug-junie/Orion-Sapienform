"""Promise acts (intention ledger D2, shadow) end to end against real Postgres.

The lifecycle Hub's events drive through sql-writer: a promise opens a ``hub.promise`` act; a later
reply's keep closes it 1 (producer); a horizon with Juniper writing and no keep closes it 0 (expired);
a horizon with no Juniper chat turn closes it void (censored). In shadow NONE of these is graded: no
``substrate_action_outcomes`` row, no posterior update, and no cost is filled (a promise's cost is
its reply's).

Opt-in like the other act-ledger tests: ORION_SQL_WRITER_TEST_DATABASE_URI is a scratch database; every
table lives in a fresh throwaway schema (the ``session`` fixture of test_act_ledger_integration).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent))

from orion.grammar.act_ledger import resolve_due_acts  # noqa: E402
from orion.grammar.ledger import apply_grammar_trace_batch  # noqa: E402
from orion.grammar.promise_acts import (  # noqa: E402
    PROMISE_ACTOR,
    VOID_NOTE,
    build_promise_act,
    plan_promise_events,
    promise_close_kind,
    promise_ref,
)
from orion.schemas.harness_finalize import FinalizeReflectionV1, PromiseKeptV1  # noqa: E402
from test_act_ledger_integration import _chat, _row, session  # noqa: E402,F401

pytestmark = pytest.mark.integration


def _open(sess, ref, at):
    act = build_promise_act(ref, parent_act_id=None)
    from orion.grammar.acts import act_opened_event

    apply_grammar_trace_batch(sess, [act_opened_event(act, source_service="orion-hub", source_component="t",
                                                      correlation_id=ref.split(":")[1].split("#")[0], at=at)])
    return act


def _graded(sess, act_id):
    return sess.execute(text("SELECT count(*) FROM substrate_action_outcomes WHERE dispatch_id = :a"),
                        {"a": act_id}).scalar()


def _posterior_rows(sess):
    return sess.execute(text("SELECT count(*) FROM substrate_action_effect_posterior WHERE target_id = :t"),
                        {"t": PROMISE_ACTOR}).scalar()


def _reflection(**kw):
    base = dict(correlation_id="c-keep", thought_event_id="t", substrate_appraisal_id="s", draft_hash="h",
                imperative="i", tone="t", strain_refs=[], alignment_verdict="aligned", alignment_notes=[],
                strain_unresolved=False)
    base.update(kw)
    return FinalizeReflectionV1(**base)


def test_kept_promise_closes_producer_and_is_not_graded_in_shadow(session):
    t0 = datetime.now(timezone.utc) - timedelta(hours=5)
    act = _open(session, promise_ref("c-promise", 0), t0)
    row = _row(session, act.act_id)
    assert row["signal_id"] == "promise_kept" and row["horizon_sec"] == 72 * 3600.0
    assert row["test_before"] == "harness_turn_trace:c-promise#promises_made[0]"
    assert row["predicted"] == pytest.approx(0.5) and row["cold_start"] is True  # one cold bucket, no history

    plan = plan_promise_events(correlation_id="c-keep", session_id="s1", reflection=_reflection(
        promises_kept=[act.act_id], promises_kept_evidence=[PromiseKeptV1(id=act.act_id, quote="here is the read")]))
    apply_grammar_trace_batch(session, plan.events)
    row = _row(session, act.act_id)
    assert row["outcome_value"] == 1.0 and row["outcome_source"] == "producer"
    assert row["outcome_note"].startswith("kept_in:c-keep evidence=reply")
    assert promise_close_kind(row["outcome_source"], row["outcome_note"]) == "kept"
    assert _graded(session, act.act_id) == 0 and _posterior_rows(session) == 0
    assert row["outcome_row_id"] is None and row["cost_json"] is None


def test_horizon_with_juniper_writing_expires_and_without_her_is_void(session):
    now = datetime.now(timezone.utc)
    t0 = now - timedelta(hours=80)  # horizon (72 h) passed 8 h ago
    wrote = _open(session, promise_ref("c-wrote", 0), t0)
    silent = _open(session, promise_ref("c-silent", 1), t0 - timedelta(hours=200))
    # Juniper writes inside the first act's window only.
    _chat(session, at=t0 + timedelta(hours=10), corr="c-later")
    session.commit()

    summary = resolve_due_acts(session, now=now, commit=session.commit)
    assert summary.closed_promises_expired == 1 and summary.closed_promises_void == 1

    w, s = _row(session, wrote.act_id), _row(session, silent.act_id)
    assert w["outcome_value"] == 0.0 and w["outcome_source"] == "expired"
    assert promise_close_kind(w["outcome_source"], w["outcome_note"]) == "expired"
    assert s["outcome_note"] == VOID_NOTE and promise_close_kind(s["outcome_source"], s["outcome_note"]) == "void"
    assert _graded(session, wrote.act_id) == 0 and _graded(session, silent.act_id) == 0
    assert _posterior_rows(session) == 0
    assert w["cost_json"] is None and s["cost_json"] is None


def test_a_keep_after_the_resolver_expired_it_is_ignored(session):
    now = datetime.now(timezone.utc)
    act = _open(session, promise_ref("c-late", 0), now - timedelta(hours=80))
    _chat(session, at=now - timedelta(hours=70), corr="c-mid")
    session.commit()
    resolve_due_acts(session, now=now, commit=session.commit)
    plan = plan_promise_events(correlation_id="c-keep", session_id="s1", reflection=_reflection(
        promises_kept=[act.act_id], promises_kept_evidence=[PromiseKeptV1(id=act.act_id, quote="here it is now")]))
    apply_grammar_trace_batch(session, plan.events)
    assert promise_close_kind(_row(session, act.act_id)["outcome_source"], _row(session, act.act_id)["outcome_note"]) == "expired"


def test_the_hub_view_sql_reads_open_promises_and_the_previous_replys_offers(session):
    """The read Hub runs before each of Juniper's turns (orion/grammar/promise_view.py), on Postgres:
    open acts oldest first with quotes from the turn record, closed and past-horizon acts left out, and
    the previous row's firm offers only when that row answered a Juniper turn."""
    import json

    from orion.grammar.promise_view import (OPEN_PROMISES_SQL, PREVIOUS_TURN_SQL, TRACE_PROMISES_SQL,
                                            offers_from_made, view_from_rows)

    session.execute(text("CREATE TABLE harness_turn_trace (correlation_id TEXT PRIMARY KEY, "
                         "created_at TIMESTAMP DEFAULT now(), run_artifact JSONB)"))
    now = datetime.now(timezone.utc)
    done = "a later reply gives Juniper the digested deep read"
    made = [{"quote": "I'll pull the deep read back once it lands.", "offer": False, "doer": "orion",
             "force": "commits", "act": "x", "done_when": done},
            {"quote": "If you want, I'll re-read it later.", "offer": True, "doer": "orion",
             "force": "commits", "act": "y", "done_when": done}]
    session.execute(text("INSERT INTO harness_turn_trace (correlation_id, run_artifact) VALUES (:c, CAST(:a AS jsonb))"),
                    {"c": "c-old", "a": json.dumps({"reflection": {"promises_made": made}})})
    older = _open(session, promise_ref("c-old", 0), now - timedelta(hours=10))
    _open(session, promise_ref("c-gone", 0), now - timedelta(hours=100))  # past its horizon: not shown
    _chat(session, at=now - timedelta(hours=10), corr="c-old")
    session.commit()

    params = {"actor": PROMISE_ACTOR, "signal": "promise_kept", "n": 8}
    open_rows = [dict(r) for r in session.execute(text(OPEN_PROMISES_SQL), params).mappings()]
    assert [r["act_id"] for r in open_rows] == [older.act_id]
    traces = [dict(r) for r in session.execute(text(TRACE_PROMISES_SQL), {"corrs": ["c-old"]}).mappings()]
    shown, unreadable = view_from_rows(open_rows, traces)
    assert [(p.id, p.quote) for p in shown] == [(older.act_id, made[0]["quote"])] and unreadable == 0

    session.execute(text("ALTER TABLE chat_history_log ADD COLUMN session_id VARCHAR"))
    session.execute(text("UPDATE chat_history_log SET session_id = 's1'"))
    prev = session.execute(text(PREVIOUS_TURN_SQL), {"sid": "s1", "corr": "c-new"}).mappings().first()
    assert prev["correlation_id"] == "c-old" and prev["is_juniper"] is True
    assert offers_from_made("c-old", traces[0]["made"]) == [(promise_ref("c-old", 1), made[1]["quote"])]
