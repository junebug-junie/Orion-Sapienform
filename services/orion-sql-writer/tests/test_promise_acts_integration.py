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
    prev = session.execute(text(PREVIOUS_TURN_SQL), {"sid": "s1", "corr": "c-new", "ttl": 72 * 3600.0}).mappings().first()
    assert prev["correlation_id"] == "c-old" and prev["is_juniper"] is True
    assert offers_from_made("c-old", traces[0]["made"]) == [(promise_ref("c-old", 1), made[1]["quote"])]


# --- settle classes (#3595) and first-stored promises (#3596), 2026-11-18 -------------------------------

def _trace_table(session):
    from app.models.harness_turn_trace import HarnessTurnTraceSQL

    HarnessTurnTraceSQL.__table__.create(bind=session.connection())


def _trace(session, corr, *, at, made=None, conditions_met=None):
    import json

    reflection = {"promises_made": made or []}
    if conditions_met is not None:
        reflection["promise_conditions_met"] = conditions_met
    session.execute(text(
        "INSERT INTO harness_turn_trace (correlation_id, created_at, updated_at, run_artifact) "
        "VALUES (:c, :at, :at, CAST(:a AS jsonb))"),
        {"c": corr, "at": at.astimezone(timezone.utc).replace(tzinfo=None),
         "a": json.dumps({"reflection": reflection})})


def _item(**kw):
    base = {"quote": "I'll read the three proposals closely once they appear.", "offer": False, "doer": "orion",
            "force": "commits", "act": "read the proposals", "done_when": "a later reply discusses the proposals"}
    base.update(kw)
    return base


def test_a_conditional_promise_whose_condition_never_arose_closes_not_due_not_a_graded_zero(session):
    """#3595 acceptance fixture: Juniper wrote inside the horizon (v1 would close expired 0), but no judged
    reply listed the promise's condition as met, so it was never due: void:not_due, censored."""
    from orion.grammar.promise_acts import promise_void_kind

    _trace_table(session)
    now = datetime.now(timezone.utc)
    t0 = now - timedelta(hours=80)
    _trace(session, "c-cond", at=t0, made=[_item(settles_in="chat", condition="the three proposals appear")])
    act = _open(session, promise_ref("c-cond", 0), t0)
    _chat(session, at=t0 + timedelta(hours=5), corr="c-later")
    _trace(session, "c-later", at=t0 + timedelta(hours=5), conditions_met=[])
    session.commit()

    summary = resolve_due_acts(session, now=now, commit=session.commit)
    row = _row(session, act.act_id)
    assert promise_close_kind(row["outcome_source"], row["outcome_note"]) == "void"
    assert promise_void_kind(row["outcome_note"]) == "not_due"
    assert summary.closed_promises_void == 1 and summary.closed_promises_not_due == 1
    assert summary.closed_promises_expired == 0
    assert _graded(session, act.act_id) == 0 and _posterior_rows(session) == 0


def test_a_conditional_promise_whose_condition_arose_follows_the_v1_rule(session):
    _trace_table(session)
    now = datetime.now(timezone.utc)
    t0 = now - timedelta(hours=80)
    _trace(session, "c-cond2", at=t0, made=[_item(condition="the three proposals appear")])
    act = _open(session, promise_ref("c-cond2", 0), t0)
    _chat(session, at=t0 + timedelta(hours=5), corr="c-met")
    _trace(session, "c-met", at=t0 + timedelta(hours=5), conditions_met=[act.act_id])
    session.commit()

    summary = resolve_due_acts(session, now=now, commit=session.commit)
    row = _row(session, act.act_id)
    assert promise_close_kind(row["outcome_source"], row["outcome_note"]) == "expired"
    assert summary.closed_promises_expired == 1 and summary.closed_promises_not_due == 0


def test_a_condition_met_only_after_the_horizon_is_still_not_due(session):
    _trace_table(session)
    now = datetime.now(timezone.utc)
    t0 = now - timedelta(hours=80)
    _trace(session, "c-cond3", at=t0, made=[_item(condition="the three proposals appear")])
    act = _open(session, promise_ref("c-cond3", 0), t0)
    _chat(session, at=t0 + timedelta(hours=5), corr="c-w")
    _trace(session, "c-after", at=t0 + timedelta(hours=73), conditions_met=[act.act_id])
    session.commit()
    resolve_due_acts(session, now=now, commit=session.commit)
    from orion.grammar.promise_acts import promise_void_kind

    assert promise_void_kind(_row(session, act.act_id)["outcome_note"]) == "not_due"


@pytest.mark.parametrize("settles_in,kind", [("run", "not_chat"), ("absence", "absence_unchecked")])
def test_run_and_absence_promises_never_close_as_a_graded_zero(session, settles_in, kind):
    """#3595 acceptance fixtures: Juniper wrote in the horizon and no reply kept them, yet a run-class
    promise (delivered outside chat) and an abstention (no reply can show it kept) close void."""
    from orion.grammar.promise_acts import is_void_note, promise_void_kind

    _trace_table(session)
    now = datetime.now(timezone.utc)
    t0 = now - timedelta(hours=80)
    _trace(session, "c-class", at=t0, made=[_item(settles_in=settles_in)])
    act = _open(session, promise_ref("c-class", 0), t0)
    _chat(session, at=t0 + timedelta(hours=5), corr="c-wrote2")
    session.commit()

    summary = resolve_due_acts(session, now=now, commit=session.commit)
    row = _row(session, act.act_id)
    assert is_void_note(row["outcome_note"]) and promise_void_kind(row["outcome_note"]) == kind
    assert summary.closed_promises_expired == 0 and summary.closed_promises_void == 1
    assert getattr(summary, f"closed_promises_{kind}") == 1
    assert _graded(session, act.act_id) == 0 and _posterior_rows(session) == 0


def test_a_chat_reply_that_delivers_a_run_promise_still_keeps_it(session):
    _trace_table(session)
    t0 = datetime.now(timezone.utc) - timedelta(hours=5)
    _trace(session, "c-run", at=t0, made=[_item(settles_in="run")])
    act = _open(session, promise_ref("c-run", 0), t0)
    plan = plan_promise_events(correlation_id="c-keep2", session_id="s1", reflection=_reflection(
        promises_kept=[act.act_id], promises_kept_evidence=[PromiseKeptV1(id=act.act_id, quote="the run finished")]))
    apply_grammar_trace_batch(session, plan.events)
    row = _row(session, act.act_id)
    assert promise_close_kind(row["outcome_source"], row["outcome_note"]) == "kept"


def test_a_chat_class_promise_with_its_item_stored_keeps_the_v1_rule(session):
    """An explicit chat item, and a pre-#3595 item (no settles_in/condition), close expired 0 when Juniper wrote."""
    _trace_table(session)
    now = datetime.now(timezone.utc)
    t0 = now - timedelta(hours=80)
    _trace(session, "c-chat", at=t0, made=[_item(settles_in="chat", condition=None), _item(quote="I'll send it.")])
    a = _open(session, promise_ref("c-chat", 0), t0)
    b = _open(session, promise_ref("c-chat", 1), t0)
    _chat(session, at=t0 + timedelta(hours=5), corr="c-w3")
    session.commit()
    summary = resolve_due_acts(session, now=now, commit=session.commit)
    assert summary.closed_promises_expired == 2
    for act in (a, b):
        row = _row(session, act.act_id)
        assert promise_close_kind(row["outcome_source"], row["outcome_note"]) == "expired"


def test_the_offer_window_hides_an_offer_older_than_the_ttl(session):
    """#3595 offer acceptance window: the previous row's offers are shown only while it is younger than
    OFFER_ACCEPT_TTL_SEC (one clock with the promise horizon), on the server's own clock."""
    from orion.grammar.promise_acts import OFFER_ACCEPT_TTL_SEC
    from orion.grammar.promise_view import PREVIOUS_TURN_SQL, offers_visible

    session.execute(text("ALTER TABLE chat_history_log ADD COLUMN session_id VARCHAR"))
    now = datetime.now(timezone.utc)
    for sid, age in (("s-in", OFFER_ACCEPT_TTL_SEC - 60), ("s-out", OFFER_ACCEPT_TTL_SEC + 60)):
        _chat(session, at=now - timedelta(seconds=age), corr=f"c-{sid}")
        session.execute(text("UPDATE chat_history_log SET session_id = :s WHERE correlation_id = :c"),
                        {"s": sid, "c": f"c-{sid}"})
    session.commit()
    seen = {}
    for sid in ("s-in", "s-out"):
        prev = session.execute(text(PREVIOUS_TURN_SQL), {"sid": sid, "corr": "c-new",
                                                         "ttl": OFFER_ACCEPT_TTL_SEC}).mappings().first()
        seen[sid] = offers_visible(dict(prev))
    assert seen == {"s-in": True, "s-out": False}


def test_a_republished_run_artifact_keeps_the_first_stored_promises(session):
    """#3596: refs point into promises_made by index, so a second harness.run.v1 for the same corr must not
    move them. The rest of the artifact updates as before; a first empty list does not pin anything."""
    from app.harness_turn_trace_persist import upsert_harness_turn_trace

    _trace_table(session)
    first = [_item(quote="I'll send the first list tomorrow.")]
    second = [_item(quote="I'll do something else entirely."), _item(quote="And a second thing.")]
    run = {"schema_version": "harness.run.v1", "correlation_id": "c-pub"}
    assert upsert_harness_turn_trace(session, {**run, "final_text": "one", "reflection": {"promises_made": first}})
    assert upsert_harness_turn_trace(session, {**run, "final_text": "two",
                                               "reflection": {"promises_made": second, "tone": "x"}})
    art = session.execute(text("SELECT run_artifact FROM harness_turn_trace WHERE correlation_id = 'c-pub'")).scalar()
    assert art["reflection"]["promises_made"] == first
    assert art["final_text"] == "two" and art["reflection"]["tone"] == "x"
    # An artifact with no reflection at all still keeps the stored list.
    assert upsert_harness_turn_trace(session, {**run, "final_text": "three"})
    art = session.execute(text("SELECT run_artifact FROM harness_turn_trace WHERE correlation_id = 'c-pub'")).scalar()
    assert art["reflection"]["promises_made"] == first and art["final_text"] == "three"
    # ... and the whole stored reflection (tone included), so the list never sits without its judged reflection.
    assert art["reflection"]["tone"] == "x"

    run2 = {"schema_version": "harness.run.v1", "correlation_id": "c-empty-first"}
    assert upsert_harness_turn_trace(session, {**run2, "reflection": {"promises_made": []}})
    assert upsert_harness_turn_trace(session, {**run2, "reflection": {"promises_made": second}})
    art = session.execute(text(
        "SELECT run_artifact FROM harness_turn_trace WHERE correlation_id = 'c-empty-first'")).scalar()
    assert art["reflection"]["promises_made"] == second


def test_the_promising_turns_own_condition_mark_does_not_count(session):
    """The judge on the promising turn itself cannot mark its own new promise's condition met."""
    from orion.grammar.promise_acts import promise_void_kind

    _trace_table(session)
    now = datetime.now(timezone.utc)
    t0 = now - timedelta(hours=80)
    act = _open(session, promise_ref("c-self", 0), t0)
    _trace(session, "c-self", at=t0 + timedelta(seconds=1),
           made=[_item(condition="the proposals appear")], conditions_met=[act.act_id])
    _chat(session, at=t0 + timedelta(hours=5), corr="c-w2")
    session.commit()
    resolve_due_acts(session, now=now, commit=session.commit)
    assert promise_void_kind(_row(session, act.act_id)["outcome_note"]) == "not_due"


def test_settle_class_read_errors_fall_back_and_are_counted():
    """An unreadable item reads as the v1 rule (None); an unreadable condition scan reads as met (graded).
    Both are counted, never silent."""
    from orion.grammar import act_ledger as al

    class Broken:
        def execute(self, *a, **k):
            raise RuntimeError("relation harness_turn_trace does not exist")

    unread: dict = {}
    assert al._promise_item(Broken(), promise_ref("c", 0), unread) is None
    t = datetime.now(timezone.utc)
    assert al._condition_met(Broken(), "act_x", opened_at=t, horizon_at=t, correlation_id="c", unread=unread) is True
    assert unread == {"item": 1, "condition": 1}
