"""output_used acts against real Postgres (act consequence bets, Patch 1a): a producer "made it" close leaves
the act open; the resolver grades a use by another organ (experience_use or act_downstream_use), a real 0
when the output was observable but unused, and a void (no posterior update) when no use could be seen.

Opt-in like the other DB tests: set ORION_SQL_WRITER_TEST_DATABASE_URI to a scratch database.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models.act_downstream_use import ActDownstreamUseSQL
from app.models.cockpit_turn_sighting import CockpitTurnSightingSQL
from app.models.grammar_trace import (  # noqa: F401 -- registers the grammar tables on Base
    GrammarActSQL,
    GrammarEventSQL,
    GrammarTraceSQL,
)
from orion.grammar.act_ledger import EXPIRY_GRACE_SEC, resolve_due_acts
from orion.grammar.acts import act_closed_event, act_opened_event, build_act
from orion.grammar.ledger import apply_grammar_trace_batch
from orion.grammar.output_used import UNOBSERVABLE_NOTE, UNUSED_NOTE

pytestmark = pytest.mark.integration

# The live columns of the two L10 tables (\d on conjourney, 2026-10-16).
L10_DDL = """
CREATE TABLE substrate_action_outcomes (
    id BIGSERIAL PRIMARY KEY, dispatch_id TEXT NOT NULL, dispatch_frame_id TEXT NOT NULL,
    feedback_frame_id TEXT NOT NULL, dispatch_kind TEXT NOT NULL, target_id TEXT NOT NULL,
    signal_id TEXT NOT NULL, direction TEXT NOT NULL, observed_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(), baseline DOUBLE PRECISION NOT NULL,
    observed_after DOUBLE PRECISION NOT NULL, observed_delta DOUBLE PRECISION NOT NULL,
    predicted_delta DOUBLE PRECISION NOT NULL, prediction_error DOUBLE PRECISION NOT NULL,
    surprise_nats DOUBLE PRECISION NOT NULL, posterior_mean DOUBLE PRECISION NOT NULL,
    posterior_variance DOUBLE PRECISION NOT NULL, posterior_n INTEGER NOT NULL,
    co_predictors INTEGER NOT NULL DEFAULT 0, latency_ms DOUBLE PRECISION, claim_upheld BOOLEAN,
    arm TEXT NOT NULL DEFAULT 'dispatched', baseline_bin SMALLINT NOT NULL DEFAULT 0,
    frame_dispatch_count INTEGER NOT NULL DEFAULT 0);
CREATE UNIQUE INDEX substrate_action_outcomes_dispatch_frame_signal_uidx
    ON substrate_action_outcomes (dispatch_id, signal_id, dispatch_frame_id);
CREATE TABLE substrate_action_effect_posterior (
    dispatch_kind TEXT NOT NULL, target_id TEXT NOT NULL, signal_id TEXT NOT NULL,
    posterior_mean DOUBLE PRECISION NOT NULL, posterior_variance DOUBLE PRECISION NOT NULL,
    posterior_n INTEGER NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    baseline_bin SMALLINT NOT NULL DEFAULT 0,
    PRIMARY KEY (dispatch_kind, target_id, signal_id, baseline_bin));
CREATE TABLE temporal_self_event (
    event_id TEXT PRIMARY KEY, source_kind TEXT NOT NULL, correlation_id TEXT,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE experience_use (
    use_id TEXT PRIMARY KEY, event_id TEXT NOT NULL, citing_organ TEXT NOT NULL, citing_ref TEXT NOT NULL,
    shown_id TEXT NOT NULL, cited_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE chat_history_log (
    id VARCHAR PRIMARY KEY, correlation_id VARCHAR, source VARCHAR, prompt TEXT, response TEXT,
    client_meta JSONB, created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now());
"""


@pytest.fixture
def session():
    import os

    uri = os.environ["ORION_SQL_WRITER_TEST_DATABASE_URI"]
    schema = f"act_test_{uuid.uuid4().hex[:10]}"
    admin = create_engine(uri)
    with admin.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_engine(uri, connect_args={"options": f"-csearch_path={schema}"})
    with engine.begin() as conn:
        for stmt in L10_DDL.split(";"):
            if stmt.strip():
                conn.execute(text(stmt))
    Base.metadata.create_all(
        bind=engine,
        tables=[GrammarActSQL.__table__, GrammarEventSQL.__table__, GrammarTraceSQL.__table__,
                CockpitTurnSightingSQL.__table__, ActDownstreamUseSQL.__table__],
    )
    sess = sessionmaker(bind=engine)()
    try:
        yield sess
    finally:
        sess.rollback()
        sess.close()
        engine.dispose()
        with admin.begin() as conn:
            conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        admin.dispose()



HORIZON = 3600.0


def _open(session, key, *, at, corr):
    act = build_act(actor="thought.painting", verb="make", key=key, signal_id="output_used",
                    horizon_sec=HORIZON, basis="24 h use window")
    apply_grammar_trace_batch(session, [act_opened_event(act, source_service="orion-thought", source_component="t",
                                                         at=at, correlation_id=corr)])
    return act


def _producer_close(session, act, *, produced, at, corr):
    apply_grammar_trace_batch(session, [act_closed_event(act, produced=produced, source_service="orion-thought",
                                                         source_component="t", at=at, correlation_id=corr)])


def _row(session, act_id):
    return session.execute(text("SELECT * FROM grammar_acts WHERE act_id = :a"), {"a": act_id}).mappings().one()


def _experience(session, corr, *, kind="visual_run"):
    eid = uuid.uuid4().hex
    session.execute(text("INSERT INTO temporal_self_event (event_id, source_kind, correlation_id) VALUES (:e, :k, :c)"),
                    {"e": eid, "k": kind, "c": corr})
    return eid


def _cite(session, event_id, *, at, organ="reverie"):  # thought.painting's organ is "painting"
    session.execute(text("INSERT INTO experience_use (use_id, event_id, citing_organ, citing_ref, shown_id, cited_at) "
                         "VALUES (:u, :e, :o, 'r', 's', :at)"),
                    {"u": uuid.uuid4().hex, "e": event_id, "o": organ, "at": at})


def _downstream(session, act_id, *, status, used=None, first_use_at=None):
    session.execute(text(
        "INSERT INTO act_downstream_use (act_id, actor, verb, link_status, used, first_use_at, site, window_closed, "
        "refreshed_at) VALUES (:a, 'thought.painting', 'make', :s, :u, :f, 'day_citations', false, now())"),
        {"a": act_id, "s": status, "u": used, "f": first_use_at})


def _outcomes(session):
    return session.execute(text("SELECT count(*) FROM substrate_action_outcomes WHERE signal_id = 'output_used'")).scalar()


def test_producer_made_close_leaves_act_open_and_nothing_made_grades_zero(session):
    t0 = datetime.now(timezone.utc) - timedelta(minutes=10)
    made = _open(session, "made", at=t0, corr="c-made")
    _producer_close(session, made, produced=True, at=t0 + timedelta(seconds=30), corr="c-made")
    assert _row(session, made.act_id)["closed_at"] is None

    empty = _open(session, "empty", at=t0, corr="c-empty")
    _producer_close(session, empty, produced=False, at=t0 + timedelta(seconds=30), corr="c-empty")
    row = _row(session, empty.act_id)
    assert row["closed_at"] is not None and row["outcome_value"] == 0.0 and row["outcome_note"].startswith("no_output")
    assert _outcomes(session) == 1


def test_use_by_another_organ_closes_one_before_horizon(session):
    t0 = datetime.now(timezone.utc) - timedelta(minutes=10)
    act = _open(session, "cited", at=t0, corr="c-cited")
    _producer_close(session, act, produced=True, at=t0 + timedelta(seconds=30), corr="c-cited")
    eid = _experience(session, "c-cited")
    # A cite of the act's own action_outcome row is not a use of its output.
    _cite(session, _experience(session, "c-cited", kind="action_outcome"), at=t0 + timedelta(minutes=1))
    assert resolve_due_acts(session).closed_output_used == 0
    _cite(session, eid, at=t0 + timedelta(minutes=5))
    summary = resolve_due_acts(session)
    row = _row(session, act.act_id)
    assert summary.closed_output_used == 1
    assert row["outcome_value"] == 1.0 and row["outcome_source"] == "resolver" and "reverie" in row["outcome_note"]
    assert row["surprise_nats"] is not None and _outcomes(session) == 1


def test_downstream_use_path_also_counts(session):
    t0 = datetime.now(timezone.utc) - timedelta(minutes=10)
    act = _open(session, "downstream", at=t0, corr=None)
    _downstream(session, act.act_id, status="linked", used=True, first_use_at=t0 + timedelta(minutes=3))
    assert resolve_due_acts(session).closed_output_used == 1
    assert _row(session, act.act_id)["outcome_value"] == 1.0


def test_horizon_observable_unused_is_zero_unobservable_is_void(session):
    t0 = datetime.now(timezone.utc) - timedelta(seconds=HORIZON + EXPIRY_GRACE_SEC + 60)
    seen = _open(session, "seen", at=t0, corr="c-seen")
    _experience(session, "c-seen")
    linked = _open(session, "linked", at=t0, corr=None)
    _downstream(session, linked.act_id, status="linked", used=False)
    unseen = _open(session, "unseen", at=t0, corr="c-unseen")
    _downstream(session, unseen.act_id, status="no_link_path")
    # Not yet due: an act inside its horizon with no use stays open.
    fresh = _open(session, "fresh", at=datetime.now(timezone.utc), corr="c-fresh")
    _experience(session, "c-fresh")

    summary = resolve_due_acts(session)
    assert (summary.closed_output_used, summary.closed_output_unused, summary.closed_output_unobservable) == (0, 2, 1)
    assert _row(session, seen.act_id)["outcome_note"] == UNUSED_NOTE
    assert _row(session, linked.act_id)["outcome_note"] == UNUSED_NOTE
    void = _row(session, unseen.act_id)
    assert void["outcome_note"] == UNOBSERVABLE_NOTE and void["surprise_nats"] is None
    assert _row(session, fresh.act_id)["closed_at"] is None
    # Two graded zeros reach L10; the void does not.
    assert _outcomes(session) == 2


def test_output_used_acts_never_reach_the_generic_expiry(session):
    # The main due list excludes output_used, so an unobservable act is voided, never "no producer close".
    t0 = datetime.now(timezone.utc) - timedelta(seconds=HORIZON + EXPIRY_GRACE_SEC + 60)
    act = _open(session, "generic", at=t0, corr=None)
    summary = resolve_due_acts(session)
    assert summary.closed_expired == 0 and summary.closed_output_unobservable == 1
    assert _row(session, act.act_id)["outcome_note"] == UNOBSERVABLE_NOTE


def test_turn_rows_of_other_organs_neither_credit_nor_make_observable(session):
    # One turn's correlation_id also carries attention/body rows. A cite of those is not a use of the
    # painting, and their presence does not make the painting observable.
    t0 = datetime.now(timezone.utc) - timedelta(seconds=HORIZON + EXPIRY_GRACE_SEC + 60)
    act = _open(session, "turn", at=t0, corr="c-turn")
    _cite(session, _experience(session, "c-turn", kind="attention_row"), at=t0 + timedelta(minutes=1))
    _experience(session, "c-turn", kind="gpu_wait")
    summary = resolve_due_acts(session)
    assert summary.closed_output_used == 0 and summary.closed_output_unobservable == 1
    assert _row(session, act.act_id)["outcome_note"] == UNOBSERVABLE_NOTE


def test_a_cite_by_the_acts_own_organ_is_not_a_use(session):
    t0 = datetime.now(timezone.utc) - timedelta(minutes=10)
    act = _open(session, "self", at=t0, corr="c-self")
    _cite(session, _experience(session, "c-self"), at=t0 + timedelta(minutes=1), organ="painting")
    assert resolve_due_acts(session).closed_output_used == 0
    assert _row(session, act.act_id)["closed_at"] is None


def test_failed_producer_close_keeps_its_failure_note(session):
    t0 = datetime.now(timezone.utc) - timedelta(minutes=10)
    act = _open(session, "failed", at=t0, corr="c-failed")
    ev = act_closed_event(act, produced=False, source_service="orion-thought", source_component="t",
                          at=t0 + timedelta(seconds=30), correlation_id="c-failed", note="failed: llm timeout")
    apply_grammar_trace_batch(session, [ev])
    assert _row(session, act.act_id)["outcome_note"] == "failed: llm timeout"


def test_deferred_close_keeps_cost_and_correlation_id(session):
    t0 = datetime.now(timezone.utc) - timedelta(minutes=10)
    act = _open(session, "cost", at=t0, corr=None)
    _producer_close(session, act, produced=True, at=t0 + timedelta(seconds=90), corr="c-late")
    row = _row(session, act.act_id)
    assert row["closed_at"] is None and row["correlation_id"] == "c-late"
    assert row["cost_json"]["seconds"] == pytest.approx(90.0, abs=1.0)
    # The resolver's close keeps that cost instead of re-filling it out to the horizon.
    _cite(session, _experience(session, "c-late"), at=t0 + timedelta(minutes=5))
    resolve_due_acts(session)
    row = _row(session, act.act_id)
    assert row["outcome_value"] == 1.0 and row["cost_json"]["seconds"] == pytest.approx(90.0, abs=1.0)


def test_close_before_an_open_that_never_arrives_still_closes(session):
    old = datetime.now(timezone.utc) - timedelta(seconds=HORIZON + EXPIRY_GRACE_SEC + 600)
    act = build_act(actor="thought.painting", verb="make", key="orphan", signal_id="output_used",
                    horizon_sec=HORIZON, basis="24 h use window")
    apply_grammar_trace_batch(session, [act_closed_event(act, produced=True, source_service="orion-thought",
                                                         source_component="t", at=old, correlation_id=None)])
    session.execute(text("UPDATE grammar_acts SET created_at = :t WHERE act_id = :a"), {"t": old, "a": act.act_id})
    row = _row(session, act.act_id)
    assert row["opened_at"] is None and row["closed_at"] is None
    assert resolve_due_acts(session).closed_output_unobservable == 1


def test_persisted_nothing_made_close_is_recovered_not_expired(session):
    # The producer's 0 close reached grammar_events but its act-ledger write was lost (lock fight).
    t0 = datetime.now(timezone.utc) - timedelta(seconds=HORIZON + EXPIRY_GRACE_SEC + 60)
    act = _open(session, "lost", at=t0, corr="c-lost")
    ev = act_closed_event(act, produced=False, source_service="orion-thought", source_component="t",
                          at=t0 + timedelta(seconds=30), correlation_id="c-lost")
    import orion.grammar.ledger as ledger

    ledger.configure_act_ledger(False)
    try:
        apply_grammar_trace_batch(session, [ev])
    finally:
        ledger.configure_act_ledger(True)
    assert _row(session, act.act_id)["closed_at"] is None
    summary = resolve_due_acts(session)
    assert summary.closed_output_no_output == 1
    row = _row(session, act.act_id)
    assert row["outcome_source"] == "producer" and row["outcome_value"] == 0.0
