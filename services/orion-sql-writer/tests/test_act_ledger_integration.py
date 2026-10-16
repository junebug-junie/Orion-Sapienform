"""Act ledger against real Postgres (rung 2): persist, close by act_id, grade into L10, resolve.

Opt-in like the other DB tests: set ORION_SQL_WRITER_TEST_DATABASE_URI to a scratch
database. Every table is created inside a fresh throwaway schema (search_path), so
even a misconfigured URI never touches the real substrate_action_* tables.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models.grammar_trace import (  # noqa: F401 -- registers the grammar tables on Base
    GrammarActSQL,
    GrammarEventSQL,
    GrammarTraceSQL,
)
from orion.grammar.act_ledger import EXPIRY_GRACE_SEC, resolve_due_acts
from orion.grammar.acts import act_closed_event, act_opened_event, build_act
from orion.grammar.ledger import apply_grammar_trace_batch

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
        tables=[GrammarActSQL.__table__, GrammarEventSQL.__table__, GrammarTraceSQL.__table__],
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


def _make_act(key="run-1"):
    return build_act(actor="thought.painting", verb="make", key=key, signal_id="artifact_produced",
                     horizon_sec=5400.0, basis="deadline")


def _say_act(key="corr-1", horizon=1800.0):
    return build_act(actor="hub.chat_reply", verb="say", key=key, signal_id="juniper_replied",
                     horizon_sec=horizon, basis="median gap")


def _row(sess, act_id):
    return sess.execute(text("SELECT * FROM grammar_acts WHERE act_id = :a"), {"a": act_id}).mappings().one()


def _posterior(sess, kind, actor, signal):
    return sess.execute(text(
        "SELECT posterior_mean, posterior_n FROM substrate_action_effect_posterior "
        "WHERE dispatch_kind = :k AND target_id = :t AND signal_id = :s"), {"k": kind, "t": actor, "s": signal}).first()


def test_open_then_producer_close_grades_into_l10(session):
    t0 = datetime.now(timezone.utc) - timedelta(minutes=5)
    act = _make_act()
    opened = act_opened_event(act, source_service="orion-thought", source_component="t", at=t0)
    closed = act_closed_event(act, produced=True, source_service="orion-thought", source_component="t",
                              at=t0 + timedelta(seconds=90))
    assert apply_grammar_trace_batch(session, [opened]) == 1
    row = _row(session, act.act_id)
    assert row["predicted"] == pytest.approx(0.5) and row["cold_start"] is True and row["predicted_n"] == 0
    assert row["horizon_at"] == t0 + timedelta(seconds=5400)

    assert apply_grammar_trace_batch(session, [closed]) == 1
    row = _row(session, act.act_id)
    assert row["outcome_value"] == 1.0 and row["outcome_source"] == "producer"
    assert row["error"] == pytest.approx(0.5)
    assert row["surprise_nats"] > 0 and row["outcome_row_id"] is not None
    out = session.execute(text("SELECT * FROM substrate_action_outcomes")).mappings().one()
    assert out["dispatch_kind"] == "act:make" and out["target_id"] == "thought.painting"
    assert out["signal_id"] == "artifact_produced" and out["claim_upheld"] is True
    assert out["predicted_delta"] == pytest.approx(0.5) and out["observed_delta"] == 1.0
    mean, n = _posterior(session, "act:make", "thought.painting", "artifact_produced")
    assert n == 1 and mean > 0.5

    # Second act of the same kind is stamped from the moved belief, not the cold prior.
    act2 = _make_act("run-2")
    apply_grammar_trace_batch(session, [act_opened_event(act2, source_service="orion-thought", source_component="t")])
    row2 = _row(session, act2.act_id)
    assert row2["predicted"] == pytest.approx(mean) and row2["predicted_n"] == 1 and row2["cold_start"] is False


def test_second_close_and_replay_do_not_double_count(session):
    act = _make_act()
    apply_grammar_trace_batch(session, [act_opened_event(act, source_service="orion-thought", source_component="t")])
    first = act_closed_event(act, produced=True, source_service="orion-thought", source_component="t")
    late = act_closed_event(act, produced=False, source_service="orion-thought", source_component="t")
    apply_grammar_trace_batch(session, [first])
    apply_grammar_trace_batch(session, [first])  # replayed event_id: deduped before the act ledger
    apply_grammar_trace_batch(session, [late])   # new event_id, same act: first close wins
    assert _row(session, act.act_id)["outcome_value"] == 1.0
    assert session.execute(text("SELECT count(*) FROM substrate_action_outcomes")).scalar() == 1
    assert _posterior(session, "act:make", "thought.painting", "artifact_produced")[1] == 1


def test_close_before_open_is_kept_and_open_does_not_rewrite_prediction(session):
    act = _make_act()
    closed = act_closed_event(act, produced=False, source_service="orion-thought", source_component="t")
    apply_grammar_trace_batch(session, [closed])
    row = _row(session, act.act_id)
    assert row["opened_at"] is None and row["outcome_value"] == 0.0 and row["predicted"] == pytest.approx(0.5)
    apply_grammar_trace_batch(session, [act_opened_event(act, source_service="orion-thought", source_component="t")])
    row = _row(session, act.act_id)
    assert row["opened_at"] is not None and row["predicted"] == pytest.approx(0.5)
    assert row["error"] == pytest.approx(-0.5)


def _chat(session, *, at, corr, prompt="hi", unsolicited=False):
    session.execute(text(
        "INSERT INTO chat_history_log (id, correlation_id, source, prompt, response, client_meta, created_at) "
        "VALUES (:id, :c, 'hub_orion', :p, 'r', CAST(:m AS jsonb), :at)"),
        {"id": uuid.uuid4().hex, "c": corr, "p": prompt,
         "m": '{"unsolicited": true}' if unsolicited else None,
         "at": at.astimezone(timezone.utc).replace(tzinfo=None)})


def test_resolver_closes_say_from_chat_history(session):
    now = datetime.now(timezone.utc)
    t0 = now - timedelta(minutes=10)
    replied, silent, too_early = _say_act("c-replied"), _say_act("c-silent", horizon=60.0), _say_act("c-open")
    def _open(act, corr, at):
        apply_grammar_trace_batch(session, [act_opened_event(
            act, source_service="orion-hub", source_component="t", correlation_id=corr, at=at)])

    _open(replied, "c-replied", t0)
    # The reply's own turn row lands just after its act opens (same correlation): must NOT count.
    _chat(session, at=t0 + timedelta(seconds=5), corr="c-replied")
    _open(silent, "c-silent", t0 + timedelta(seconds=10))
    _open(too_early, "c-open", t0 + timedelta(seconds=10))
    # An unsolicited outreach row is not Juniper speaking: must NOT count.
    _chat(session, at=t0 + timedelta(seconds=12), corr="outreach-x", prompt=None, unsolicited=True)
    summary = resolve_due_acts(session, now=now)
    # Nothing qualifying yet: the 60 s act has passed its horizon -> 0; others stay open.
    assert summary.closed_not_replied == 1 and summary.closed_replied == 0
    assert _row(session, silent.act_id)["outcome_value"] == 0.0
    assert _row(session, replied.act_id)["closed_at"] is None

    _chat(session, at=t0 + timedelta(seconds=300), corr="c-next-turn")
    summary = resolve_due_acts(session, now=now)
    assert summary.closed_replied == 2  # replied and too_early both saw the next Juniper turn
    row = _row(session, replied.act_id)
    assert row["outcome_value"] == 1.0 and row["outcome_source"] == "resolver"
    assert abs((row["closed_at"] - (t0 + timedelta(seconds=300))).total_seconds()) < 0.01
    mean, n = _posterior(session, "act:say", "hub.chat_reply", "juniper_replied")
    assert n == 3


def test_resolver_expires_producer_signal_only_after_grace(session):
    now = datetime.now(timezone.utc)
    act = build_act(actor="hub.world_pulse_read", verb="look", key="seed-1", signal_id="artifact_produced",
                    horizon_sec=60.0, basis="p90")
    opened_at = now - timedelta(seconds=60 + EXPIRY_GRACE_SEC / 2)
    apply_grammar_trace_batch(session, [act_opened_event(
        act, source_service="orion-hub", source_component="t", at=opened_at)])
    assert resolve_due_acts(session, now=now).closed_expired == 0, "inside grace: producer may still close"
    later = now + timedelta(seconds=EXPIRY_GRACE_SEC)
    assert resolve_due_acts(session, now=later).closed_expired == 1
    row = _row(session, act.act_id)
    assert row["outcome_value"] == 0.0 and row["outcome_source"] == "expired"


def test_ledger_summary_reports_coverage_and_error(session):
    from orion.grammar.act_ledger import ledger_summary

    act = _make_act()
    apply_grammar_trace_batch(session, [act_opened_event(act, source_service="orion-thought", source_component="t")])
    apply_grammar_trace_batch(session, [act_closed_event(act, produced=True, source_service="orion-thought",
                                                         source_component="t")])
    open_only = _make_act("run-open")
    apply_grammar_trace_batch(session, [act_opened_event(open_only, source_service="orion-thought",
                                                         source_component="t")])
    (row,) = ledger_summary(session)["by_act"]
    assert (row["verb"], row["actor"], row["opened"], row["closed"], row["open"]) == (
        "make", "thought.painting", 2, 1, 1)
    assert row["closed_in_horizon"] == 1 and row["with_expect"] == 2
    assert row["outcome_rate"] == 1.0 and row["mean_abs_error"] == 0.5


def test_success_reported_after_the_horizon_grades_zero(session):
    t0 = datetime.now(timezone.utc) - timedelta(hours=3)
    act = _make_act()  # 5400 s horizon
    apply_grammar_trace_batch(session, [act_opened_event(act, source_service="orion-thought",
                                                         source_component="t", at=t0)])
    late = act_closed_event(act, produced=True, source_service="orion-thought", source_component="t",
                            at=t0 + timedelta(seconds=5460))
    apply_grammar_trace_batch(session, [late])
    row = _row(session, act.act_id)
    assert row["outcome_value"] == 0.0 and row["outcome_note"].startswith("late:60s")


def test_resolver_recovers_a_persisted_close_instead_of_expiring(session, monkeypatch):
    import orion.grammar.ledger as ledger

    now = datetime.now(timezone.utc)
    t0 = now - timedelta(seconds=5400 + 2 * EXPIRY_GRACE_SEC)
    act = _make_act()
    apply_grammar_trace_batch(session, [act_opened_event(act, source_service="orion-thought",
                                                         source_component="t", at=t0)])
    # The close lands in grammar_events but its act-ledger step is lost (e.g. a lost lock fight).
    ledger.configure_act_ledger(False)
    try:
        apply_grammar_trace_batch(session, [act_closed_event(
            act, produced=True, source_service="orion-thought", source_component="t",
            at=t0 + timedelta(seconds=600))])
    finally:
        ledger.configure_act_ledger(True)
    assert _row(session, act.act_id)["closed_at"] is None
    commits = []
    summary = resolve_due_acts(session, now=now, commit=lambda: commits.append(1))
    assert summary.recovered_producer_closes == 1 and summary.closed_expired == 0
    assert commits == [1]
    row = _row(session, act.act_id)
    assert row["outcome_value"] == 1.0 and row["outcome_source"] == "producer"


def test_resolver_zeros_are_not_counted_as_in_horizon_coverage(session):
    from orion.grammar.act_ledger import ledger_summary

    now = datetime.now(timezone.utc)
    act = _say_act("c-quiet", horizon=60.0)
    apply_grammar_trace_batch(session, [act_opened_event(
        act, source_service="orion-hub", source_component="t", correlation_id="c-quiet",
        at=now - timedelta(minutes=10))])
    resolve_due_acts(session, now=now)
    (row,) = ledger_summary(session, now=now)["by_act"]
    assert row["closed"] == 1 and row["closed_in_horizon"] == 0 and row["resolved_absent"] == 1
