"""Reading's real cost against real Postgres (attention patch 3a).

The read act (``hub.world_pulse_read``) carries the seed's request_id as its correlation_id; the
read's GPU pool leases carry its durable turns' ids. Until patch 3a the fill joined on the act's own
id and every read closed with ``gpu_slot_s_by_lane`` NULL (live 2026-11-10: 0 of 1,117 reads in 30
days). The shape below is a real read from 2026-11-09 (seed reading:9204432b..., ids and lease
durations copied from gpu_pool_events / reading_durable_turn), with a subset of its leases.

Opt-in like the other DB tests: ORION_SQL_WRITER_TEST_DATABASE_URI names a scratch database; every
table lives in a throwaway schema.
"""
from __future__ import annotations

import importlib.util
import io
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models.cockpit_turn_sighting import CockpitTurnSightingSQL
from app.models.grammar_trace import GrammarActSQL, GrammarEventSQL, GrammarTraceSQL
from orion.durable_runs.turn_identity import durable_turn_correlation_id
from orion.grammar.act_ledger import (
    LATE_COST_FINAL_AFTER_SEC,
    _gpu_slot_seconds,
    refill_late_costs,
)
from orion.grammar.acts import act_closed_event, act_opened_event, build_act
from orion.grammar.ledger import apply_grammar_trace_batch

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[3]

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
CREATE TABLE gpu_pool_events (event_id VARCHAR PRIMARY KEY, generated_at TIMESTAMPTZ NOT NULL,
    event VARCHAR NOT NULL, lease_id VARCHAR, holder VARCHAR, role VARCHAR, cards JSON,
    turn_correlation_id VARCHAR, held_ms DOUBLE PRECISION, detail JSON);
CREATE TABLE world_pulse_read_seed (seed_id TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'pending',
    trace_id TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), completed_at TIMESTAMPTZ,
    stage2_status TEXT NOT NULL DEFAULT 'pending', stage2_completed_at TIMESTAMPTZ,
    stage2_trace_id TEXT, request_id UUID UNIQUE);
CREATE TABLE reading_durable_turn (seed_id TEXT NOT NULL, stage INTEGER NOT NULL, attempt INTEGER NOT NULL,
    run_id TEXT NOT NULL UNIQUE, request_json JSONB NOT NULL, consumed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(), PRIMARY KEY (seed_id, stage, attempt));
"""

# The live read (2026-11-09 23:14 .. 11-10 00:03).
SEED = "reading:9204432b-fefc-5552-aaaa-3af765224f5d"
REQUEST_ID = "9204432b-fefc-5552-aaaa-3af765224f5d"
S1_RUN, S1_CORR, S1_HOLD = ("reading-95c9b912-01a7-5d79-b071-0a4c8fadcab5", "86e84801-7ff4-4cb8-81df-35ab8fc95f25",
                            "fbaf0d796f1247e1a57705763db91f7f")
S2_RUN, S2_CORR, S2_HOLD = ("reading-082a44fd-3a2b-55a9-9c3c-822d9590bbd5", "4596b4da-53b6-462a-8193-b96607dae642",
                            "98fe968a322c4204b38195eeef61f505")
S1_TRACE = "fd92e090-eb5e-5c35-bcdb-67dae5175579"   # the seed's trace_id
S2_TRACE = "a92fef07-c5fa-5e0a-ab9a-a10223fdaf8f"   # the seed's stage2_trace_id


@pytest.fixture
def session():
    import os

    uri = os.environ["ORION_SQL_WRITER_TEST_DATABASE_URI"]
    schema = f"act_rc_{uuid.uuid4().hex[:10]}"
    admin = create_engine(uri)
    with admin.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_engine(uri, connect_args={"options": f"-csearch_path={schema}"})
    with engine.begin() as conn:
        for stmt in L10_DDL.split(";"):
            if stmt.strip():
                conn.execute(text(stmt))
    Base.metadata.create_all(bind=engine, tables=[GrammarActSQL.__table__, GrammarEventSQL.__table__,
                                                  GrammarTraceSQL.__table__, CockpitTurnSightingSQL.__table__])
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


def _ev(sess, *, at, event, lease, corr, held_ms=None, role=None, holder="http:anthropic", detail=None):
    sess.execute(text(
        "INSERT INTO gpu_pool_events (event_id, generated_at, event, lease_id, holder, role, cards, "
        "turn_correlation_id, held_ms, detail) VALUES (:e, :at, :ev, :l, :h, :r, CAST(:cards AS json), :c, :ms, "
        "CAST(:d AS json))"),
        {"e": uuid.uuid4().hex, "at": at, "ev": event, "l": lease, "h": holder, "r": role,
         "cards": json.dumps(["gpu1"]), "c": corr, "ms": held_ms, "d": json.dumps(detail or {})})


def _hold(sess, *, run, corr, lease, at, generation=1, released_at=None, role="agent-gpu2"):
    """A durable run's pool hold: granted to durable-runs:<run>, carrying the run's correlation id."""
    _ev(sess, at=at, event="granted", lease=lease, corr=corr, role=role, holder=f"durable-runs:{run}",
        detail={"grant": {"lease_id": lease, "generation": generation, "role": role}})
    if released_at is not None:
        _ev(sess, at=released_at, event="released", lease=lease, corr=corr, role=role, holder=f"durable-runs:{run}",
            held_ms=(released_at - at).total_seconds() * 1000.0)


def _turn(sess, *, stage, attempt, run, corr, at):
    sess.execute(text(
        "INSERT INTO reading_durable_turn (seed_id, stage, attempt, run_id, request_json, created_at) "
        "VALUES (:s, :st, :a, :r, CAST(:j AS jsonb), :at)"),
        {"s": SEED, "st": stage, "a": attempt, "r": run, "at": at,
         "j": json.dumps({"run_id": run, "workflow": "reading.turn", "correlation_id": corr})})


T0 = datetime(2026, 11, 9, 23, 14, 45, tzinfo=timezone.utc)


def _stage1(sess):
    """Stage 1 as it ran live: one hold (not charged) and its turn's calls (charged)."""
    sess.execute(text("INSERT INTO world_pulse_read_seed (seed_id, status, created_at, request_id) "
                      "VALUES (:s, 'claimed', :at, CAST(:r AS uuid))"),
                 {"s": SEED, "at": T0 - timedelta(minutes=5), "r": REQUEST_ID})
    _turn(sess, stage=1, attempt=0, run=S1_RUN, corr=S1_CORR, at=T0 - timedelta(seconds=1))
    _hold(sess, run=S1_RUN, corr=S1_CORR, lease=S1_HOLD, at=T0, released_at=T0 + timedelta(minutes=24))
    _ev(sess, at=T0 + timedelta(seconds=2), event="released", lease="91180cac", corr=S1_TRACE, role="fast",
        held_ms=1192.454)
    _ev(sess, at=T0 + timedelta(seconds=5), event="released", lease="1c612d11", corr=S1_TRACE, role="metacog",
        held_ms=3134.564)
    _ev(sess, at=T0 + timedelta(minutes=13), event="released", lease="b0e5f29a", corr=S1_TRACE, role="agent-gpu2",
        held_ms=752218.568)
    # An expired-then-released lease: counted once, at its longest hold.
    _ev(sess, at=T0 + timedelta(minutes=23), event="expired", lease="d3456741", corr=S1_TRACE, role="agent-gpu2",
        held_ms=100000.0)
    _ev(sess, at=T0 + timedelta(minutes=23, seconds=37), event="released", lease="d3456741", corr=S1_TRACE,
        role="agent-gpu2", held_ms=237264.12)
    # Another read's call in the same minutes: never charged to this act.
    _ev(sess, at=T0 + timedelta(minutes=10), event="released", lease="other", corr=str(uuid.uuid4()),
        role="agent-gpu2", held_ms=999000.0)


S1_LANES = {"fast": 1.192, "metacog": 3.135, "agent-gpu2": 989.483}
S2_START = T0 + timedelta(minutes=26, seconds=7)


def _stage1_done(sess):
    sess.execute(text("UPDATE world_pulse_read_seed SET status = 'done', trace_id = :t, completed_at = :at "
                      "WHERE seed_id = :s"), {"t": S1_TRACE, "at": T0 + timedelta(minutes=25, seconds=27), "s": SEED})


def _stage2(sess):
    _turn(sess, stage=2, attempt=0, run=S2_RUN, corr=S2_CORR, at=S2_START - timedelta(seconds=1))
    _hold(sess, run=S2_RUN, corr=S2_CORR, lease=S2_HOLD, at=S2_START, role="agent",
          released_at=S2_START + timedelta(minutes=22))
    _ev(sess, at=S2_START + timedelta(minutes=4), event="released", lease="74bb29c5", corr=S2_TRACE, role="agent",
        held_ms=181697.888)
    _ev(sess, at=S2_START + timedelta(minutes=20), event="released", lease="5a503f2f", corr=S2_TRACE, role="agent",
        held_ms=78892.503)


def _stage2_done(sess):
    sess.execute(text("UPDATE world_pulse_read_seed SET stage2_status = 'done', stage2_trace_id = :t, "
                      "stage2_completed_at = :at WHERE seed_id = :s"),
                 {"t": S2_TRACE, "at": S2_START + timedelta(minutes=22, seconds=21), "s": SEED})


def _read_act():
    return build_act(actor="hub.world_pulse_read", verb="look", key=SEED, signal_id="artifact_produced",
                     horizon_sec=3600.0, basis="p90", object_ref="https://example.org/a",
                     test_before=f"read_seed:{SEED}")


def _open_close(sess, act, *, close_at):
    apply_grammar_trace_batch(sess, [act_opened_event(
        act, source_service="orion-hub", source_component="world_pulse_read.act", correlation_id=REQUEST_ID,
        at=T0)])
    apply_grammar_trace_batch(sess, [act_closed_event(
        act, produced=True, source_service="orion-hub", source_component="world_pulse_read.act",
        correlation_id=REQUEST_ID, at=close_at)])


def _cost(sess, act_id):
    return sess.execute(text("SELECT cost_json FROM grammar_acts WHERE act_id = :a"), {"a": act_id}).scalar()


def test_live_trace_ids_are_the_durable_turn_rule():
    """The seed's recorded trace ids ARE durable-runs' turn ids under each stage's hold (live data)."""
    assert durable_turn_correlation_id(S1_RUN, S1_CORR, S1_HOLD, 1) == S1_TRACE
    assert durable_turn_correlation_id(S2_RUN, S2_CORR, S2_HOLD, "1") == S2_TRACE


def test_old_join_on_the_acts_own_id_charges_a_read_nothing(session):
    """Regression: the pre-3a fill looked up leases by the act's correlation_id (the request_id)."""
    _stage1(session)
    window = (T0 - timedelta(hours=1), T0 + timedelta(hours=2))
    assert _gpu_slot_seconds(session, [REQUEST_ID], *window) is None
    assert _gpu_slot_seconds(session, [S1_TRACE], *window) == pytest.approx(S1_LANES)


def test_read_cost_lifecycle_close_then_stage2_then_late_fill_once(session):
    act = _read_act()
    _stage1(session)
    _stage1_done(session)
    close_at = T0 + timedelta(minutes=25, seconds=27)
    _open_close(session, act, close_at=close_at)

    # 1. At close: wall seconds, and NO GPU number (a partial stage-1 cost would read as the whole read).
    cost = _cost(session, act.act_id)
    assert cost["seconds"] == pytest.approx(25 * 60 + 27)
    assert cost["gpu_slot_s_by_lane"] is None

    # 2. Stage 2 is running: the late fill leaves the act alone.
    _stage2(session)
    s = refill_late_costs(session, now=S2_START + timedelta(minutes=5))
    assert s.examined == 0 and _cost(session, act.act_id)["gpu_slot_s_by_lane"] is None

    # 3. Stage 2 ended: both stages' calls, never the holds, never another read's call.
    _stage2_done(session)
    now = S2_START + timedelta(minutes=30)
    s = refill_late_costs(session, now=now)
    assert (s.examined, s.filled, s.no_lease, s.failed) == (1, 1, 0, 0)
    cost = _cost(session, act.act_id)
    want = dict(S1_LANES, agent=260.59)
    assert cost["gpu_slot_s_by_lane"] == pytest.approx(want, abs=0.01)
    assert cost["seconds"] == pytest.approx(25 * 60 + 27)  # the close's own fields are kept
    assert cost["basis"]["gpu"].startswith("late fill") and "stage 2 done" in cost["basis"]["gpu"]
    assert s.slot_s == pytest.approx(sum(want.values()), abs=0.01)

    # 4. Filled once: a later lease and a second pass change nothing.
    _ev(session, at=now + timedelta(minutes=1), event="released", lease="late", corr=S2_TRACE, role="agent",
        held_ms=50000.0)
    assert refill_late_costs(session, now=now + timedelta(minutes=2)).examined == 0
    assert _cost(session, act.act_id)["gpu_slot_s_by_lane"] == pytest.approx(want, abs=0.01)


def test_a_failed_attempts_turn_is_charged_though_the_seed_never_recorded_it(session):
    """Attempt 0 of stage 1 failed: its turn id is in no seed column, only re-derivable from its hold."""
    act = _read_act()
    _stage1(session)
    fail_run, fail_corr, fail_hold = "reading-0000aaaa-0000-5000-8000-000000000001", str(uuid.uuid4()), "holdfail"
    session.execute(text("UPDATE reading_durable_turn SET attempt = 1 WHERE run_id = :r"), {"r": S1_RUN})
    _turn(session, stage=1, attempt=0, run=fail_run, corr=fail_corr, at=T0 - timedelta(minutes=3))
    _hold(session, run=fail_run, corr=fail_corr, lease=fail_hold, at=T0 - timedelta(minutes=3), generation=2,
          released_at=T0 - timedelta(minutes=1))
    fail_turn = durable_turn_correlation_id(fail_run, fail_corr, fail_hold, 2)
    _ev(session, at=T0 - timedelta(minutes=2), event="released", lease="failcall", corr=fail_turn,
        role="agent-gpu2", held_ms=60000.0)
    _stage1_done(session)
    _stage2(session)
    _stage2_done(session)
    _open_close(session, act, close_at=T0 + timedelta(minutes=25, seconds=27))
    refill_late_costs(session, now=S2_START + timedelta(hours=1))
    lanes = _cost(session, act.act_id)["gpu_slot_s_by_lane"]
    assert lanes["agent-gpu2"] == pytest.approx(S1_LANES["agent-gpu2"] + 60.0, abs=0.01)


def test_stage1_failed_for_good_is_filled_with_stage1_only(session):
    act = _read_act()
    _stage1(session)
    session.execute(text("UPDATE world_pulse_read_seed SET status = 'failed', completed_at = :at WHERE seed_id = :s"),
                    {"at": T0 + timedelta(minutes=25), "s": SEED})
    _open_close(session, act, close_at=T0 + timedelta(minutes=25))
    s = refill_late_costs(session, now=T0 + timedelta(minutes=30))
    cost = _cost(session, act.act_id)
    assert s.filled == 1 and cost["gpu_slot_s_by_lane"] == pytest.approx(S1_LANES, abs=0.01)
    assert "stage 1 failed" in cost["basis"]["gpu"]


def test_a_read_that_never_reports_its_end_is_filled_after_the_wait(session):
    act = _read_act()
    _stage1(session)
    _stage1_done(session)  # stage 2 never claimed or finished
    close_at = T0 + timedelta(minutes=25, seconds=27)
    _open_close(session, act, close_at=close_at)
    assert refill_late_costs(session, now=close_at + timedelta(hours=2)).examined == 0
    late = close_at + timedelta(seconds=LATE_COST_FINAL_AFTER_SEC + 60)
    assert refill_late_costs(session, now=late).filled == 1
    assert "no end recorded" in _cost(session, act.act_id)["basis"]["gpu"]


def test_a_read_with_no_lease_is_marked_once_and_not_reselected(session):
    act = _read_act()
    session.execute(text("INSERT INTO world_pulse_read_seed (seed_id, status, created_at, completed_at, request_id) "
                         "VALUES (:s, 'skipped', :at, :at, CAST(:r AS uuid))"),
                    {"s": SEED, "at": T0, "r": REQUEST_ID})
    _open_close(session, act, close_at=T0 + timedelta(seconds=30))
    s = refill_late_costs(session, now=T0 + timedelta(minutes=5))
    assert (s.examined, s.filled, s.no_lease) == (1, 0, 1)
    cost = _cost(session, act.act_id)
    assert cost["gpu_slot_s_by_lane"] is None and "no lease found" in cost["basis"]["gpu"]
    assert refill_late_costs(session, now=T0 + timedelta(minutes=6)).examined == 0


def test_other_actors_are_still_filled_at_close_on_their_own_id(session):
    act = build_act(actor="thought.painting", verb="make", key="p-1", signal_id="artifact_produced",
                    horizon_sec=5400.0, basis="deadline")
    _ev(session, at=T0 + timedelta(seconds=30), event="released", lease="P1", corr="corr-paint", role="diffusion",
        held_ms=4000.0)
    apply_grammar_trace_batch(session, [act_opened_event(
        act, source_service="orion-thought", source_component="t", correlation_id="corr-paint", at=T0)])
    apply_grammar_trace_batch(session, [act_closed_event(
        act, produced=True, source_service="orion-thought", source_component="t", correlation_id="corr-paint",
        at=T0 + timedelta(seconds=60))])
    assert _cost(session, act.act_id)["gpu_slot_s_by_lane"] == {"diffusion": pytest.approx(4.0)}


def _backfill_module():
    spec = importlib.util.spec_from_file_location("backfill_reading_act_cost",
                                                  REPO / "scripts" / "backfill_reading_act_cost.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_backfill_dry_run_writes_nothing_then_apply_then_restore(session, tmp_path):
    bf = _backfill_module()
    act = _read_act()
    _stage1(session)
    _stage1_done(session)
    _stage2(session)
    _stage2_done(session)
    _open_close(session, act, close_at=T0 + timedelta(minutes=25, seconds=27))
    session.commit()
    before = _cost(session, act.act_id)
    now = T0 + timedelta(days=3)  # outside the live 48 h window: only the backfill reaches it
    assert refill_late_costs(session, now=now).examined == 0

    dry = bf.run(session, days=30, apply=False, job_dir=tmp_path, now=now, out=io.StringIO())
    assert (dry["targets"], dry["gain_cost"], dry["written"], dry["errors"]) == (1, 1, 0, 0)
    assert dry["slot_hours"] == pytest.approx((sum(S1_LANES.values()) + 260.59) / 3600, abs=0.05)
    assert _cost(session, act.act_id) == before
    for name in ("snapshot.jsonl", "progress.log", "report.md", "before_after.csv"):
        assert (tmp_path / name).exists(), name
    assert "100.0%" in (tmp_path / "progress.log").read_text()

    applied = bf.run(session, days=30, apply=True, job_dir=tmp_path / "apply", now=now, commit=session.commit,
                     out=io.StringIO())
    assert applied["written"] == 1
    assert _cost(session, act.act_id)["gpu_slot_s_by_lane"]["agent"] == pytest.approx(260.59, abs=0.01)
    again = bf.run(session, days=30, apply=True, job_dir=tmp_path / "again", now=now, commit=session.commit,
                   out=io.StringIO())
    assert again["targets"] == 0

    assert bf.restore(session, tmp_path / "apply", commit=session.commit) == 1
    assert _cost(session, act.act_id) == before
