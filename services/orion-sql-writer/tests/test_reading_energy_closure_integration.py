"""Reading's energy, counted once (attention patch 3e), against real Postgres.

Patch 3a charged a read act its GPU slot-seconds but no energy: the hourly attribution
(``attribute_host_hour``) counted a read's leases as "leased without an act", so also charging the
read would count those watt-hours twice. 3e moves both sides together. The check here sums the
energy BOTH ways over one host-hour:

    sum of every act's wh_by_host  +  the hour's remainder (sponsored + unattributed)
        ==  the host's metered chassis energy above rest (above_rest_wh)

before 3e (read uncharged, read leases unmetered), after 3e (read charged, read leases metered),
and the double count 3e prevents (read charged, read leases still unmetered).

The shape is the live read of 2026-11-09 23:14 (seed reading:9204432b..., its real run ids, holds and
trace ids; durations from gpu_pool_events) and circe's real readings that minute (chassis 1,122 W,
CPU 188 W over a 100 W idle, cards 38 / 152 / 155 / 74 W; rest 541 W from rest_body.yaml), squeezed
into one hour with a failed attempt, an ordinary act and a turn with no act on the same cards.

Opt-in like the other act ledger DB tests: ORION_SQL_WRITER_TEST_DATABASE_URI names a scratch
database; every table lives in a throwaway schema.
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
from app.models.orion_energy_attribution_hourly import OrionEnergyAttributionHourlySQL
from orion.durable_runs.turn_identity import durable_turn_correlation_id
from orion.grammar import act_cost_ids as ids
from orion.grammar import act_energy_fill as fill
from orion.grammar import act_ledger as ledger
from orion.grammar.acts import act_closed_event, act_opened_event, build_act
from orion.grammar.ledger import apply_grammar_trace_batch
from orion.physiology.rest_body import parse_rest_body

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[3]

DDL = """
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
    turn_correlation_id VARCHAR, held_ms DOUBLE PRECISION, detail JSON, work_class VARCHAR);
CREATE TABLE world_pulse_read_seed (seed_id TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'pending',
    trace_id TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), completed_at TIMESTAMPTZ,
    stage2_status TEXT NOT NULL DEFAULT 'pending', stage2_completed_at TIMESTAMPTZ,
    stage2_trace_id TEXT, request_id UUID UNIQUE);
CREATE TABLE reading_durable_turn (seed_id TEXT NOT NULL, stage INTEGER NOT NULL, attempt INTEGER NOT NULL,
    run_id TEXT NOT NULL UNIQUE, request_json JSONB NOT NULL, consumed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(), PRIMARY KEY (seed_id, stage, attempt));
CREATE TABLE orion_biometrics_summary (id VARCHAR PRIMARY KEY, "timestamp" VARCHAR NOT NULL, node VARCHAR,
    measurements JSONB);
CREATE TABLE energy_cost_accrued (interval_start TIMESTAMPTZ NOT NULL, marginal_usd_per_kwh DOUBLE PRECISION,
    tariff_version TEXT, computed_at TIMESTAMPTZ NOT NULL DEFAULT now())
"""

# The live read (2026-11-09 23:14 .. 11-10 00:03): real ids.
SEED = "reading:9204432b-fefc-5552-aaaa-3af765224f5d"
REQUEST_ID = "9204432b-fefc-5552-aaaa-3af765224f5d"
S1_RUN, S1_CORR, S1_HOLD = ("reading-95c9b912-01a7-5d79-b071-0a4c8fadcab5", "86e84801-7ff4-4cb8-81df-35ab8fc95f25",
                            "fbaf0d796f1247e1a57705763db91f7f")
S2_RUN, S2_CORR, S2_HOLD = ("reading-082a44fd-3a2b-55a9-9c3c-822d9590bbd5", "4596b4da-53b6-462a-8193-b96607dae642",
                            "98fe968a322c4204b38195eeef61f505")
S1_TRACE = "fd92e090-eb5e-5c35-bcdb-67dae5175579"
S2_TRACE = "a92fef07-c5fa-5e0a-ab9a-a10223fdaf8f"
F_RUN, F_CORR, F_HOLD = "reading-0000aaaa-0000-5000-8000-000000000001", "0f0f0f0f-0000-4000-8000-000000000001", "holdfail"
F_TURN = durable_turn_correlation_id(F_RUN, F_CORR, F_HOLD, 2)

H = datetime(2026, 11, 9, 23, 0, tzinfo=timezone.utc)
STAGE1_END = H + timedelta(minutes=30)
STAGE2_END = H + timedelta(minutes=56)
CARDS = {"gpu0": ("circe", 0), "gpu1": ("circe", 1), "gpu2": ("circe", 2), "gpu3": ("circe", 3)}
# circe 2026-11-09 23:20:31, 23:21:02, 23:21:33 (orion_biometrics_summary), cycled every 30 s.
LIVE_ROWS = [
    {"chassis_watts": 1122.0, "cpu_watts_total": 188.14, "cpu_idle_watts": 100.0,
     "gpu0_watts": 37.77, "gpu1_watts": 151.89, "gpu2_watts": 154.62, "gpu3_watts": 73.62},
    {"chassis_watts": 1122.0, "cpu_watts_total": 182.50, "cpu_idle_watts": 100.0,
     "gpu0_watts": 37.86, "gpu1_watts": 154.94, "gpu2_watts": 151.75, "gpu3_watts": 70.70},
    {"chassis_watts": 1131.0, "cpu_watts_total": 163.58, "cpu_idle_watts": 100.0,
     "gpu0_watts": 37.86, "gpu1_watts": 158.35, "gpu2_watts": 146.85, "gpu3_watts": 70.70},
]
IDLE_ROW = {"gpu0_watts": 37.5, "gpu1_watts": 34.0, "gpu2_watts": 33.0, "gpu3_watts": 36.0}
NOW = H + timedelta(hours=2)


@pytest.fixture
def session(monkeypatch):
    import os

    uri = os.environ["ORION_SQL_WRITER_TEST_DATABASE_URI"]
    schema = f"read_wh_{uuid.uuid4().hex[:10]}"
    admin = create_engine(uri)
    with admin.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_engine(uri, connect_args={"options": f"-csearch_path={schema}"})
    with engine.begin() as conn:
        for stmt in DDL.split(";"):
            if stmt.strip():
                conn.execute(text(stmt))
    Base.metadata.create_all(bind=engine, tables=[GrammarActSQL.__table__, GrammarEventSQL.__table__,
                                                  GrammarTraceSQL.__table__, CockpitTurnSightingSQL.__table__,
                                                  OrionEnergyAttributionHourlySQL.__table__])
    rest = parse_rest_body({"version": 1, "hosts": {"circe": {"rest_w": 541, "approved_rest_w": 541}}})
    fill._FLOOR_CACHE.clear()
    monkeypatch.setattr(fill, "rest_body_config", lambda: (rest, None))
    monkeypatch.setattr(fill, "_card_locations", lambda: (dict(CARDS), None))
    monkeypatch.setattr(ledger, "_LATE_COST_FAILURES", {})
    ledger.configure_act_energy_cost(True)
    sess = sessionmaker(bind=engine)()
    try:
        yield sess
    finally:
        ledger.configure_act_energy_cost(True)
        sess.rollback()
        sess.close()
        engine.dispose()
        with admin.begin() as conn:
            conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        admin.dispose()


def _summary(sess, at, m):
    sess.execute(text("INSERT INTO orion_biometrics_summary VALUES (:i, :t, 'circe', CAST(:m AS jsonb))"),
                 {"i": uuid.uuid4().hex, "t": at.strftime("%Y-%m-%d %H:%M:%S.%f+00"), "m": json.dumps(m)})


def _ev(sess, *, at, event, lease, corr, card, held_ms=None, role="agent", holder="http:anthropic", detail=None):
    sess.execute(text(
        "INSERT INTO gpu_pool_events (event_id, generated_at, event, lease_id, holder, role, cards, "
        "turn_correlation_id, held_ms, detail) VALUES (:e, :at, :ev, :l, :h, :r, CAST(:cards AS json), :c, :ms, "
        "CAST(:d AS json))"),
        {"e": uuid.uuid4().hex, "at": at, "ev": event, "l": lease, "h": holder, "r": role,
         "cards": json.dumps([card]), "c": corr, "ms": held_ms, "d": json.dumps(detail or {})})


def _lease(sess, lease, card, start, end, corr, role="agent"):
    _ev(sess, at=start, event="granted", lease=lease, corr=corr, card=card, role=role)
    _ev(sess, at=end, event="released", lease=lease, corr=corr, card=card, role=role,
        held_ms=(end - start).total_seconds() * 1000.0)


def _hold(sess, *, run, corr, lease, card, start, end, generation=1):
    """A durable run's pool hold: carries the RUN's correlation id, never charged to the read."""
    for at, ev in ((start, "granted"), (end, "released")):
        _ev(sess, at=at, event=ev, lease=lease, corr=corr, card=card, holder=f"durable-runs:{run}",
            held_ms=None if ev == "granted" else (end - start).total_seconds() * 1000.0,
            detail={"grant": {"lease_id": lease, "generation": generation, "cards": [card]}} if ev == "granted" else {})


def _turn(sess, *, stage, attempt, run, corr, at):
    sess.execute(text(
        "INSERT INTO reading_durable_turn (seed_id, stage, attempt, run_id, request_json, created_at) "
        "VALUES (:s, :st, :a, :r, CAST(:j AS jsonb), :at)"),
        {"s": SEED, "st": stage, "a": attempt, "r": run, "at": at,
         "j": json.dumps({"run_id": run, "workflow": "reading.turn", "correlation_id": corr})})


def _readings(sess):
    day0 = H.replace(hour=0)
    for i in range(120):  # the day before: each card's idle floor
        _summary(sess, day0 - timedelta(hours=3) + timedelta(seconds=30 * i), IDLE_ROW)
    for i in range(-4, 125):
        _summary(sess, H + timedelta(seconds=30 * i), LIVE_ROWS[i % len(LIVE_ROWS)])
    sess.execute(text("INSERT INTO energy_cost_accrued VALUES (:at, 0.1245, 'rmp-2026', :at)"), {"at": H})


def _read_world(sess, *, stage2=True):
    """The read's seed, turns, holds and calls, plus a painting act's and a no-act turn's leases."""
    sess.execute(text("INSERT INTO world_pulse_read_seed (seed_id, status, trace_id, created_at, completed_at, "
                      "request_id) VALUES (:s, 'done', :t, :c, :d, CAST(:r AS uuid))"),
                 {"s": SEED, "t": S1_TRACE, "c": H, "d": STAGE1_END, "r": REQUEST_ID})
    # Stage 1, attempt 0 failed: its turn is in no seed column, only re-derivable from its hold.
    _turn(sess, stage=1, attempt=0, run=F_RUN, corr=F_CORR, at=H + timedelta(seconds=50))
    _hold(sess, run=F_RUN, corr=F_CORR, lease=F_HOLD, card="gpu2", start=H + timedelta(minutes=1),
          end=H + timedelta(minutes=4), generation=2)
    _lease(sess, "f-call", "gpu2", H + timedelta(minutes=1, seconds=30), H + timedelta(minutes=3, seconds=30),
           F_TURN, "agent-gpu2")
    # Stage 1, attempt 1: the live hold and its calls.
    _turn(sess, stage=1, attempt=1, run=S1_RUN, corr=S1_CORR, at=H + timedelta(minutes=5))
    _hold(sess, run=S1_RUN, corr=S1_CORR, lease=S1_HOLD, card="gpu2", start=H + timedelta(minutes=5),
          end=STAGE1_END)
    _lease(sess, "91180cac", "gpu3", H + timedelta(minutes=5, seconds=10), H + timedelta(minutes=5, seconds=11.19),
           S1_TRACE, "fast")
    _lease(sess, "b0e5f29a", "gpu2", H + timedelta(minutes=6), H + timedelta(minutes=6, seconds=752.2),
           S1_TRACE, "agent-gpu2")
    _lease(sess, "d3456741", "gpu2", H + timedelta(minutes=20), H + timedelta(minutes=20, seconds=237.3),
           S1_TRACE, "agent-gpu2")
    # A turn no act is charged for, on a card the read also uses.
    _lease(sess, "noact", "gpu3", H + timedelta(minutes=10), H + timedelta(minutes=20), "corr-noact", "metacog")
    if stage2:
        _turn(sess, stage=2, attempt=0, run=S2_RUN, corr=S2_CORR, at=H + timedelta(minutes=31))
        _hold(sess, run=S2_RUN, corr=S2_CORR, lease=S2_HOLD, card="gpu1", start=H + timedelta(minutes=31),
              end=H + timedelta(minutes=55))
        _lease(sess, "74bb29c5", "gpu1", H + timedelta(minutes=32), H + timedelta(minutes=32, seconds=181.7),
               S2_TRACE)
        _lease(sess, "5a503f2f", "gpu1", H + timedelta(minutes=40), H + timedelta(minutes=40, seconds=78.9),
               S2_TRACE)
        sess.execute(text("UPDATE world_pulse_read_seed SET stage2_status = 'done', stage2_trace_id = :t, "
                          "stage2_completed_at = :at WHERE seed_id = :s"),
                     {"t": S2_TRACE, "at": STAGE2_END, "s": SEED})
    # An ordinary act's lease, sharing gpu1 with stage 2's hold.
    _lease(sess, "paint", "gpu1", H + timedelta(minutes=36), H + timedelta(minutes=38), "corr-paint", "diffusion")


def _read_act():
    return build_act(actor="hub.world_pulse_read", verb="look", key=SEED, signal_id="artifact_produced",
                     horizon_sec=3600.0, basis="p90", object_ref="https://example.org/a",
                     test_before=f"read_seed:{SEED}")


def _paint_act():
    return build_act(actor="thought.painting", verb="make", key="p-1", signal_id="artifact_produced",
                     horizon_sec=5400.0, basis="deadline")


def _open_close(sess, act, *, corr, source, opened, closed):
    apply_grammar_trace_batch(sess, [act_opened_event(act, source_service=source, source_component="t",
                                                      correlation_id=corr, at=opened)])
    apply_grammar_trace_batch(sess, [act_closed_event(act, produced=True, source_service=source,
                                                      source_component="t", correlation_id=corr, at=closed)])


def _world(sess, *, stage2=True):
    _readings(sess)
    _read_world(sess, stage2=stage2)
    _open_close(sess, _read_act(), corr=REQUEST_ID, source="orion-hub", opened=H + timedelta(minutes=5),
                closed=STAGE1_END)
    _open_close(sess, _paint_act(), corr="corr-paint", source="orion-thought",
                opened=H + timedelta(minutes=35, seconds=30), closed=H + timedelta(minutes=38, seconds=30))


def _cost(sess, act_id):
    return sess.execute(text("SELECT cost_json FROM grammar_acts WHERE act_id = :a"), {"a": act_id}).scalar() or {}


def _acts_wh(sess) -> float:
    total = 0.0
    for (cost,) in sess.execute(text("SELECT cost_json FROM grammar_acts")).fetchall():
        total += sum(float(v) for v in ((cost or {}).get("wh_by_host") or {}).values())
    return total


def _hour(sess, *, charge_reads=True):
    got, _, _ = fill.attribute_host_hour(sess, "circe", H, H + timedelta(hours=1), now=NOW, charge_reads=charge_reads)
    assert got is not None and got.covered_s == pytest.approx(3600.0) and got.split_unknown_s == 0.0
    return got


def _both_ways(sess, got) -> float:
    """Per-act total + the hour's remainder - the metered host energy (0 when each Wh counts once)."""
    return _acts_wh(sess) + got.sponsored_wh + got.unattributed_wh - got.above_rest_wh


def test_turn_owners_find_every_read_turn_and_nothing_else(session):
    _world(session)
    corrs = [S1_TRACE, S2_TRACE, F_TURN, "corr-noact", "corr-paint", S1_CORR]
    owners = ids.reading_turn_owners(session, corrs, since=H - timedelta(hours=6), until=H + timedelta(hours=1))
    # The seed's two trace ids, the failed attempt's turn re-derived from its hold, and the hold
    # itself (the run's own correlation id: energy only); never another act's turn.
    assert owners == {S1_TRACE: REQUEST_ID, S2_TRACE: REQUEST_ID, F_TURN: REQUEST_ID, S1_CORR: REQUEST_ID}
    charged = ids.turns_charged_to_acts(session, corrs, since=H - timedelta(hours=6), until=H + timedelta(hours=1))
    assert charged == {S1_TRACE, S2_TRACE, F_TURN, S1_CORR, "corr-paint"}
    # The same turn set the read act's energy is charged over (GPU: without the holds).
    energy_ids = (set(ids.reading_cost_ids(session, REQUEST_ID, until=STAGE2_END))
                  | set(ids.reading_hold_ids(session, REQUEST_ID)))
    assert set(owners) <= energy_ids and energy_ids - set(owners) == {S2_CORR, F_CORR}  # not in ``corrs``


def test_a_read_with_no_act_is_not_metered(session):
    """A read turn counts as metered only when the read has an act to carry it."""
    _readings(session)
    _read_world(session)
    owners = ids.reading_turn_owners(session, [S1_TRACE, F_TURN], since=H - timedelta(hours=6),
                                     until=H + timedelta(hours=1))
    assert owners == {}


def test_sum_both_ways_before_3e_regression_and_after(session):
    _world(session)
    # 1. BEFORE 3e: the read act carries no energy and its leases are leased-without-act.
    ledger.configure_act_energy_cost(False)
    ledger.refill_late_costs(session, now=NOW)
    read = session.execute(text("SELECT act_id FROM grammar_acts WHERE actor = 'hub.world_pulse_read'")).scalar()
    assert _cost(session, read).get("wh_by_host") is None and _cost(session, read)["gpu_slot_s_by_lane"]
    paint_wh = _acts_wh(session)
    assert paint_wh > 0  # the painting was charged at its close
    old = _hour(session, charge_reads=False)
    assert old.metered_wh == pytest.approx(paint_wh, rel=2e-3)
    assert _both_ways(session, old) == pytest.approx(0.0, abs=0.05)

    # 2. AFTER 3e: the read act gets its Wh; the same turns become metered.
    ledger.configure_act_energy_cost(True)
    s = ledger.refill_late_costs(session, now=NOW)
    assert (s.examined, s.filled, s.failed) == (1, 1, 0)
    cost = _cost(session, read)
    read_wh = cost["wh_by_host"]["circe"]
    assert read_wh > 0 and s.wh == pytest.approx(read_wh, abs=1e-3)
    assert cost["wh_gpu_by_host"]["circe"] > 0 and cost["usd"] > 0
    assert cost["basis"]["wh"].startswith("late fill") and "whole machine above rest" in cost["basis"]["wh"]
    assert "0.1245" in cost["basis"]["usd"]  # priced at the read's end, not at the fill's clock
    new = _hour(session)
    assert new.above_rest_wh == pytest.approx(old.above_rest_wh)
    assert new.metered_wh - old.metered_wh == pytest.approx(read_wh, rel=2e-3)
    assert old.leased_unmetered_wh - new.leased_unmetered_wh == pytest.approx(read_wh, rel=2e-3)
    assert _both_ways(session, new) == pytest.approx(0.0, abs=0.05)

    # 3. The double count 3e prevents: the read charged while the hour keeps the old rule.
    assert _both_ways(session, old) == pytest.approx(read_wh, rel=2e-3)


def test_lifecycle_running_then_ended_then_filled_once(session):
    _readings(session)
    _read_world(session, stage2=False)
    _open_close(session, _read_act(), corr=REQUEST_ID, source="orion-hub", opened=H + timedelta(minutes=5),
                closed=STAGE1_END)
    # Stage 2 has not ended: no GPU, no energy.
    assert ledger.refill_late_costs(session, now=H + timedelta(minutes=40)).examined == 0
    read = session.execute(text("SELECT act_id FROM grammar_acts")).scalar()
    assert _cost(session, read).get("wh_by_host") is None
    # The 12 h wait passes with no end: a provisional fill, energy included.
    late = STAGE1_END + timedelta(seconds=ledger.LATE_COST_FINAL_AFTER_SEC + 60)
    assert ledger.refill_late_costs(session, now=late).filled == 1
    prov = _cost(session, read)
    assert prov["basis"]["wh"].startswith("late fill provisional") and prov["wh_by_host"]["circe"] > 0
    # Stage 2 shows up and ends: the final fill replaces the energy too, with both stages.
    _turn(session, stage=2, attempt=0, run=S2_RUN, corr=S2_CORR, at=H + timedelta(minutes=31))
    _hold(session, run=S2_RUN, corr=S2_CORR, lease=S2_HOLD, card="gpu1", start=H + timedelta(minutes=31),
          end=H + timedelta(minutes=55))
    _lease(session, "74bb29c5", "gpu1", H + timedelta(minutes=32), H + timedelta(minutes=32, seconds=181.7), S2_TRACE)
    session.execute(text("UPDATE world_pulse_read_seed SET stage2_status = 'done', stage2_trace_id = :t, "
                         "stage2_completed_at = :at WHERE seed_id = :s"), {"t": S2_TRACE, "at": STAGE2_END, "s": SEED})
    assert ledger.refill_late_costs(session, now=late + timedelta(hours=1)).filled == 1
    final = _cost(session, read)
    assert final["basis"]["wh"].startswith("late fill at")
    assert final["wh_by_host"]["circe"] > prov["wh_by_host"]["circe"]
    # Filled once: a later pass selects nothing.
    assert ledger.refill_late_costs(session, now=late + timedelta(hours=2)).examined == 0
    assert _cost(session, read) == final


def test_a_gpu_only_fill_from_before_3e_is_picked_up_for_its_energy_and_flag_off_leaves_it(session):
    _world(session)
    ledger.configure_act_energy_cost(False)
    ledger.refill_late_costs(session, now=NOW)
    read = session.execute(text("SELECT act_id FROM grammar_acts WHERE actor = 'hub.world_pulse_read'")).scalar()
    gpu_only = _cost(session, read)
    assert "wh" not in gpu_only["basis"]
    # Energy still off: the GPU-only fill is final, nothing re-selected.
    assert ledger.refill_late_costs(session, now=NOW + timedelta(minutes=1)).examined == 0
    ledger.configure_act_energy_cost(True)
    s = ledger.refill_late_costs(session, now=NOW + timedelta(minutes=2))
    assert (s.examined, s.filled) == (1, 1)
    after = _cost(session, read)
    assert after["wh_by_host"]["circe"] > 0
    assert after["gpu_slot_s_by_lane"] == pytest.approx(gpu_only["gpu_slot_s_by_lane"])
    assert ledger.refill_late_costs(session, now=NOW + timedelta(minutes=3)).examined == 0


def test_a_read_with_no_lease_gets_a_wh_line_and_is_not_reselected(session):
    session.execute(text("INSERT INTO world_pulse_read_seed (seed_id, status, created_at, completed_at, request_id) "
                         "VALUES (:s, 'skipped', :at, :at, CAST(:r AS uuid))"), {"s": SEED, "at": H, "r": REQUEST_ID})
    _open_close(session, _read_act(), corr=REQUEST_ID, source="orion-hub", opened=H, closed=H + timedelta(seconds=30))
    assert ledger.refill_late_costs(session, now=H + timedelta(minutes=5)).no_lease == 1
    cost = _cost(session, session.execute(text("SELECT act_id FROM grammar_acts")).scalar())
    assert cost.get("wh_by_host") is None and cost["basis"]["wh"].endswith("no lease found")
    assert ledger.refill_late_costs(session, now=H + timedelta(minutes=6)).examined == 0


def test_the_hourly_writer_stores_the_read_as_metered(session):
    from app.orion_energy_attribution import run_one_attribution

    _world(session)
    session.commit()
    engine = session.get_bind()
    run_one_attribution(engine, now=H + timedelta(hours=1, minutes=10))
    row = session.execute(text("SELECT metered_wh, leased_unmetered_wh FROM orion_energy_attribution_hourly "
                               "WHERE host = 'circe' AND hour_start = :h"), {"h": H}).first()
    new, old = _hour(session), _hour(session, charge_reads=False)
    assert row[0] == pytest.approx(new.metered_wh) and row[1] == pytest.approx(new.leased_unmetered_wh)
    assert row[0] > old.metered_wh + 10.0  # the read's ~37 Wh is in the stored metered_wh


def _backfill_module():
    spec = importlib.util.spec_from_file_location("backfill_reading_act_energy",
                                                  REPO / "scripts" / "backfill_reading_act_energy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stored_hour(sess):
    return sess.execute(text("SELECT metered_wh, leased_unmetered_wh, unattributed_wh, above_rest_wh, sponsored_wh "
                             "FROM orion_energy_attribution_hourly WHERE host = 'circe' AND hour_start = :h"),
                        {"h": H}).first()


def test_backfill_moves_both_sides_together_dry_run_writes_nothing_then_apply_then_restore(session, tmp_path):
    from app.orion_energy_attribution import UPSERT_SQL

    bf = _backfill_module()
    _world(session)
    # The pre-3e state as it is live: the read GPU-filled with no energy, the hour stored by the old rule.
    ledger.configure_act_energy_cost(False)
    ledger.refill_late_costs(session, now=NOW)
    ledger.configure_act_energy_cost(True)
    old = _hour(session, charge_reads=False)
    session.execute(UPSERT_SQL, {
        "hour_start": H, "host": "circe", "rest_w": 541.0, "rest_wh": old.rest_wh, "above_rest_wh": old.above_rest_wh,
        "sponsored_wh": old.sponsored_wh, "metered_wh": old.metered_wh, "leased_unmetered_wh": old.leased_unmetered_wh,
        "unattributed_wh": old.unattributed_wh, "covered_sec": old.covered_s, "split_unknown_sec": 0.0,
        "rest_source": "test"})
    session.commit()
    read = session.execute(text("SELECT act_id FROM grammar_acts WHERE actor = 'hub.world_pulse_read'")).scalar()
    act_before, hour_before = _cost(session, read), tuple(_stored_hour(session))
    stored = _stored_hour(session)
    assert _acts_wh(session) + stored.sponsored_wh + stored.unattributed_wh == pytest.approx(stored.above_rest_wh,
                                                                                            abs=0.05)
    now = H + timedelta(days=5)  # outside the live 96 h late-fill window: only the backfill reaches it

    # Dry run through main(): READ ONLY session, nothing written on either side.
    schema = session.execute(text("SELECT current_schema()")).scalar()
    uri = f"{session.get_bind().url.render_as_string(hide_password=False)}?options=-csearch_path%3D{schema}"
    assert bf.main(["--uri", uri, "--job-dir", str(tmp_path), "--days", "3650"]) == 0
    assert _cost(session, read) == act_before and tuple(_stored_hour(session)) == hour_before
    dry = bf.run(session, days=30, apply=False, job_dir=tmp_path, now=now, release=session.rollback,
                 out=io.StringIO())
    assert (dry["act_targets"], dry["acts_gain_energy"], dry["hour_targets"], dry["hours_changed"]) == (1, 1, 1, 1)
    assert dry["errors"] == 0 and dry["acts_written"] == 0 and dry["hours_written"] == 0
    assert dry["wh_moved_hourly_to_reads"] == pytest.approx(dry["read_wh_total"], rel=5e-3)
    assert dry["max_hour_closure_error_wh"] < 1e-6
    assert _cost(session, read) == act_before and tuple(_stored_hour(session)) == hour_before
    for name in ("progress.log", "report.md", "before_after.csv"):
        assert (tmp_path / name).exists(), name
    assert "100.0%" in (tmp_path / "progress.log").read_text()

    applied = bf.run(session, days=30, apply=True, job_dir=tmp_path, now=now, commit=session.commit,
                     out=io.StringIO())
    assert (applied["acts_written"], applied["hours_written"]) == (1, 1)
    stored = _stored_hour(session)
    read_wh = _cost(session, read)["wh_by_host"]["circe"]
    assert stored.metered_wh - hour_before[0] == pytest.approx(read_wh, rel=2e-3)
    # Each Wh once, after: per-act total + stored remainder == stored metered host energy.
    assert _acts_wh(session) + stored.sponsored_wh + stored.unattributed_wh == pytest.approx(stored.above_rest_wh,
                                                                                            abs=0.05)
    # A second apply changes nothing (no act target; the hour's absolute values are the same).
    again = bf.run(session, days=30, apply=True, job_dir=tmp_path, now=now + timedelta(minutes=1),
                   commit=session.commit, out=io.StringIO())
    assert again["act_targets"] == 0 and again["hours_changed"] == 0
    # ...because the stored hour already IS the new rule, not because it was rejected as drift.
    assert (again["hours_already_applied"], again["hours_drift"], again["anomalies"]) == (1, 0, 0)
    assert _stored_hour(session).metered_wh == pytest.approx(stored.metered_wh)

    got = bf.restore(session, tmp_path, commit=session.commit)
    assert got == {"acts": 1, "hours": 1}
    assert _cost(session, read) == act_before
    assert tuple(_stored_hour(session)) == pytest.approx(hour_before)


def test_backfill_leaves_a_drifted_hour_alone(session, tmp_path):
    from app.orion_energy_attribution import UPSERT_SQL

    bf = _backfill_module()
    _world(session)
    old = _hour(session, charge_reads=False)
    session.execute(UPSERT_SQL, {
        "hour_start": H, "host": "circe", "rest_w": 541.0, "rest_wh": old.rest_wh,
        "above_rest_wh": old.above_rest_wh + 5.0, "sponsored_wh": 0.0, "metered_wh": old.metered_wh,
        "leased_unmetered_wh": old.leased_unmetered_wh, "unattributed_wh": old.unattributed_wh + 5.0,
        "covered_sec": 3600.0, "split_unknown_sec": 0.0, "rest_source": "test"})
    session.commit()
    out = bf.run(session, days=30, apply=True, job_dir=tmp_path, now=H + timedelta(days=5), commit=session.commit,
                 out=io.StringIO())
    assert out["hours_drift"] == 1 and out["hours_written"] == 0
    assert out["wh_still_double_counted_in_drift_hours"] > 10.0
    assert _stored_hour(session).metered_wh == pytest.approx(old.metered_wh)


def test_the_reads_own_hold_is_charged_its_energy_not_its_slot_time(session):
    """Review S1: a call inside its run's hold shares the card 1/2 with the hold on both meters; the
    hold is the read's too, so the read carries the card's whole above-idle share."""
    _readings(session)
    session.execute(text("INSERT INTO world_pulse_read_seed (seed_id, status, trace_id, created_at, completed_at, "
                         "request_id) VALUES (:s, 'failed', :t, :c, :d, CAST(:r AS uuid))"),
                    {"s": SEED, "t": S1_TRACE, "c": H, "d": STAGE1_END, "r": REQUEST_ID})
    _turn(session, stage=1, attempt=1, run=S1_RUN, corr=S1_CORR, at=H + timedelta(minutes=5))
    _hold(session, run=S1_RUN, corr=S1_CORR, lease=S1_HOLD, card="gpu2", start=H + timedelta(minutes=10),
          end=H + timedelta(minutes=20))
    _lease(session, "solo", "gpu2", H + timedelta(minutes=10), H + timedelta(minutes=20), S1_TRACE, "agent-gpu2")
    _open_close(session, _read_act(), corr=REQUEST_ID, source="orion-hub", opened=H + timedelta(minutes=5),
                closed=STAGE1_END)
    ledger.refill_late_costs(session, now=NOW)
    cost = _cost(session, session.execute(text("SELECT act_id FROM grammar_acts")).scalar())
    assert cost["gpu_slot_s_by_lane"] == {"agent-gpu2": pytest.approx(600.0)}  # the call only
    hour = _hour(session)
    # Nothing else is on gpu2: the card's whole share is metered, and all of it is the read's.
    assert hour.leased_unmetered_wh == pytest.approx(0.0, abs=1e-6)
    assert cost["wh_by_host"]["circe"] == pytest.approx(hour.metered_wh, rel=2e-3)
    assert "1 pool hold(s)" in cost["basis"]["wh"]
    assert _both_ways(session, hour) == pytest.approx(0.0, abs=0.05)


def test_an_energy_fill_that_raises_is_retried_not_written_off(session, monkeypatch):
    """Review S2: one raise must not lock the read uncharged while the hour meters its turns."""
    _world(session)
    real = ledger.act_energy_cost
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return real(*a, **k)

    monkeypatch.setattr(ledger, "act_energy_cost", flaky)
    ledger.refill_late_costs(session, now=NOW)
    read = session.execute(text("SELECT act_id FROM grammar_acts WHERE actor = 'hub.world_pulse_read'")).scalar()
    first = _cost(session, read)
    assert first["gpu_slot_s_by_lane"] and "wh" not in first["basis"]
    s = ledger.refill_late_costs(session, now=NOW + timedelta(minutes=1))
    assert s.examined == 1 and _cost(session, read)["wh_by_host"]["circe"] > 0
    assert _both_ways(session, _hour(session)) == pytest.approx(0.0, abs=0.05)
