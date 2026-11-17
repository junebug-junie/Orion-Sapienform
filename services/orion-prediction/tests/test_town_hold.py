"""Family C Experiment 1, the town-hold runner (app/town_hold.py) and its analysis (app/family_c.py).

Pins (design 2026-11-16, "Acceptance checks / Patch 1"):
- the pending hold is set 60 s before ``start_eligible_at``; an ON arm deletes it; OFF keeps it 40 min with
  ``set_at`` fixed at the block start; every write carries a TTL (a dead runner frees the town);
- the race: the town is read AGAIN after the key is set; a start in flight (or an open session) voids the
  decision and no arm is used;
- a decision is refused (no arm, key deleted, refusal row) with tier-0/1 work queued on fast, a hot or
  missing cabinet reading, strain latched or unknown, a deferred bid, or no answer from the market;
- the arm is logged BEFORE the key changes; more than 2 OFF in a row is impossible (drawn, and guarded);
- a session inside an OFF block aborts the night as contaminated; recover() cancels an interrupted night and
  deletes the run-cap key; the run resumes from its next night;
- the pooled analysis recovers a planted effect on a Family A reading and on a lane cost, drops a manual
  block, and its restricted permutation is calibrated under the null.
"""
from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from app.causal import analyze_stratified_switchback, draw_arm_sequence, max_off_run
from app.experiment_runner import ACTIVE_KEY, LOCK_KEY, LUA_REFRESH, TOWN_HOLD_SCHEDULE_KEY, recover
from app.family_c import EVENT_KIND, EffectTable, event_blocks, lane_outcomes
from app.store import PostgresStore
from app.town_hold import (
    ACTION,
    TownHoldConfig,
    TownHoldRefused,
    TownHoldRunner,
    TownView,
    first_night_after,
    night_window,
    plan_night,
    run_cap_key,
)
from orion.schemas.physiology_run_cap import RunCapOverrideV1

KEY = run_cap_key()
REPO = Path(__file__).resolve().parents[3]
DEN = "America/Denver"


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


class Redis:
    """SET NX/EX, GET, DELETE, TTL, the two Lua scripts; ``ops`` records every key write in order."""

    def __init__(self, clock, log):
        self.clock, self.data, self.log = clock, {}, log
        self.h: dict = {}

    def _live(self, k):
        v = self.data.get(k)
        if v and v[1] is not None and v[1] <= self.clock():
            del self.data[k]
            return None
        return v

    async def set(self, k, value, nx=False, ex=None):
        if nx and self._live(k):
            return None
        self.data[k] = (value, self.clock() + timedelta(seconds=ex) if ex else None)
        self.log.append(("set", k, ex))
        return True

    async def get(self, k):
        v = self._live(k)
        return v[0] if v else None

    async def delete(self, k):
        self.log.append(("del", k, None))
        return 1 if self.data.pop(k, None) else 0

    async def eval(self, script, numkeys, key, owner, *args):
        cur = self._live(key)
        if not cur or cur[0] != owner:
            return None if script == LUA_REFRESH else 0
        if script == LUA_REFRESH:
            return await self.set(key, owner, ex=int(args[0]))
        return await self.delete(key)

    async def hgetall(self, key):
        return dict(self.h.get(key, {}))

    def override(self):
        v = self._live(KEY)
        return RunCapOverrideV1.model_validate_json(v[0]) if v else None


class Store:
    def __init__(self, log):
        self.log = log
        self.exps, self.ivs, self.effect_rows = {}, [], []
        self.queued = 0
        self.cabinet = 25.0
        self.acts: list = []
        self.leases: list = []
        self.kev: list = []
        self.fail_leases = False

    def insert_experiment(self, row):
        self.exps[row["experiment_id"]] = {**row, "status": "designed"}

    def experiment(self, eid):
        return self.exps.get(eid)

    def experiments(self, status=None, limit=500):
        return [e for e in self.exps.values() if status is None or e["status"] == status]

    def set_experiment_status(self, eid, status):
        self.exps[eid]["status"] = status

    def insert_intervention(self, row):
        row = {"intervention_id": f"iv{len(self.ivs)}", **row}
        self.ivs.append(row)
        self.log.append(("iv", row.get("arm"), json.loads(row["notes"] or "{}")))
        return row

    def interventions(self, action=None, experiment_id=None, limit=500):
        return [r for r in self.ivs if experiment_id is None or r["experiment_id"] == experiment_id][:limit]

    def insert_effect(self, row):
        self.effect_rows.append(row)
        return row

    def queued_fast_leases(self, older, priorities):
        if self.fail_leases:
            raise ConnectionError("db down")
        return self.queued

    def points(self, spec, since, until=None):
        return [] if self.cabinet is None else [(since + timedelta(minutes=5), self.cabinet)]

    def town_acts(self, a, b):
        return [x for x in self.acts if x["opened_at"] < b and (x.get("closed_at") is None or x["closed_at"] > a)]

    def lane_leases(self, a, b):
        return self.leases

    def kev_latencies(self, a, b):
        return self.kev


class Effects:
    def __init__(self, clock, store):
        self.engine = SimpleNamespace(clock=clock)
        self.store = store

    def _spec(self, series_id, role=None):
        return SimpleNamespace(threshold=lambda name: SimpleNamespace(value=31.0) if name == "experiment_guard" else None)

    def log_intervention(self, action, started_at, ended_at, source, notes="", arm=None, experiment_id=None):
        assert ended_at > started_at
        return self.store.insert_intervention({"action": action, "started_at": started_at, "ended_at": ended_at,
                                               "source": source, "notes": notes, "arm": arm,
                                               "experiment_id": experiment_id})


class Probe:
    """The town: eligible at ``eligible_at`` unless a session is open. ``on_view`` lets a test change state
    between the first read and the post-set read."""

    def __init__(self, clock):
        self.clock = clock
        self.eligible_at = None
        self.session_open = self.manual = self.in_flight = False
        self.configured = True
        self.down = False
        self.reads = 0
        self.after_first = None

    def view(self):
        self.reads += 1
        if self.down:
            return None
        if self.after_first is not None and self.reads % 2 == 0:
            self.after_first(self)
        return TownView(eligible_at=None if self.session_open or self.manual else self.eligible_at,
                        session_open=self.session_open, manual_session_open=self.manual,
                        start_in_flight=self.in_flight, override_configured=self.configured, running=True)


async def _direct(fn, *a):
    return fn(*a)


def funded(want):
    return SimpleNamespace(outcome="funded", reason="net_positive", decision_id=f"d-{want.want_id}")


def regulation(clock, latched=False):
    return json.dumps({"generated_at": clock().isoformat(), "arousal": {"strain_latched": latched}})


def world(t0, *, arms=None, bid=funded):
    clock = Clock(t0)
    log: list = []
    redis, store = Redis(clock, log), Store(log)
    probe = Probe(clock)
    runner = TownHoldRunner(Effects(clock, store), redis, probe, bid=bid, release=lambda w: None,
                            to_thread=_direct, cfg=TownHoldConfig())
    asyncio.run(redis.set("orion:regulation:latest", regulation(clock)))
    sched = {"run_id": "run-1", "first_night": "2026-11-17", "nights": 1, "seed": 11}
    return SimpleNamespace(clock=clock, redis=redis, store=store, probe=probe, runner=runner, log=log, sched=sched,
                           arms=arms)


def open_night(w, day=date(2026, 11, 17)):
    """Create the night's experiment and state the way _run_night does, without the sleep loop."""
    from app.town_hold import NightState

    plan = plan_night(w.sched["seed"], day, w.runner.cfg)
    arms = w.arms or plan["arms"]
    eid = "exp-night-1"
    w.store.insert_experiment({"experiment_id": eid, "action": ACTION, "series_id": "circe.gpu3_watts",
                               "created_at": w.clock(), "burn_in_min": 2, "guard": {},
                               "design": {"kind": EVENT_KIND, "run_id": "run-1", "night": day.isoformat(),
                                          "arms": arms, "max_off_run": 2, "hold_min": 40, "window_min": [2, 35],
                                          "n_blocks": len(arms), "seed": plan["seed"]}})
    w0, w1 = night_window(day, w.runner.cfg)
    st = NightState(eid, day.isoformat(), w0, w1, list(arms), plan["seed"])
    asyncio.run(w.redis.set(LOCK_KEY, eid, ex=180))
    return st


def arrive(w, st):
    """The town will be eligible in 30 s: the pending hold goes on now (verified), the decision comes next tick."""
    w.probe.eligible_at = w.clock() + timedelta(seconds=30)
    assert step(w, st) == "pending"
    w.clock.t += timedelta(seconds=30)


def step(w, st):
    asyncio.run(w.redis.set("orion:regulation:latest", regulation(w.clock)))
    return asyncio.run(w.runner.step(st))


T_NIGHT = datetime(2026, 11, 17, 7, 0, tzinfo=timezone.utc)   # 00:00 America/Denver (MST)


# --- arm sequences -----------------------------------------------------------------------------------

def test_arm_sequences_are_balanced_and_never_hold_three_times_in_a_row():
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(3000):
        arms = draw_arm_sequence(8, 2, rng)
        assert arms.count("on") == arms.count("off") == 4
        assert max_off_run(arms) <= 2
        seen.add(tuple(arms))
    assert len(seen) > 30   # many distinct sequences, not one


def test_a_night_plan_is_seeded_and_the_window_is_local_midnight_to_eight():
    cfg = TownHoldConfig()
    a, b = plan_night(5, date(2026, 11, 17), cfg), plan_night(5, date(2026, 11, 17), cfg)
    assert a == b and plan_night(5, date(2026, 11, 18), cfg)["seed"] != a["seed"]
    w0, w1 = night_window(date(2026, 11, 17), cfg)
    assert (w0, w1) == (T_NIGHT, T_NIGHT + timedelta(hours=8))
    assert first_night_after(T_NIGHT + timedelta(hours=1), cfg) == date(2026, 11, 18)


# --- the pending hold, the arms, the key ------------------------------------------------------------

def test_pending_hold_is_set_60s_before_eligibility_and_an_on_arm_deletes_it():
    w = world(T_NIGHT + timedelta(minutes=10), arms=["on", "off", "on", "off", "on", "off", "on", "off"])
    st = open_night(w)
    w.probe.eligible_at = w.clock() + timedelta(seconds=90)
    assert step(w, st) == "waiting" and w.redis.override() is None
    w.clock.t += timedelta(seconds=31)    # 59 s before eligibility
    assert step(w, st) == "pending"
    ov = w.redis.override()
    assert ov.phase == "pending" and ov.block_index == 0
    w.clock.t += timedelta(seconds=60)
    assert step(w, st) == "decided:on"
    assert w.redis.override() is None and st.k == 1 and st.taken == ["on"]
    # The arm is on the record before the key changed.
    i_iv = next(i for i, e in enumerate(w.log) if e[0] == "iv" and e[1] == "on")
    i_del = max(i for i, e in enumerate(w.log) if e[0] == "del" and e[1] == KEY)
    assert i_iv < i_del
    assert all(ex for op, k, ex in w.log if op == "set" and k == KEY)   # never a key without a TTL


def test_an_off_block_holds_40_min_with_set_at_fixed_then_hands_over_to_the_next_decision():
    w = world(T_NIGHT + timedelta(minutes=10), arms=["off", "on", "on", "off", "on", "off", "on", "off"])
    st = open_night(w)
    w.probe.eligible_at = w.clock() + timedelta(seconds=30)
    assert step(w, st) == "pending"
    w.clock.t += timedelta(seconds=30)
    assert step(w, st) == "decided:off"
    start = w.redis.override().set_at
    i_iv = next(i for i, e in enumerate(w.log) if e[0] == "iv" and e[1] == "off")
    i_block = next(i for i, e in enumerate(w.log) if e[0] == "set" and e[1] == KEY and i > i_iv)
    assert i_iv < i_block
    for _ in range(79):
        w.clock.t += timedelta(seconds=30)
        assert step(w, st) == "holding"
        ov = w.redis.override()
        assert ov.phase == "block" and ov.set_at == start and ov.block_index == 0
    w.clock.t += timedelta(seconds=30)    # 40 min: the block ends; the town is still eligible -> next decision
    assert step(w, st) == "decided:on"
    assert w.redis.override() is None and st.taken == ["off", "on"]


def test_three_holds_in_a_row_are_impossible_even_with_a_bad_arm_list():
    w = world(T_NIGHT + timedelta(minutes=10), arms=["off", "off", "off", "on", "on", "on", "on", "off"])
    st = open_night(w)
    st.taken = ["off", "off"]
    st.k = 2
    arrive(w, st)
    with pytest.raises(Exception, match="play floor"):
        step(w, st)


# --- the race --------------------------------------------------------------------------------------

def test_a_start_in_flight_when_the_key_lands_voids_the_decision():
    w = world(T_NIGHT + timedelta(minutes=10))
    st = open_night(w)
    w.probe.eligible_at = w.clock()

    def started_meanwhile(p):
        p.in_flight = True

    w.probe.after_first = started_meanwhile
    assert step(w, st) == "void"
    assert w.redis.override() is None and st.k == 0 and st.voided == 1
    assert not [e for e in w.log if e[0] == "iv"]


def test_a_session_already_open_when_the_key_lands_voids_the_decision():
    w = world(T_NIGHT + timedelta(minutes=10))
    st = open_night(w)
    w.probe.eligible_at = w.clock()

    def opened(p):
        p.session_open = True

    w.probe.after_first = opened
    assert step(w, st) == "void" and st.k == 0


# --- stop rules and the bid ---------------------------------------------------------------------------

@pytest.mark.parametrize("setup,reason", [
    (lambda w: setattr(w.store, "queued", 2), "tier0_1_queued_on_fast"),
    (lambda w: setattr(w.store, "cabinet", 31.5), "cabinet_hot"),
    (lambda w: setattr(w.store, "cabinet", None), "cabinet_missing"),
    (lambda w: setattr(w.store, "fail_leases", True), "lease_read_failed"),
])
def test_a_stop_rule_refuses_the_decision_without_using_an_arm(setup, reason):
    w = world(T_NIGHT + timedelta(minutes=10))
    st = open_night(w)
    setup(w)
    arrive(w, st)
    assert step(w, st) == "refused"
    assert st.k == 0 and w.redis.override() is None and reason in st.refusals
    refusal = [e for e in w.log if e[0] == "iv"][-1]
    assert refusal[1] is None and any(r.startswith(reason) for r in refusal[2]["refused"])


def test_strain_latched_or_unknown_refuses():
    w = world(T_NIGHT + timedelta(minutes=10))
    st = open_night(w)
    arrive(w, st)
    asyncio.run(w.redis.set("orion:regulation:latest", regulation(w.clock, latched=True)))
    assert asyncio.run(w.runner.stop_reasons(w.clock())) == ["strain_latched"]
    asyncio.run(w.redis.delete("orion:regulation:latest"))
    assert asyncio.run(w.runner.stop_reasons(w.clock())) == ["strain_unknown"]


@pytest.mark.parametrize("bid,why", [
    (lambda want: SimpleNamespace(outcome="deferred", reason="price", decision_id="d"), "bid:deferred:price"),
    (lambda want: None, "bid:no_answer"),
])
def test_only_a_funded_bid_runs_a_block(bid, why):
    w = world(T_NIGHT + timedelta(minutes=10), bid=bid)
    st = open_night(w)
    arrive(w, st)
    assert step(w, st) == "refused"
    assert st.k == 0 and st.refusals == {why: 1}


def test_the_bid_is_a_binding_tier2_want_hungry_since_the_last_decision():
    seen = []

    def bid(want):
        seen.append(want)
        return funded(want)

    w = world(T_NIGHT + timedelta(minutes=50), bid=bid)
    st = open_night(w)
    arrive(w, st)
    step(w, st)
    (want,) = seen
    from orion.physiology.binding import BINDING_PRODUCERS
    assert (want.family, want.tier, want.producer) == ("prediction.experiment", 2, "orion-prediction.experiment.binding")
    assert want.producer in BINDING_PRODUCERS
    assert want.opened_at == w.clock() - timedelta(minutes=40)   # capped at one block


# --- contamination, the night's end, restart ---------------------------------------------------------

def test_a_session_open_inside_an_off_block_aborts_the_night():
    w = world(T_NIGHT + timedelta(minutes=10), arms=["off", "on", "on", "off", "on", "off", "on", "off"])
    st = open_night(w)
    arrive(w, st)
    assert step(w, st) == "decided:off"
    w.clock.t += timedelta(seconds=30)
    w.probe.session_open = True
    with pytest.raises(Exception, match="contaminated"):
        step(w, st)


def test_an_act_opened_inside_an_off_block_aborts_the_night_at_its_end():
    w = world(T_NIGHT + timedelta(minutes=10), arms=["off", "on", "on", "off", "on", "off", "on", "off"])
    st = open_night(w)
    arrive(w, st)
    step(w, st)
    w.store.acts.append({"opened_at": w.clock() + timedelta(minutes=5), "closed_at": w.clock() + timedelta(minutes=6)})
    for _ in range(79):
        w.clock.t += timedelta(seconds=30)
        assert step(w, st) == "holding"
    w.clock.t += timedelta(seconds=30)
    with pytest.raises(Exception, match="contaminated"):
        step(w, st)


def test_no_decisions_after_eight_and_the_night_drains_its_checks():
    w = world(T_NIGHT + timedelta(hours=8, seconds=1))
    st = open_night(w)
    w.probe.eligible_at = w.clock()
    assert step(w, st) == "done" and w.redis.override() is None


def test_recover_cancels_an_interrupted_night_and_deletes_the_hold():
    w = world(T_NIGHT + timedelta(minutes=10), arms=["off", "on", "on", "off", "on", "off", "on", "off"])
    st = open_night(w)
    asyncio.run(w.redis.set(ACTIVE_KEY, st.experiment_id))
    arrive(w, st)
    step(w, st)
    assert w.redis.override() is not None
    eff = w.runner.effects

    async def go():
        return await recover(eff, w.redis, to_thread=_direct)

    assert asyncio.run(go()) == [st.experiment_id]
    assert w.store.exps[st.experiment_id]["status"] == "cancelled" and w.redis.override() is None
    assert asyncio.run(w.redis.get(LOCK_KEY)) is None


def test_the_schedule_skips_a_night_already_recorded():
    w = world(T_NIGHT + timedelta(minutes=10))
    open_night(w)
    assert w.runner._existing_nights("run-1") == {"2026-11-17"}


# --- start refusals --------------------------------------------------------------------------------

def test_start_refuses_when_the_town_cannot_be_read_or_does_not_obey_the_override():
    w = world(T_NIGHT - timedelta(hours=3))
    w.probe.down = True
    with pytest.raises(TownHoldRefused) as e:
        asyncio.run(w.runner.start(16))
    assert e.value.status == 503
    w.probe.down = False
    w.probe.configured = False
    with pytest.raises(TownHoldRefused) as e:
        asyncio.run(w.runner.start(16))
    assert e.value.status == 422
    assert asyncio.run(w.redis.get(TOWN_HOLD_SCHEDULE_KEY)) is None


def test_start_schedules_the_next_night_and_holds_nothing_yet():
    w = world(T_NIGHT - timedelta(hours=3))

    async def go():
        out = await w.runner.start(16, seed=3)
        await asyncio.sleep(0)
        sched = json.loads(await w.redis.get(TOWN_HOLD_SCHEDULE_KEY))
        await w.runner.shutdown("test", clear_schedule=True)
        return out, sched

    out, sched = asyncio.run(go())
    assert out["schedule"]["first_night"] == "2026-11-17" and sched["nights"] == 16 and sched["seed"] == 3
    assert w.redis.override() is None and not w.store.exps


def test_the_fast_lane_classes_match_the_pool_config():
    pool = yaml.safe_load((REPO / "config" / "gpu_pool.yaml").read_text(encoding="utf-8"))
    first_fast = {c for c, body in pool["classes"].items() if body["roles"][0] == "fast" and not c.startswith("operator")}
    assert first_fast == {"fast", "metacog"}
    assert "work_class IN ('fast', 'metacog')" in PostgresStore.FAST_LANE_SQL


# --- the pooled analysis --------------------------------------------------------------------------------

def _redraw(n=8):
    return lambda s, rng: draw_arm_sequence(n, 2, rng)


def _blocks(nights, effect, rng, noise=1.0):
    out = []
    for s in range(nights):
        arms = draw_arm_sequence(8, 2, rng)[:7]
        night_level = rng.normal(0, 5)
        for i, a in enumerate(arms):
            out.append({"stratum": f"n{s}", "seq": i, "arm": a,
                        "value": night_level + (effect if a == "on" else 0.0) + rng.normal(0, noise)})
    return out


def test_the_stratified_estimate_recovers_a_planted_effect_through_night_drift():
    est = analyze_stratified_switchback(_blocks(16, 3.0, np.random.default_rng(1)), _redraw(), n_perm=500)
    assert est.ci_lo < 3.0 < est.ci_hi and est.p_value < 0.01


def test_the_restricted_permutation_is_calibrated_under_the_null():
    rng = np.random.default_rng(2)
    ps = [analyze_stratified_switchback(_blocks(6, 0.0, rng), _redraw(), n_perm=199).p_value for _ in range(120)]
    assert 0.0 <= np.mean(np.array(ps) <= 0.05) <= 0.12


def test_the_vectorised_permutation_matches_the_plain_estimator():
    from app.causal import _perm_estimates, _stratified_diff, redraws

    blocks = _blocks(5, 1.0, np.random.default_rng(3))
    blocks[2]["value"] = None
    strata = sorted({b["stratum"] for b in blocks})
    draws = redraws(strata, _redraw(), 50, seed=1)
    values = {s: [b["value"] for b in blocks if b["stratum"] == s] for s in strata}
    seqs = {s: [b["seq"] for b in blocks if b["stratum"] == s] for s in strata}
    fast = _perm_estimates(values, seqs, draws["on"], 50)
    for k in range(50):
        arms = {s: ["on" if draws["on"][s][k, i] else "off" for i in seqs[s]] for s in strata}
        slow, _w, _a, _b = _stratified_diff(values, arms)
        assert fast[k] == pytest.approx(slow)


def test_too_few_blocks_is_no_estimate_not_zero():
    est = analyze_stratified_switchback([{"stratum": "a", "seq": 0, "arm": "on", "value": 1.0}], _redraw())
    assert est.effect is None and "need" in est.details["reason"]


def _ts(t):
    return pd.Timestamp(t)


def test_lane_outcomes_measure_tier1_hold_and_non_town_wait_only():
    w0 = _ts("2026-11-17T07:02:00Z")
    w1 = _ts("2026-11-17T07:35:00Z")
    t = w0 + pd.Timedelta(minutes=1)
    leases = [
        {"holder": "cortex-exec", "work_class": "metacog", "priority": "system", "status": "released",
         "queued_since": t, "granted_at": t + pd.Timedelta(seconds=1), "updated_at": t + pd.Timedelta(seconds=11)},
        {"holder": "http:openai", "work_class": "fast", "priority": "background", "status": "released",
         "queued_since": t, "granted_at": t + pd.Timedelta(seconds=9), "updated_at": t + pd.Timedelta(seconds=20)},
        {"holder": "cortex-exec", "work_class": "metacog", "priority": "system", "status": "released",
         "queued_since": w1, "granted_at": w1, "updated_at": w1 + pd.Timedelta(seconds=99)},   # outside
    ]
    out = lane_outcomes(leases, [(t, 200.0), (t, 300.0)], w0, w1)
    assert out["lane.fast.tier1_hold_s"] == pytest.approx(10.0)
    assert out["lane.fast.grant_wait_s"] == pytest.approx(1.0)   # the town's own wait is not counted
    assert out["lane.kev.latency_ms_p50"] == pytest.approx(250.0)
    assert lane_outcomes([], [], w0, w1) == {k: None for k in out}


def test_event_blocks_refuse_an_arm_that_is_not_the_drawn_one():
    exp = {"experiment_id": "e", "action": ACTION, "design": {"arms": ["on", "off"], "window_min": [2, 35]}}
    good = {"action": ACTION, "arm": "on", "started_at": T_NIGHT, "notes": json.dumps({"block_index": 0})}
    assert event_blocks(exp, [good])[0]["seq"] == 0
    bad = {**good, "arm": "off"}
    with pytest.raises(Exception, match="design says"):
        event_blocks(exp, [bad])


def _run_world(effect_w=8.0, effect_hold=5.0, nights=10, seed=4):
    """Analysed nights with planted effects on circe.gpu3_watts and on tier-1 hold; one manual block."""
    rng = np.random.default_rng(seed)
    store = Store([])
    idx = pd.date_range(T_NIGHT - timedelta(days=15), T_NIGHT + timedelta(days=nights + 1), freq="5min", tz="UTC")
    tod = (idx.hour * 60 + idx.minute).to_numpy()
    gpu = 100 + 10 * np.sin(2 * np.pi * tod / 1440) + rng.normal(0, 3, len(idx))
    gpu = pd.Series(gpu, index=idx)
    for n in range(nights):
        day = date(2026, 11, 17) + timedelta(days=n)
        arms = draw_arm_sequence(8, 2, np.random.default_rng(100 + n))
        eid = f"night-{n}"
        store.exps[eid] = {"experiment_id": eid, "action": ACTION, "status": "analyzed",
                           "design": {"kind": EVENT_KIND, "run_id": "run-x", "night": day.isoformat(), "arms": arms,
                                      "max_off_run": 2, "hold_min": 40, "window_min": [2, 35]}}
        t = night_window(day, TownHoldConfig())[0] + timedelta(minutes=3)
        for i, a in enumerate(arms[:6]):
            store.ivs.append({"intervention_id": f"{eid}-{i}", "experiment_id": eid, "action": ACTION, "arm": a,
                              "started_at": t, "ended_at": t + timedelta(minutes=40),
                              "notes": json.dumps({"block_index": i})})
            manual = (n == 0 and i == 0)
            store.ivs.append({"intervention_id": f"{eid}-c{i}", "experiment_id": eid, "action": ACTION, "arm": None,
                              "started_at": t + timedelta(minutes=41), "ended_at": t + timedelta(minutes=41, seconds=1),
                              "notes": json.dumps({"block_check": i, "manual_session": manual})})
            w0, w1 = pd.Timestamp(t) + pd.Timedelta(minutes=2), pd.Timestamp(t) + pd.Timedelta(minutes=35)
            if a == "on":
                gpu[(gpu.index >= w0) & (gpu.index < w1)] += effect_w
                store.acts.append({"opened_at": t + timedelta(minutes=1), "closed_at": t + timedelta(minutes=31)})
            hold = 4.0 + (effect_hold if a == "on" else 0.0) + rng.normal(0, 1)
            g = w0 + pd.Timedelta(minutes=3)
            store.leases.append({"holder": "cortex-exec", "work_class": "metacog", "priority": "system",
                                 "status": "released", "queued_since": g, "granted_at": g,
                                 "updated_at": g + pd.Timedelta(seconds=hold)})
            t = t + timedelta(minutes=60 if a == "off" else 95)

    def loader(a, b):
        s = gpu[(gpu.index >= pd.Timestamp(a)) & (gpu.index < pd.Timestamp(b))]
        return pd.DataFrame({"circe.gpu3_watts": s}), {}

    return store, EffectTable(store, loader, units={"circe.gpu3_watts": "W"}, grain_min=5, n_perm=300)


def test_the_effect_table_has_randomised_rows_for_watts_and_tier1_hold():
    store, table = _run_world()
    out = table.build()
    assert not out["errors"], out["errors"]
    rows = {r["target"]: r for r in out["rows"]}
    gw, hold = rows["circe.gpu3_watts"], rows["lane.fast.tier1_hold_s"]
    assert gw["actuator"] == ACTION and gw["method"] == "randomised" and gw["window_start_min"] == 2
    assert gw["ci_lo"] < 8.0 < gw["ci_hi"] and gw["profile_adjusted"]
    assert hold["ci_lo"] < 5.0 < hold["ci_hi"] and hold["p_value"] < 0.05
    assert len(gw["experiment_ids"]) == 10
    assert gw["n_on"] + gw["n_off"] == 59   # 60 blocks, the manual one left out


def test_the_run_result_is_one_row_with_the_experience_payload():
    from app.town_hold import run_result

    store, table = _run_world()
    row = run_result(table, store, "run-x")
    d = row["details"]
    assert d["estimator"] == "family_c_pooled_run" and d["effect_z"] > 2
    assert d["manipulation"]["verdict"] == "passed" and d["manipulation"]["manual_blocks_excluded"] == 1
    assert "GPU3" in d["summary"] and "tier-1" in d["summary"]


def test_the_manipulation_check_catches_a_leak():
    from app.family_c import manipulation_summary

    blocks = [{"stratum": "a", "seq": 0, "arm": "off", "manip": {"session_s": 600.0, "sessions_started_in_hold": 1}},
              {"stratum": "a", "seq": 1, "arm": "on", "manip": {"session_s": 1800.0, "sessions_started_in_hold": 1}}]
    assert manipulation_summary(blocks)["verdict"] == "contaminated"


# --- the whole night, end to end, against a simulated town ---------------------------------------------

class SimTown(Probe):
    """The town's own rules: it ticks every 300 s, starts a session when its gate is open AND the run-cap key
    is not live (read at the start, like the controller), plays 30 min, then waits a 60-min gap."""

    def __init__(self, clock, redis, store, daily_cap_sessions=None):
        super().__init__(clock)
        self.redis, self.store = redis, store
        self.eligible_at = clock()
        self.session_end = None
        self.next_tick = clock()
        self.held = 0
        self.daily_cap_sessions = daily_cap_sessions   # the 2 h cap = 4 sessions per local day

    def _local_day(self, t):
        from zoneinfo import ZoneInfo

        return t.astimezone(ZoneInfo(DEN)).date()

    def advance(self):
        now = self.clock()
        if self.session_end is not None and now >= self.session_end:
            self.session_open, self.eligible_at = False, self.session_end + timedelta(minutes=60)
            today = [a for a in self.store.acts if self._local_day(a["opened_at"]) == self._local_day(self.session_end)]
            if self.daily_cap_sessions is not None and len(today) >= self.daily_cap_sessions:
                from zoneinfo import ZoneInfo

                local = self.session_end.astimezone(ZoneInfo(DEN))
                midnight = local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                self.eligible_at = max(self.eligible_at, midnight.astimezone(timezone.utc))
            self.session_end = None
        while self.next_tick <= now:
            t = self.next_tick
            self.next_tick += timedelta(seconds=300)
            today = [a for a in self.store.acts if self._local_day(a["opened_at"]) == self._local_day(t)]
            if self.daily_cap_sessions is not None and len(today) >= self.daily_cap_sessions:
                from zoneinfo import ZoneInfo

                local = t.astimezone(ZoneInfo(DEN))
                midnight = (local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
                self.eligible_at = max(self.eligible_at, midnight.astimezone(timezone.utc))
                continue
            if not self.session_open and self.eligible_at <= t:
                if self.redis._live(KEY):
                    self.held += 1
                    continue
                self.session_open, self.session_end = True, t + timedelta(minutes=30)
                self.store.acts.append({"opened_at": t, "closed_at": t + timedelta(minutes=30)})

    def view(self):
        self.advance()
        return super().view()


def test_a_whole_night_alternates_holds_and_play_with_no_leak():
    w = world(T_NIGHT - timedelta(minutes=1))
    town = SimTown(w.clock, w.redis, w.store)
    w.runner.probe = town

    async def sleep(sec):
        w.clock.t += timedelta(seconds=sec)
        await w.redis.set("orion:regulation:latest", regulation(w.clock))

    w.runner._sleep = sleep
    from orion.embodiment.town_sessions import load_rows  # noqa: F401 -- the block check reads the ledger

    asyncio.run(w.runner._run_night(w.sched, date(2026, 11, 17)))
    night = w.runner.nights_done[-1]
    assert night["status"] == "analyzed", night
    exp = next(iter(w.store.exps.values()))
    arms = [r["arm"] for r in sorted((r for r in w.store.ivs if r.get("arm")), key=lambda r: r["started_at"])]
    assert arms == exp["design"]["arms"][:len(arms)] and len(arms) >= 4
    assert max_off_run(arms) <= 2 and "off" in arms and "on" in arms
    assert town.held > 0   # the key really held the town
    # No session started inside any OFF hold; every ON block's session started within one town tick.
    for r in w.store.ivs:
        if r.get("arm") == "off":
            assert not [a for a in w.store.acts if r["started_at"] <= a["opened_at"] < r["ended_at"]]
        if r.get("arm") == "on":
            assert [a for a in w.store.acts if r["started_at"] <= a["opened_at"] < r["started_at"] + timedelta(minutes=6)]
    checks = [json.loads(r["notes"]) for r in w.store.ivs if r.get("arm") is None and "block_check" in r["notes"]]
    assert sorted(c["block_check"] for c in checks) == list(range(len(arms)))
    assert w.redis.override() is None and asyncio.run(w.redis.get(LOCK_KEY)) is None
