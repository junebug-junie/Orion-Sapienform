"""Orion Prime step 3: the market reads Prime's regulate signal, applies it (treated) or only records it
(control), writes what it did to each bid back for Prime, and never lets a Redis problem fail a bid.

Fakes for Redis / Postgres / the bus (shared with test_physiology_bid.py)."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import timedelta
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SERVICE_ROOT.parents[1]), str(SERVICE_ROOT)]

from app.physiology_market import Market  # noqa: E402
from app.run_bids import bind_verdict  # noqa: E402
from orion.physiology import prime_charge as pc  # noqa: E402
from orion.schemas.physiology_want import WantV1  # noqa: E402
from tests.test_physiology_bid import FAMS, NOW, _Pool, _state  # noqa: E402

C = "durable.curiosity_investigate"


def _curiosity_acts():
    """Closes at uneven gaps, the last 10 h ago: a want's value rank follows its intensity."""
    out, t = [], NOW - timedelta(hours=10)
    for i, g in enumerate([0.2, 0.5, 1.0, 2.0, 3.0, 0.3, 0.7, 1.5, 4.0, 0.4, 2.5, 0.8]):
        out.append({"act_id": f"c{i}", "actor": C, "opened_at": t - timedelta(minutes=1), "closed_at": t,
                    "outcome_value": 1.0, "outcome_source": "producer", "signal_id": None,
                    "cost_json": json.dumps({"seconds": 1000.0, "gpu_slot_s_by_lane": {"agent": 1800.0}})})
        t -= timedelta(hours=g)
    return out


class Redis:
    def __init__(self, state_raw, *, fail=False):
        self.state_raw = state_raw
        self.h: dict[str, dict[str, str]] = {}
        self.ttl: dict[str, int] = {}
        self.fail = fail
        self.reads = 0

    async def get(self, key):
        return self.state_raw

    def _check(self):
        if self.fail:
            raise ConnectionError("redis down")

    async def hget(self, key, field):
        self.reads += 1
        self._check()
        return self.h.get(key, {}).get(field)

    async def hset(self, key, field, value):
        self._check()
        self.h.setdefault(key, {})[field] = value

    async def hlen(self, key):
        return len(self.h.get(key, {}))

    async def expire(self, key, sec):
        self.ttl[key] = sec


def _other_acts(n, family="durable.self_inquiry"):
    """``n`` acts of another Tier 2 family started yesterday (the floor counts acts started, patch 3b)."""
    at = NOW - timedelta(days=1)
    return [{"act_id": f"o{i}", "actor": family, "opened_at": at, "closed_at": at + timedelta(minutes=5),
             "outcome_value": 1.0, "outcome_source": "producer", "signal_id": None,
             "cost_json": json.dumps({"seconds": 300.0})} for i in range(n)]


# By default curiosity's own acts are all of Tier 2's acts started this week: well above the 5% floor.
def market(redis, *, on=True, extra_acts=()):
    async def publish(d):
        return None

    return Market(families=FAMS, families_error=None,
                  pool_getter=lambda: _Pool(acts=_curiosity_acts() + list(extra_acts)),
                  redis_getter=lambda: redis, publish=publish, prime_regulation=on)


def signal(direction="slow", arm="treated", family=C, *, start=-1.0, end=2.0):
    return pc.PrimeRegulationV1(signal_id="sig1", act_id="act_r", family=family, direction=direction, arm=arm,
                                magnitude=0.5, why="curiosity ran five times this morning",
                                opened_at=NOW + timedelta(hours=start), expires_at=NOW + timedelta(hours=end))


def put(redis, sig):
    redis.h.setdefault(pc.key(sig.family), {})[pc.SIGNAL_FIELD] = pc.dumps(sig)


def cwant(i=0, intensity=0.25, family=C):
    return WantV1(want_id=f"run{i}", family=family, producer="orion-durable-runs.admission.binding", verb="infer",
                  tier=2, intensity=intensity)


def effects(redis, family=C):
    return {e.want_id: e for e in pc.parse_effects(redis.h.get(pc.key(family), {}))}


def test_a_treated_slow_signal_holds_a_low_value_run_and_records_why_for_prime():
    r = Redis(_state().model_dump_json())
    put(r, signal())
    m = market(r)
    d = asyncio.run(m.bid(cwant(), NOW))
    assert d.outcome == "deferred" and d.binding_constraint == "prime:slow" and d.net <= 0
    # Patch D obeys it: a price deferral holds the run and re-bids at the default (hunger keeps growing).
    v = bind_verdict(d, NOW, NOW + timedelta(hours=6))
    assert not v.admit and v.why == "deferred:price" and v.retry_at == NOW + timedelta(seconds=600)
    assert "did not beat Prime's slow charge" in v.detail["why"]
    e = effects(r)["run0"]
    assert (e.effect, e.bids, e.signal_id) == ("held", 1, "sig1") and "Prime's slow charge" in e.why
    assert r.ttl[pc.key(C)] == pytest.approx(2 * 3600 + pc.GRACE_SEC, abs=2)
    # The same run bids again, hungrier: it outbids the charge, and stays recorded as held once.
    d2 = asyncio.run(m.bid(cwant(intensity=1.0), NOW + timedelta(minutes=2)))
    assert d2.outcome == "funded" and "beat Prime's slow charge" in d2.why
    e = effects(r)["run0"]
    assert (e.effect, e.last, e.bids) == ("held", "outbid", 2)
    assert m.health(NOW)["prime_applied"] == 2


def test_a_control_signal_is_never_applied_but_records_what_it_would_have_done():
    r = Redis(_state().model_dump_json())
    put(r, signal(arm="control"))
    d = asyncio.run(market(r).bid(cwant(), NOW))
    assert d.outcome == "funded" and "Prime" not in d.why          # not applied
    e = effects(r)["run0"]
    assert e.effect == "held" and e.why.startswith("held back (control, not applied)")
    assert "did not beat Prime's slow charge" in e.why          # the charged decision's own words


@pytest.mark.parametrize("arm", ["treated", "control"])
def test_both_arms_class_an_exploration_floor_fund_the_same_way(arm):
    """Review finding: control once read the floor off the UNcharged decision and called this 'held'."""
    # Curiosity's 12 acts against 300 of another family: 3.8% of Tier 2's acts started, under the floor.
    r = Redis(_state().model_dump_json())
    put(r, signal(arm=arm))
    asyncio.run(market(r, extra_acts=_other_acts(300)).bid(cwant(), NOW))
    assert effects(r)["run0"].effect == "floor"


def test_off_expired_not_received_or_foreign_signals_are_never_read_as_charges():
    for sig, on in ((signal(), False), (signal(start=-5.0, end=-1.0), True), (signal(start=1.0, end=3.0), True)):
        r = Redis(_state().model_dump_json())
        put(r, sig)
        d = asyncio.run(market(r, on=on).bid(cwant(), NOW))
        assert d.outcome == "funded" and "Prime" not in d.why and not effects(r)
        assert (r.reads == 0) is (not on)            # off: the signal is never even read
    # A family without receptors: its key is never even read.
    r = Redis(_state().model_dump_json())
    put(r, signal(family="durable.self_inquiry"))
    asyncio.run(market(r).bid(cwant(family="durable.self_inquiry"), NOW))
    assert r.reads == 0 and not effects(r, "durable.self_inquiry")
    # A signal whose stored family does not match the key is ignored.
    r = Redis(_state().model_dump_json())
    r.h[pc.key(C)] = {pc.SIGNAL_FIELD: pc.dumps(signal(family="durable.self_inquiry"))}
    d = asyncio.run(market(r).bid(cwant(), NOW))
    assert "Prime" not in d.why


def test_a_redis_failure_never_fails_a_bid_and_never_charges():
    r = Redis(_state().model_dump_json(), fail=True)
    m = market(r)
    d = asyncio.run(m.bid(cwant(), NOW))
    assert d.outcome == "funded" and m.health(NOW)["prime_read_failed"] == 1


def test_real_redis_client_semantics_with_fakeredis():
    """hget/hset/hlen/expire against redis-py's own asyncio client (fakeredis), bytes in and out. Skipped
    where fakeredis is not installed."""
    fakeredis = pytest.importorskip("fakeredis")
    from orion.schemas.physiology import PHYSIOLOGY_STATE_REDIS_KEY

    r = fakeredis.aioredis.FakeRedis()
    m = market(r)

    async def flow():
        await r.set(PHYSIOLOGY_STATE_REDIS_KEY, _state().model_dump_json())
        await r.hset(pc.key(C), pc.SIGNAL_FIELD, pc.dumps(signal()))
        d = await m.bid(cwant(), NOW)
        return d, await r.hgetall(pc.key(C)), await r.ttl(pc.key(C))

    d, fields, ttl = asyncio.run(flow())
    assert d.binding_constraint == "prime:slow"
    [e] = pc.parse_effects(fields, signal_id="sig1")
    assert e.effect == "held" and 0 < ttl <= int(2 * 3600 + pc.GRACE_SEC) + 1


def test_the_market_charges_a_dream_bid_once_dream_declares_slow_and_refuses_quicken():
    """Orion Prime step 4: dream.cycle declares `slow` in families.yaml, so the market reads Prime's signal for its
    bids (sent over HTTP by orion-dream) like curiosity's; a quicken signal to dream is not received."""
    from orion.schemas.physiology import PhysiologyConstraintReadingV1 as R

    D = "dream.cycle"
    base = _state()
    st = base.model_copy(update={"constraints": list(base.constraints) + [
        R(constraint="lane:fast", unit="slot_s", level=0.0, budget=3600.0, budget_kind="capacity",
          detail={"slots": 1, "window_sec": 3600.0},
          price={"value": 0.0, "ceiling": 1.0, "bind_state": "slack", "horizon_sec": 3600.0, "why": ""})]})
    r = Redis(st.model_dump_json())
    put(r, signal(family=D))
    m = market(r)
    want = WantV1(want_id="sleep1", family=D, producer="orion-dream.dream_cycle.binding", verb="make", tier=1,
                  intensity=0.25)
    d = asyncio.run(m.bid(want, NOW))
    assert "Prime's slow charge" in d.why and d.charge is not None and d.charge >= 0.5 - 1e-9
    assert effects(r, D)["sleep1"].signal_id == "sig1"
    r2 = Redis(st.model_dump_json())
    put(r2, signal(direction="quicken", family=D))
    d2 = asyncio.run(market(r2).bid(want.model_copy(update={"want_id": "sleep2"}), NOW))
    assert "Prime" not in d2.why and effects(r2, D) == {}


def test_a_shadow_bid_of_a_receiving_family_is_never_charged_or_recorded():
    """Rollback to shadow (or a forced dream sleep's shadow bid): the charge would change nothing that is obeyed, so
    the market neither applies it nor records an effect Prime would be graded on."""
    from orion.physiology.binding import BINDING_PRODUCERS

    r = Redis(_state().model_dump_json())
    put(r, signal())
    shadow = WantV1(want_id="run9", family=C, producer="orion-durable-runs.admission", verb="infer", tier=2,
                    intensity=0.25)
    assert shadow.producer not in BINDING_PRODUCERS
    d = asyncio.run(market(r).bid(shadow, NOW))
    assert "Prime" not in d.why and effects(r) == {} and r.reads == 0


# --- the organ's refusal (curiosity replica patch 4, docs/superpowers/specs/2026-11-04-curiosity-replica-design.md
# section 9): written next to the signal, read from the next bid, never applied on either arm. ---

def refuse(redis, sig, *, signal_id=None, reason="curiosity was idle all morning; a slower pace would starve it"):
    ref = pc.PrimeChargeRefusalV1(signal_id=signal_id or sig.signal_id, act_id="act_answer", by="curiosity.replica",
                                  reason=reason, at=NOW - timedelta(minutes=5))
    redis.h.setdefault(pc.key(sig.family), {})[pc.REFUSAL_FIELD] = pc.dumps(ref)


def test_a_refused_treated_signal_is_not_applied_and_the_decision_says_who_refused_it():
    r = Redis(_state().model_dump_json())
    sig = signal()
    put(r, sig)
    refuse(r, sig)
    m = market(r)
    d = asyncio.run(m.bid(cwant(), NOW))
    # Without the refusal this bid is held (test above); refused, it goes ahead and says why.
    assert d.outcome == "funded" and d.binding_constraint is None and "prime:slow" not in (d.why or "").split(";")[0]
    assert "Prime's slow charge refused by curiosity.replica: curiosity was idle all morning" in d.why
    assert "would have been 0." in d.why
    e = effects(r)["run0"]
    # What really happened: the charge changed nothing. What it would have done is in the why.
    assert e.effect == "unchanged" and e.why.startswith("refused by curiosity.replica:")
    assert "without the refusal it would have been held" in e.why
    assert m.health(NOW)["prime_refused"] == 1 and m.health(NOW)["prime_applied"] == 0
    # Graded on what happened: the regulate act reads 0 (nothing changed).
    assert not pc.changed_any(pc.parse_effects(r.h[pc.key(C)], signal_id="sig1"))


def test_a_refusal_counts_from_the_next_bid_so_earlier_holds_stay_on_record():
    r = Redis(_state().model_dump_json())
    sig = signal()
    put(r, sig)
    m = market(r)
    assert asyncio.run(m.bid(cwant(), NOW)).outcome == "deferred"
    refuse(r, sig)
    d = asyncio.run(m.bid(cwant(), NOW + timedelta(minutes=2)))
    assert d.outcome == "funded"
    e = effects(r)["run0"]
    assert (e.effect, e.last, e.bids) == ("held", "unchanged", 2)   # the hold really happened once


def test_a_refused_control_signal_records_unchanged_on_the_same_rule():
    r = Redis(_state().model_dump_json())
    sig = signal(arm="control")
    put(r, sig)
    refuse(r, sig)
    d = asyncio.run(market(r).bid(cwant(), NOW))
    assert d.outcome == "funded" and "Prime" not in d.why           # a control signal never shows in a decision
    e = effects(r)["run0"]
    assert e.effect == "unchanged" and "a control signal" in e.why and "would have been held" in e.why


def test_a_refusal_of_an_earlier_signal_or_an_unreadable_one_never_lifts_the_charge():
    for write in (lambda r, s: refuse(r, s, signal_id="older_signal"),
                  lambda r, s: r.h.setdefault(pc.key(C), {}).__setitem__(pc.REFUSAL_FIELD, "{not json")):
        r = Redis(_state().model_dump_json())
        sig = signal()
        put(r, sig)
        write(r, sig)
        d = asyncio.run(market(r).bid(cwant(), NOW))
        assert d.outcome == "deferred" and d.binding_constraint == "prime:slow" and "refused" not in d.why
        assert effects(r)["run0"].effect == "held"
