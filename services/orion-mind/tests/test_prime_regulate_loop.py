"""Orion Prime step 3 in the resident loop: a regulate move writes the signal where the market reads it,
opens a ``regulate`` act, and closes it from the MARKET's record after the signal ends. Fake bus, Redis,
model and market; the real config/physiology/families.yaml supplies the receptors."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

import pytest

from orion.physiology import prime_charge as pc
from test_prime_loop import T0, Bus, Redis, runtime

FAMILIES = str(Path(__file__).resolve().parents[3] / "config/physiology/families.yaml")
C = "durable.curiosity_investigate"
SLOW = {"reading": "curiosity has run five times since 06:00", "concerns": {},
        "move": {"kind": "regulate", "family": C, "direction": "slow", "why": "five runs since 06:00", "hours": 2}}


class HashRedis(Redis):
    def __init__(self, fail_write=False):
        super().__init__()
        self.h: dict[str, dict[str, str]] = {}
        self.ttl: dict[str, int] = {}
        self.fail_write = fail_write

    async def hgetall(self, key):
        return dict(self.h.get(key, {}))

    async def hset(self, key, field, value):
        if self.fail_write:
            raise ConnectionError("redis down")
        self.h.setdefault(key, {})[field] = value

    async def delete(self, key):
        self.h.pop(key, None)

    async def expire(self, key, sec):
        self.ttl[key] = sec

    def pipeline(self, transaction=True):
        return Pipe(self)


class Pipe:
    """MULTI/EXEC: queued commands apply together at execute(), or none do."""

    def __init__(self, r):
        self.r, self.ops = r, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def __getattr__(self, name):
        return lambda *a: self.ops.append((name, a))

    async def execute(self):
        if self.r.fail_write:
            raise ConnectionError("redis down")
        for name, a in self.ops:
            await getattr(self.r, name)(*a)


def rt_with(answer, *, holdback=0.0, enabled=True, fail_write=False, path=FAMILIES):
    bus = Bus()
    bus.redis = HashRedis(fail_write=fail_write)
    rt = runtime(answer=answer, bus=bus, holdback=holdback)
    rt.s.PRIME_REGULATE_ENABLED, rt.s.PRIME_FAMILIES_PATH = enabled, path
    return rt


def first_tick(rt):
    """One thought that sends the move; later thoughts (an outcome wakes Prime) answer with no move."""
    asyncio.run(rt.tick())
    rt.llm.answer = {"reading": "watching", "concerns": {}, "move": None}


def regulate_acts(rt):
    return [(k, a) for k, a in rt.bus.acts() if a["verb"] == "regulate"]


def test_a_slow_move_writes_the_signal_for_the_market_and_opens_a_regulate_act():
    rt = rt_with(SLOW)
    asyncio.run(rt.tick())
    [call] = rt.llm.calls
    assert '"kind": "regulate"' in call["system_prompt"] and f"- {C}: slow\n" in call["user_prompt"]
    [(kind, act)] = regulate_acts(rt)
    assert kind == "act_opened" and act["actor"] == "prime" and act["test_before"] == "holdback:treated"
    assert act["object_ref"] == f"physiology_family:{C}" and act["expect"]["horizon_sec"] == 2 * 3600 + pc.GRACE_SEC
    sig = pc.parse_signal(rt.bus.redis.h[pc.key(C)][pc.SIGNAL_FIELD])
    assert (sig.direction, sig.arm, sig.magnitude, sig.act_id) == ("slow", "treated", 0.5, act["act_id"])
    assert sig.expires_at - sig.opened_at == timedelta(hours=2)
    assert rt.bus.redis.ttl[pc.key(C)] == int(2 * 3600 + pc.GRACE_SEC)
    [reg] = rt.state.open_regulations
    assert reg.written is True and reg.signal_id == sig.signal_id


def test_held_back_signals_are_written_as_control_so_the_market_only_records_them():
    rt = rt_with(SLOW, holdback=1.0)
    asyncio.run(rt.tick())
    assert pc.parse_signal(rt.bus.redis.h[pc.key(C)][pc.SIGNAL_FIELD]).arm == "control"
    assert regulate_acts(rt)[0][1]["test_before"] == "holdback:control"


def test_after_the_signal_ends_the_act_closes_from_the_markets_record_and_prime_sees_why():
    rt = rt_with(SLOW)
    first_tick(rt)
    [reg] = rt.state.open_regulations
    rec = pc.merge(None, "run9", "held", T0 + timedelta(minutes=30), "value 0.31 - lane charge 0.00 did not beat "
                   "Prime's slow charge 0.54", signal_id=reg.signal_id)
    rec = pc.merge(rec, "run9", "outbid", T0 + timedelta(minutes=80), "value 0.71 - lane charge 0.00 beat Prime's "
                   "slow charge 0.54", signal_id=reg.signal_id)
    rt.bus.redis.h[pc.key(C)]["w:run9"] = pc.dumps(rec)
    # A record left by an earlier signal in the same hash never counts for this one.
    rt.bus.redis.h[pc.key(C)]["w:old"] = pc.dumps(pc.merge(None, "old", "held", T0, "x", signal_id="earlier"))
    rt.clock_box["now"] = reg.expires_at + timedelta(seconds=30)
    asyncio.run(rt.tick())
    assert [k for k, _ in regulate_acts(rt)] == ["act_opened"]      # not yet: the lag after the signal ends
    rt.clock_box["now"] = reg.expires_at + timedelta(minutes=3)
    asyncio.run(rt.tick())
    closes = [a for k, a in regulate_acts(rt) if k == "act_closed"]
    assert len(closes) == 1 and closes[0]["outcome"]["value"] == 1.0
    assert "1 of its runs bid, 1 held by my charge" in closes[0]["outcome"]["note"]
    assert rt.state.open_regulations == []
    assert any("beat Prime's slow charge 0.54" in o for o in rt.state.recent_outcomes)


def test_a_signal_no_bid_met_closes_zero_and_a_lost_record_is_left_to_the_ledger():
    rt = rt_with(SLOW)
    first_tick(rt)
    [reg] = rt.state.open_regulations
    rt.clock_box["now"] = reg.expires_at + timedelta(minutes=3)
    asyncio.run(rt.tick())
    [close] = [a for k, a in regulate_acts(rt) if k == "act_closed"]
    assert close["outcome"]["value"] == 0.0 and "no bid from it came" in close["outcome"]["note"]

    rt = rt_with(SLOW)
    first_tick(rt)
    [reg] = rt.state.open_regulations
    rt.bus.redis.h.clear()                                      # Redis lost the hash
    rt.clock_box["now"] = reg.expires_at + timedelta(minutes=3)
    asyncio.run(rt.tick())
    assert not [a for k, a in regulate_acts(rt) if k == "act_closed"] and rt.state.open_regulations == []
    assert any("could not be graded" in o for o in rt.state.recent_outcomes)


def test_a_failed_write_closes_the_act_zero_and_never_wakes_prime_again():
    rt = rt_with(SLOW, fail_write=True)
    asyncio.run(rt.tick())
    [reg] = rt.state.open_regulations
    assert reg.written is False and "ConnectionError" in reg.write_error
    assert not rt.bus.redis.h            # all-or-nothing: nothing of the signal is in force
    rt.clock_box["now"] = T0 + timedelta(seconds=15)
    asyncio.run(rt.tick())
    [close] = [a for k, a in regulate_acts(rt) if k == "act_closed"]
    assert close["outcome"]["value"] == 0.0 and "not delivered" in close["outcome"]["note"]
    assert rt.state.pending_changes == [] and any("was not delivered" in o for o in rt.state.recent_outcomes)


def test_off_the_move_is_not_offered_and_an_answer_using_it_is_refused_visibly():
    rt = rt_with(SLOW, enabled=False)
    asyncio.run(rt.tick())
    [call] = rt.llm.calls
    assert '"kind": "regulate"' not in call["system_prompt"] and "ORGANS YOU CAN SIGNAL" not in call["user_prompt"]
    assert not regulate_acts(rt) and not rt.bus.redis.h
    assert any("regulation is off" in o for o in rt.state.recent_outcomes)


def test_a_signal_to_an_organ_without_receptors_is_shown_to_prime_never_dropped():
    move = {**SLOW, "move": {**SLOW["move"], "family": "durable.self_inquiry"}}
    rt = rt_with(move)
    asyncio.run(rt.tick())
    assert not regulate_acts(rt) and not rt.bus.redis.h
    assert any("durable.self_inquiry declares no receptors" in o for o in rt.state.recent_outcomes)


def test_an_unreadable_families_config_is_a_gap_in_the_frame_not_a_crash():
    rt = rt_with(SLOW, path="/nonexistent/families.yaml")
    asyncio.run(rt.tick())
    [call] = rt.llm.calls
    assert "unavailable: families config unreadable" in call["user_prompt"]
    assert not regulate_acts(rt)


def test_after_a_signal_the_organ_rests_as_long_as_it_lasted_then_prime_may_signal_again():
    rt = rt_with(SLOW)
    first_tick(rt)
    [reg] = rt.state.open_regulations
    rt.clock_box["now"] = reg.expires_at + timedelta(minutes=3)
    asyncio.run(rt.tick())                                   # graded; Prime thinks again (no move)
    assert rt.state.regulation_rest_until[C] == reg.expires_at + timedelta(hours=2)
    rt.llm.answer = SLOW
    rt.state = rt.state.model_copy(update={"pending_changes": ["something moved"], "last_thought_at": None})
    asyncio.run(rt.tick())
    assert len([a for k, a in regulate_acts(rt) if k == "act_opened"]) == 1
    assert any("rests from my signals" in o for o in rt.state.recent_outcomes)
    rt.clock_box["now"] = reg.expires_at + timedelta(hours=2, minutes=1)
    rt.state = rt.state.model_copy(update={"pending_changes": ["something moved"], "last_thought_at": None})
    asyncio.run(rt.tick())
    assert len([a for k, a in regulate_acts(rt) if k == "act_opened"]) == 2 and C not in rt.state.regulation_rest_until


def test_with_the_markets_switch_off_prime_is_told_so_not_that_no_bid_came():
    rt = rt_with(SLOW)
    first_tick(rt)
    [reg] = rt.state.open_regulations
    base = rt.get_json

    def get_json(url):
        if url.endswith("/health"):
            return {"ok": True, "physiology_market": {"prime_regulation": False}}
        return base(url)

    rt.get_json = get_json
    rt.clock_box["now"] = reg.expires_at + timedelta(minutes=3)
    asyncio.run(rt.tick())
    [close] = [a for k, a in regulate_acts(rt) if k == "act_closed"]
    assert close["outcome"]["value"] == 0.0 and "switched off" in close["outcome"]["note"]


def test_a_malformed_hours_never_loses_the_thought():
    move = {**SLOW, "move": {**SLOW["move"], "hours": "three"}}
    rt = rt_with(move)
    asyncio.run(rt.tick())
    [reg] = rt.state.open_regulations
    assert (reg.expires_at - reg.opened_at) == timedelta(hours=pc.DEFAULT_HOURS)
    assert rt.state.last_reading.startswith("curiosity has run")


def test_real_redis_client_semantics_with_fakeredis():
    """The same write/read path against redis-py's own asyncio client (fakeredis): MULTI pipeline, TTL,
    bytes keys and values from HGETALL. Skipped where fakeredis is not installed."""
    fakeredis = pytest.importorskip("fakeredis")
    bus = Bus()
    bus.redis = fakeredis.aioredis.FakeRedis()
    rt = runtime(answer=SLOW, bus=bus)
    rt.s.PRIME_REGULATE_ENABLED, rt.s.PRIME_FAMILIES_PATH = True, FAMILIES
    rt.llm.answer = SLOW

    async def flow():
        await rt.tick()
        rt.llm.answer = {"reading": "watching", "concerns": {}, "move": None}
        [reg] = rt.state.open_regulations
        ttl = await bus.redis.ttl(pc.key(C))
        rec = pc.merge(None, "run1", "held", T0, "did not beat Prime's slow charge 0.54", signal_id=reg.signal_id)
        await bus.redis.hset(pc.key(C), "w:run1", pc.dumps(rec))
        rt.clock_box["now"] = reg.expires_at + timedelta(minutes=3)
        await rt.tick()
        return reg, ttl

    reg, ttl = asyncio.run(flow())
    assert reg.written is True and 0 < ttl <= int(2 * 3600 + pc.GRACE_SEC)
    [close] = [a for k, a in regulate_acts(rt) if k == "act_closed"]
    assert close["outcome"]["value"] == 1.0
