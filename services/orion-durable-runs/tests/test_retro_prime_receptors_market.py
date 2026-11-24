"""Game company patch 4: Prime's slow/quicken reach Orion's eyes and voice in a person's game.

Acceptance (spec 2026-11-16, "Acceptance checks: Patch 4"): "A `slow` signal from Prime changes the look count in the
next hour, and the market records the effect." hub.retro_watch runs a look only on a `funded` answer
(orion/physiology/binding.py), so the look count of an hour is the count of funded look bids in it. This replays one
hour of look bids (one a minute, the look ceiling) through the real market, with and without a slow charge.
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import timedelta
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SERVICE_ROOT.parents[1]), str(SERVICE_ROOT)]

from orion.physiology import prime_charge as pc  # noqa: E402
from orion.schemas.physiology import PhysiologyConstraintReadingV1 as R  # noqa: E402
from orion.schemas.physiology_want import WantV1  # noqa: E402
from tests.test_physiology_bid import NOW, _state  # noqa: E402
from tests.test_prime_charge_market import Redis, effects, market, put, signal  # noqa: E402

W = "hub.retro_watch"
L = "hub.retro_commentary"


def _chat_state(at=None):
    base = _state(at)
    return base.model_copy(update={"constraints": list(base.constraints) + [
        R(constraint="lane:chat", unit="slot_s", level=0.0, budget=3600.0, budget_kind="capacity",
          detail={"slots": 1, "window_sec": 3600.0},
          price={"value": 0.0, "ceiling": 1.0, "bind_state": "slack", "horizon_sec": 3600.0, "why": ""})]})


def _look_acts():
    """A week of looks closing at uneven gaps (1 to 6 min apart), so a look want's rank follows its hunger."""
    out, t = [], NOW - timedelta(minutes=1)
    for i in range(240):
        gap = [1.0, 2.0, 1.5, 6.0, 1.0, 3.0][i % 6]
        out.append({"act_id": f"l{i}", "actor": W, "opened_at": t - timedelta(seconds=20), "closed_at": t,
                    "outcome_value": 1.0, "outcome_source": "producer", "signal_id": "artifact_produced",
                    "cost_json": json.dumps({"seconds": 6.0, "gpu_slot_s_by_lane": {"chat": 6.0}})})
        t -= timedelta(minutes=gap)
    return out


def _look(i, intensity):
    return WantV1(want_id=f"look{i}", family=W, producer="orion-hub.retro_watch.binding", verb="look", tier=1,
                  object_ref="retro_sitting:s1", intensity=intensity)


def _hour(redis):
    """One hour of look bids, one a minute, hunger rising between looks like a player who keeps moving."""
    m = market(redis, extra_acts=_look_acts())
    funded = 0
    for i in range(60):
        at = NOW + timedelta(minutes=i)
        redis.state_raw = _chat_state(at).model_dump_json()      # the body state is fresh at every bid
        d = asyncio.run(m.bid(_look(i, intensity=0.1 + 0.15 * (i % 6)), at))
        funded += d.outcome == "funded"
    return funded


def test_a_slow_signal_lowers_the_look_count_of_the_next_hour_and_the_market_records_it():
    plain = Redis(_chat_state().model_dump_json())
    slowed = Redis(_chat_state().model_dump_json())
    put(slowed, signal(family=W, start=-0.1, end=1.5))
    looks_plain, looks_slowed = _hour(plain), _hour(slowed)
    assert looks_slowed < looks_plain, (looks_plain, looks_slowed)
    rec = effects(slowed, W)
    held = [e for e in rec.values() if e.effect == "held"]
    # Every held look is on Prime's record with the signal, and a held look is exactly a look that did not happen.
    assert held and all(e.signal_id == "sig1" for e in held)
    assert len(held) == looks_plain - looks_slowed
    # Never a stop (spec section 6 part 4): the hungriest looks still beat the charge.
    assert looks_slowed > 0
    assert effects(plain, W) == {}


def test_both_game_families_receive_slow_and_quicken():
    for fam, producer in ((W, "orion-hub.retro_watch.binding"), (L, "orion-hub.retro_commentary.binding")):
        for direction in ("slow", "quicken"):
            r = Redis(_chat_state().model_dump_json())
            put(r, signal(family=fam, direction=direction))
            want = WantV1(want_id=f"{fam}-{direction}", family=fam, producer=producer, verb="say", tier=1,
                          intensity=0.6)
            asyncio.run(market(r, extra_acts=_look_acts()).bid(want, NOW))
            assert r.reads >= 1, (fam, direction)
            e = effects(r, fam)
            assert set(e) == {want.want_id}, (fam, direction, e)


def test_a_shadow_look_bid_is_never_charged():
    r = Redis(_chat_state().model_dump_json())
    put(r, signal(family=W))
    want = _look(0, 0.1).model_copy(update={"producer": "orion-hub.retro_watch"})
    d = asyncio.run(market(r, extra_acts=_look_acts()).bid(want, NOW))
    assert "Prime" not in (d.why or "") and effects(r, W) == {} and r.reads == 0


def test_the_signal_key_is_the_one_prime_writes():
    assert pc.key(W) == "orion:prime:regulate:hub.retro_watch"
