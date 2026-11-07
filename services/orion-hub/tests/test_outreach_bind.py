"""Body economy Patch D for hub.outreach: Hub outreach OBEYS its bid (scripts/outreach_bind.py).

Mirrors #3010 (curiosity investigate) and the dream cycle: funded and lane-queued wants compose; price,
sated and net <= 0 deferrals hold and re-bid at the market's relief time; the 1 h window ends a hold;
no answer falls back to today's gates; forced sends are never held; one bid per hold window.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.schemas.physiology_want import WantDecisionV1, WantV1

# conftest's _hub_service_isolation drops every ``scripts.*`` module before each test, so hub
# modules are imported inside fixtures and tests.

T0 = datetime(2026, 11, 7, 18, 0, tzinfo=timezone.utc)


def _decision(want: WantV1, outcome: str, reason: str, *, net: float = 0.3, retry: float | None = None,
              binding: str | None = None) -> WantDecisionV1:
    now = datetime.now(timezone.utc)
    return WantDecisionV1(
        decision_id=f"{want.want_id}@{now.isoformat()}", want_id=want.want_id, family=want.family,
        producer=want.producer, verb=want.verb, tier=want.tier, decided_at=now, outcome=outcome, reason=reason,
        net=net, charge=0.1, retry_after_sec=retry, binding_constraint=binding, want=want,
        why=f"hub.outreach: test {reason}",
    )


class _Market:
    """``answers``: a list of (outcome, reason, net, retry) consumed per bid; the last one repeats.
    ``None`` -> no answer; ``"raise"`` -> the client raised; ``"slow"`` -> never answers in time."""

    def __init__(self):
        self.answers: list = [("funded", "net_positive", 0.3, None)]
        self.wants: list[WantV1] = []
        self.released: list[str] = []

    async def abid(self, want, timeout_sec):
        self.wants.append(want)
        a = self.answers[0] if len(self.answers) == 1 else self.answers.pop(0)
        if a is None:
            return None
        if a == "raise":
            raise RuntimeError("market down")
        if a == "slow":
            await asyncio.sleep(10)
            return None
        outcome, reason, net, retry = a
        return _decision(want, outcome, reason, net=net, retry=retry,
                         binding="lane:agent" if reason == "capacity_reserved" else None)

    async def arelease(self, want_id):
        self.released.append(want_id)
        return 1


@pytest.fixture
def market(monkeypatch):
    import scripts.outreach_bind as ob

    m = _Market()
    monkeypatch.setattr(ob, "_abid", m.abid)
    monkeypatch.setattr(ob, "_arelease", m.arelease)
    from scripts.settings import settings

    monkeypatch.setattr(settings, "ORION_PHYSIOLOGY_BID_HUB_OUTREACH", "on", raising=False)
    m.ob = ob
    return m


async def _drain():
    for _ in range(5):
        await asyncio.sleep(0)


# --- the contract -----------------------------------------------------------------------------


def test_window_matches_families_yaml_and_producer_is_registered_binding() -> None:
    import yaml

    from orion.physiology.binding import BINDING_PRODUCERS

    from scripts import outreach_bind as ob

    root = Path(__file__).resolve().parents[3]
    fams = yaml.safe_load((root / "config/physiology/families.yaml").read_text())
    entry = fams["families"]["hub.outreach"] if "families" in fams else fams["hub.outreach"]
    assert float(entry["deferrable_window_s"]) == ob.DEFERRABLE_WINDOW_S
    assert int(entry["tier"]) == ob.TIER
    assert ob.BINDING_PRODUCER in BINDING_PRODUCERS


def test_flag_ships_on_and_unknown_is_shadow_with_a_warning(caplog) -> None:
    from app.settings import Settings

    from scripts import outreach_bind as ob

    assert Settings.model_fields["ORION_PHYSIOLOGY_BID_HUB_OUTREACH"].default == "on"
    root = Path(__file__).resolve().parents[1]
    assert "ORION_PHYSIOLOGY_BID_HUB_OUTREACH=on" in (root / ".env_example").read_text().splitlines()
    assert ob.bind_mode("on") == "on" and ob.bind_mode(" SHADOW ") == "shadow"
    caplog.set_level(logging.WARNING, logger="orion-hub.outreach_bind")
    assert ob.bind_mode("binding") == "shadow"
    assert ob.bind_mode("") == "shadow"
    assert any("physiology_bind_unknown_mode" in r.getMessage() for r in caplog.records)


def test_unknown_flag_on_the_outreach_loop_means_shadow_and_does_not_crash(monkeypatch) -> None:
    from scripts.settings import settings

    monkeypatch.setattr(settings, "ORION_PHYSIOLOGY_BID_HUB_OUTREACH", "yes please", raising=False)
    o = _outreach()
    assert o.economy_mode == "shadow" and o.economy_binding is False


# --- the binder (pure rule + state) ----------------------------------------------------------


@pytest.mark.asyncio
async def test_funded_proceeds_and_the_want_says_who_it_is(market) -> None:
    b = market.ob.OutreachBinder()
    v = await b.bind("endogenous_tick", door="endogenous_tick", now=T0, cap_counted=True)
    assert v.admit and v.why == "funded"
    (want,) = market.wants
    assert (want.family, want.producer, want.verb, want.tier, want.object_ref) == (
        "hub.outreach", "orion-hub.outreach.binding", "say", 2, "juniper")
    assert want.cost is None                      # cap-counted: the family's 1 message is charged
    assert want.window.latest == T0 + timedelta(seconds=3600)


@pytest.mark.asyncio
async def test_cap_exempt_door_wants_cost_no_juniper_message(market) -> None:
    b = market.ob.OutreachBinder()
    await b.bind("reading:k", door="reading", now=T0, cap_counted=False)
    (want,) = market.wants
    assert want.cost is not None and want.cost.juniper_msgs == 0


@pytest.mark.asyncio
async def test_capacity_deferral_worth_its_price_proceeds_and_reserves_nothing(market) -> None:
    market.answers = [("deferred", "capacity_reserved", 0.3, 200.0)]
    b = market.ob.OutreachBinder()
    v = await b.bind("k", door="d", now=T0)
    await _drain()
    assert v.admit and v.why == "queued_at_pool"
    assert market.released == [market.wants[0].want_id]


@pytest.mark.parametrize("reason,net", [("price", -0.1), ("sated", 0.0), ("capacity_reserved", 0.0)])
@pytest.mark.asyncio
async def test_price_sated_or_worthless_deferral_holds_then_rebids(market, reason, net) -> None:
    market.answers = [("deferred", reason, net, 30.0), ("funded", "net_positive", 0.3, None)]
    b = market.ob.OutreachBinder()
    v = await b.bind("k", door="d", now=T0)
    assert not v.admit and v.why == f"deferred:{reason}"
    # retry_after 30 s is clamped up to one physiology tick (120 s)
    assert v.retry_at == T0 + timedelta(seconds=120)
    # Every check before the relief time holds WITHOUT a bid.
    for s in range(10, 120, 10):
        vv = await b.bind("k", door="d", now=T0 + timedelta(seconds=s))
        assert not vv.admit and vv.why == f"waiting:{reason}"
    assert len(market.wants) == 1
    v2 = await b.bind("k", door="d", now=T0 + timedelta(seconds=120))
    assert v2.admit and v2.why == "funded" and len(market.wants) == 2
    # Same want across the hold: one want_id.
    assert len({w.want_id for w in market.wants}) == 1
    rec = b.record("k", v2)
    assert rec["holds"] == 1 and rec["held_sec"] == 120.0 and rec["verdict"] == "funded"


@pytest.mark.asyncio
async def test_window_end_proceeds_without_dropping_the_want(market) -> None:
    market.answers = [("deferred", "price", -0.2, 3600.0)]
    b = market.ob.OutreachBinder()
    assert not (await b.bind("k", door="d", now=T0)).admit
    v = await b.bind("k", door="d", now=T0 + timedelta(seconds=3600))
    assert v.admit and v.why == "window_end"
    assert len(market.wants) == 1  # the window's end goes without asking again


@pytest.mark.parametrize("answer,why", [(None, "no_answer:no_decision"), ("raise", "no_answer:no_decision"),
                                        ("slow", "no_answer:no_decision")])
@pytest.mark.asyncio
async def test_broken_slow_or_absent_market_falls_back_to_the_old_gates(market, answer, why) -> None:
    market.answers = [answer]
    b = market.ob.OutreachBinder(timeout_sec=0.05)
    v = await b.bind("k", door="d", now=T0)
    assert v.admit and v.why == why


@pytest.mark.asyncio
async def test_market_unavailable_reason_is_no_answer_not_a_hold(market) -> None:
    market.answers = [("deferred", "physiology_unavailable", None, 120.0)]
    v = await market.ob.OutreachBinder().bind("k", door="d", now=T0)
    assert v.admit and v.why == "no_answer:physiology_unavailable"


@pytest.mark.asyncio
async def test_forced_is_never_held_and_never_bids(market) -> None:
    market.answers = [("deferred", "price", -1.0, 600.0)]
    v = await market.ob.OutreachBinder().bind("k", door="d", now=T0, forced=True)
    assert v.admit and v.why == "forced" and market.wants == []


@pytest.mark.asyncio
async def test_two_checks_racing_on_one_want_compose_once(market) -> None:
    gate = asyncio.Event()

    async def slow_funded(want, timeout_sec):
        market.wants.append(want)
        await gate.wait()
        return _decision(want, "funded", "net_positive")

    market.ob._abid = slow_funded
    b = market.ob.OutreachBinder()
    first = asyncio.create_task(b.bind("k", door="d", now=T0))
    second = asyncio.create_task(b.bind("k", door="d", now=T0))
    await _drain()
    gate.set()
    v1, v2 = await first, await second
    assert sorted([v1.admit, v2.admit]) == [False, True]
    assert {v1.why, v2.why} == {"funded", "duplicate"}
    assert len(market.wants) == 1
    # The attempt ends: the next check opens a NEW want.
    await b.close("k", sent=True)
    v3 = await b.bind("k", door="d", now=T0 + timedelta(seconds=5))
    assert v3.admit and v3.why == "funded"   # answered by a fresh want
    assert len({w.want_id for w in market.wants}) == 2


@pytest.mark.asyncio
async def test_close_releases_a_funded_reservation_sent_or_not(market) -> None:
    b = market.ob.OutreachBinder()
    await b.bind("a", door="d", now=T0)
    want_a = market.wants[-1].want_id
    await b.close("a", sent=True)
    await b.bind("b", door="d", now=T0)
    want_b = market.wants[-1].want_id
    await b.close("b", sent=False)
    assert market.released == [want_a, want_b]
    # A close naming another want closes nothing.
    await b.bind("c", door="d", now=T0)
    await b.close("c", sent=True, want_id="someone-else")
    assert "c" in b.wants


@pytest.mark.asyncio
async def test_a_lapsed_hold_bids_afresh_instead_of_a_free_window_end(market) -> None:
    """Review finding 1: held at 09:00, nobody re-checked (no content, quiet hours); a compose at 14:00
    must bid again with a new want and a new window, not pass as ``window_end`` with a 5 h hold."""
    market.answers = [("deferred", "price", -0.1, 600.0), ("funded", "net_positive", 0.3, None)]
    b = market.ob.OutreachBinder()
    assert not (await b.bind("k", door="d", now=T0)).admit
    v = await b.bind("k", door="d", now=T0 + timedelta(hours=5))
    assert v.admit and v.why == "funded" and len(market.wants) == 2
    assert market.wants[0].want_id != market.wants[1].want_id
    assert b.record("k", v)["held_sec"] == 0.0


@pytest.mark.asyncio
async def test_a_timed_out_bid_is_released(market) -> None:
    """Review finding 4: the HTTP call outlives the timeout and may still reserve; release it."""
    market.answers = ["slow"]
    b = market.ob.OutreachBinder(timeout_sec=0.05)
    v = await b.bind("k", door="d", now=T0)
    await _drain()
    assert v.admit and market.released == [market.wants[0].want_id]


# --- the background tick ---------------------------------------------------------------------


def _outreach(**overrides):
    from scripts.endogenous_outreach import EndogenousOutreach

    base = dict(enabled=True, tick_interval_sec=10.0, min_cooldown_sec=0.0, daily_cap=4, quiet_start_hour=-1,
                quiet_end_hour=-1, timeout_sec=5.0, agent_lane_timeout_sec=5.0,
                notify_channel="orion:notify:in_app", fallback_session_id="orion_outreach",
                trigger_evaluator=lambda: object())
    base.update(overrides)
    return EndogenousOutreach(**base)


def _stub_tick(monkeypatch, generated: list, rows: list) -> None:
    from scripts import act_emit
    from scripts import endogenous_outreach_decisions as dec
    from scripts.endogenous_outreach import EndogenousOutreach, OutreachContext

    async def fake_gather(self, session_id, open_prior_previews=None, open_prior_ids=None):
        return OutreachContext(curiosity_summaries=["sustained prediction error on node:x"], recent_turns=[],
                               presence=None)

    async def fake_generate(self, prompt, session_id, correlation_id, **_kw):
        generated.append(correlation_id)
        return "something worth saying", {"stub": True}

    async def fake_act(bus, **kw):
        return True

    async def noop(self, **kw):
        return None

    monkeypatch.setattr(EndogenousOutreach, "_gather_context", fake_gather)
    monkeypatch.setattr(EndogenousOutreach, "_generate", fake_generate)
    monkeypatch.setattr(EndogenousOutreach, "_publish_history", noop)
    monkeypatch.setattr(EndogenousOutreach, "_publish_notification", noop)
    monkeypatch.setattr(EndogenousOutreach, "_should_roll", lambda self: _true())
    monkeypatch.setattr(act_emit, "publish_outreach_act", fake_act)
    monkeypatch.setattr(EndogenousOutreach, "_named_ungrounded_terms", lambda self, *a, **k: [])
    monkeypatch.setattr(dec, "record_decision", lambda result, **kw: rows.append(dict(result)))
    monkeypatch.setattr(dec, "fetch_recently_used_outreach_content_ids", lambda **kw: (set(), set()))


async def _true():
    return True


async def _tick(o, *, force=False):
    o._bus = object()
    o._sent_today_recovered = True
    r = await o.maybe_outreach(force=force)
    await _drain()
    return r


@pytest.mark.asyncio
async def test_tick_funded_sends_and_the_compose_id_is_the_want_id(market, monkeypatch) -> None:
    generated, rows = [], []
    _stub_tick(monkeypatch, generated, rows)
    o = _outreach()
    r = await _tick(o)
    assert r["outreach"] is True and r["reason"] == "sent"
    (want,) = market.wants
    assert generated == [want.want_id] and r["correlation_id"] == want.want_id
    assert r["economy"]["verdict"] == "funded" and r["economy"]["want_id"] == want.want_id
    assert rows[-1]["economy"]["producer"] == "orion-hub.outreach.binding"
    assert o._binder.wants == {}   # the attempt closed the want


@pytest.mark.asyncio
async def test_tick_price_hold_composes_nothing_and_rebids_once_per_window(market, monkeypatch) -> None:
    """No #2988-style flood: a held want is one bid per relief window, not one per 10 s tick."""
    generated, rows = [], []
    _stub_tick(monkeypatch, generated, rows)
    market.answers = [("deferred", "price", -0.05, 600.0)]
    o = _outreach()
    clock = {"t": T0}
    import scripts.endogenous_outreach as eo

    class _DT(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock["t"]

    monkeypatch.setattr(eo, "datetime", _DT)
    real_bind = o._binder.bind

    async def bind_at(key, **kw):
        kw["now"] = clock["t"]
        return await real_bind(key, **kw)

    o._binder.bind = bind_at
    reasons = []
    for _ in range(90):   # 15 minutes of 10 s ticks (re-checked every tick: never lapses)
        r = await _tick(o)
        reasons.append(r["reason"])
        clock["t"] += timedelta(seconds=10)
    assert generated == []
    assert set(reasons) == {"economy_held"}
    assert len(market.wants) == 2            # t=0 and t=600 s
    assert len({w.want_id for w in market.wants}) == 1
    held = rows[0]
    assert held["economy"]["verdict"] == "deferred:price" and held["economy"]["why"].startswith("hub.outreach")
    assert held["economy"]["reason"] == "price" and held["economy"]["retry_at"]
    assert len(rows) == 90 and all(r["reason"] == "economy_held" for r in rows)
    # Keep checking until the window ends: the held want composes anyway.
    while clock["t"] < T0 + timedelta(seconds=3600):
        r = await _tick(o)
        clock["t"] += timedelta(seconds=10)
    r = await _tick(o)
    assert r["reason"] == "sent" and r["economy"]["verdict"] == "window_end"


@pytest.mark.asyncio
async def test_tick_capacity_deferral_sends(market, monkeypatch) -> None:
    generated, rows = [], []
    _stub_tick(monkeypatch, generated, rows)
    market.answers = [("deferred", "capacity_reserved", 0.3, 200.0)]
    r = await _tick(_outreach())
    assert r["reason"] == "sent" and r["economy"]["verdict"] == "queued_at_pool"


@pytest.mark.asyncio
async def test_tick_market_down_sends_as_before(market, monkeypatch) -> None:
    generated, rows = [], []
    _stub_tick(monkeypatch, generated, rows)
    market.answers = ["raise"]
    r = await _tick(_outreach())
    assert r["reason"] == "sent" and r["economy"]["verdict"].startswith("no_answer")


@pytest.mark.asyncio
async def test_tick_forced_is_never_held(market, monkeypatch) -> None:
    generated, rows = [], []
    _stub_tick(monkeypatch, generated, rows)
    market.answers = [("deferred", "price", -1.0, 600.0)]
    r = await _tick(_outreach(), force=True)
    assert r["reason"] == "sent" and r["economy"]["verdict"] == "forced" and market.wants == []


@pytest.mark.asyncio
async def test_tick_forced_never_borrows_a_held_wants_id(market, monkeypatch) -> None:
    """Review finding 2: a forced send during a hold gets its own correlation id, and the held want stays."""
    generated, rows = [], []
    _stub_tick(monkeypatch, generated, rows)
    market.answers = [("deferred", "price", -1.0, 600.0)]
    o = _outreach()
    assert (await _tick(o))["reason"] == "economy_held"
    held_id = market.wants[0].want_id
    r = await _tick(o, force=True)
    assert r["reason"] == "sent" and r["correlation_id"] != held_id and "want_id" not in r["economy"]
    assert o._binder.wants["endogenous_tick"].want_id == held_id     # still held, not closed by the forced send


@pytest.mark.asyncio
async def test_tick_inside_a_hold_with_nothing_to_say_records_no_trigger_not_economy_held(market, monkeypatch) -> None:
    """Review finding 3: a held tick that has no content says so; it is not counted as an economy hold."""
    from scripts.endogenous_outreach import EndogenousOutreach

    generated, rows = [], []
    _stub_tick(monkeypatch, generated, rows)
    market.answers = [("deferred", "price", -1.0, 600.0)]
    o = _outreach()
    assert (await _tick(o))["reason"] == "economy_held"

    async def no_roll(self):
        return False

    monkeypatch.setattr(EndogenousOutreach, "_should_roll", no_roll)
    monkeypatch.setattr("scripts.endogenous_outreach._fetch_open_priors", lambda: ([], []))
    r = await _tick(o)
    assert r["reason"] == "no_tension_trigger" and len(market.wants) == 1


@pytest.mark.asyncio
async def test_tick_shadow_mode_is_todays_behaviour(market, monkeypatch) -> None:
    from scripts.settings import settings

    monkeypatch.setattr(settings, "ORION_PHYSIOLOGY_BID_HUB_OUTREACH", "shadow", raising=False)
    generated, rows = [], []
    _stub_tick(monkeypatch, generated, rows)
    market.answers = [("deferred", "price", -1.0, 600.0)]
    r = await _tick(_outreach())
    assert r["reason"] == "sent" and "economy" not in r
    assert market.wants == []    # no binding bid (the C3b shadow bid goes through physiology_shadow)


@pytest.mark.asyncio
async def test_strained_veto_is_still_in_place_and_the_binding_bid_never_decides_during_strain(
        market, monkeypatch) -> None:
    """The arousal_strained outreach veto stays in this PR (deleted after 7 days of this reader on)."""
    from scripts import endogenous_outreach as eo

    inp = eo.OutreachGateInputs(enabled=True, turn_in_flight=False, local_hour=12, quiet_start_hour=-1,
                                quiet_end_hour=-1, seconds_since_last_outreach=None, min_cooldown_sec=0.0,
                                sent_today=0, daily_cap=4, arousal_level="strained")
    assert eo.outreach_block_reason(inp) == eo.AROUSAL_STRAINED_REASON
    assert eo.outreach_block_reason(inp, skip_rate_gates=True) == eo.AROUSAL_STRAINED_REASON
    generated, rows = [], []
    _stub_tick(monkeypatch, generated, rows)
    monkeypatch.setattr(eo, "outreach_block_reason", lambda inputs, **kw: eo.AROUSAL_STRAINED_REASON)
    r = await _tick(_outreach(strain_report_enabled=False))
    assert r["reason"] == eo.AROUSAL_STRAINED_REASON and generated == []
    assert market.wants == []   # the binding bid never sees a strained tick


@pytest.mark.asyncio
async def test_door_admit_holds_and_records_why(market, monkeypatch) -> None:
    market.answers = [("deferred", "sated", 0.0, 900.0)]
    o = _outreach()
    admit, econ = await o.economy_admit(door="reading_finding_outreach", key="b1", cap_counted=False)
    assert admit is False and econ["verdict"] == "deferred:sated" and econ["door"] == "reading_finding_outreach"
    assert market.wants[0].cost.juniper_msgs == 0
    # Shadow mode: no bid, always admitted, nothing recorded.
    o.economy_mode = "shadow"
    assert await o.economy_admit(door="reading_finding_outreach", key="b2") == (True, {})
    assert len(market.wants) == 1
