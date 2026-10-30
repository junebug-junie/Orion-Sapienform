"""Body economy Patch C3b: hub's SHADOW bids (scripts/physiology_shadow.py).

The contract under test: a producer's real path is byte-identical whatever the economy answers
(funded, deferred, no answer, or an error), and the logged shadow decision carries the act's
correlation id (want_id) and the producer's real gate outcome.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import pytest

from orion.schemas.physiology_want import WantDecisionV1, WantV1

# conftest's _hub_service_isolation drops every ``scripts.*`` module before each test, so hub
# modules are imported inside fixtures and tests (a module-level import would be a stale copy).

ANSWERS = ["funded", "deferred", "none", "raises"]


def _decision(want: WantV1, outcome: str) -> WantDecisionV1:
    now = datetime.now(timezone.utc)
    return WantDecisionV1(
        decision_id=f"{want.want_id}@{now.isoformat()}", want_id=want.want_id, family=want.family,
        producer=want.producer, verb=want.verb, tier=want.tier, decided_at=now, outcome=outcome,
        reason="net_positive" if outcome == "funded" else "price", lane="agent", want=want,
    )


@pytest.fixture
def economy(monkeypatch):
    """Stub the client. ``economy.answer`` picks the reply; ``economy.wants`` records each bid."""

    class _Economy:
        answer = "funded"
        wants: list[WantV1] = []
        released: list[str] = []

    import scripts.physiology_shadow as ps

    eco = _Economy()
    eco.wants, eco.released, eco.ps = [], [], ps

    async def fake_abid(want):
        eco.wants.append(want)
        if eco.answer == "raises":
            raise RuntimeError("economy down")
        if eco.answer == "none":
            return None
        return _decision(want, eco.answer)

    monkeypatch.setattr(ps, "abid", fake_abid)
    monkeypatch.setattr(ps, "release", lambda want_id: eco.released.append(want_id))
    from scripts.settings import settings

    monkeypatch.setattr(settings, "HUB_PHYSIOLOGY_SHADOW_BID_ENABLED", True, raising=False)
    return eco


def _shadow_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("physiology_shadow_bid ")]


def _fields(line: str) -> dict:
    return dict(part.split("=", 1) for part in line.split()[1:])


async def _drain() -> None:
    for _ in range(5):
        await asyncio.sleep(0)
    await asyncio.sleep(0.01)


# --- the helper ------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("answer,expected", [("funded", "funded"), ("deferred", "deferred"), ("none", "none"),
                                             ("raises", "none")])
async def test_helper_logs_the_decision_beside_the_real_outcome(economy, caplog, answer, expected) -> None:
    caplog.set_level(logging.INFO, logger="orion-hub.physiology_shadow")
    economy.answer = answer
    shadow = economy.ps.ShadowBid(family="hub.outreach", verb="say", tier=2, want_id="corr-1", component="t")
    shadow.place()
    shadow.settle("blocked:quiet_hours")
    await _drain()
    (want,) = economy.wants
    assert (want.want_id, want.correlation_id, want.family, want.tier, want.producer) == (
        "corr-1", "corr-1", "hub.outreach", 2, "orion-hub.t")
    (line,) = _shadow_lines(caplog)
    f = _fields(line)
    assert (f["family"], f["want_id"], f["shadow"], f["real"], f["act_correlation_id"]) == (
        "hub.outreach", "corr-1", expected, "blocked:quiet_hours", "-")
    # A funded shadow reservation whose act was blocked is released; nothing else is.
    assert economy.released == (["corr-1"] if answer == "funded" else [])


@pytest.mark.asyncio
async def test_helper_ran_keeps_the_reservation_and_names_a_moved_act_id(economy, caplog) -> None:
    caplog.set_level(logging.INFO, logger="orion-hub.physiology_shadow")
    shadow = economy.ps.ShadowBid(family="hub.outreach", verb="say", tier=2, want_id="corr-1")
    shadow.place()
    shadow.settle("ran", act_correlation_id="corr-2")
    await _drain()
    f = _fields(_shadow_lines(caplog)[0])
    assert (f["real"], f["act_correlation_id"]) == ("ran", "corr-2")
    assert economy.released == []


@pytest.mark.asyncio
async def test_helper_flag_off_never_bids_or_logs(economy, caplog, monkeypatch) -> None:
    from scripts.settings import settings

    monkeypatch.setattr(settings, "HUB_PHYSIOLOGY_SHADOW_BID_ENABLED", False, raising=False)
    caplog.set_level(logging.INFO, logger="orion-hub.physiology_shadow")
    shadow = economy.ps.ShadowBid(family="hub.outreach", verb="say", tier=2)
    shadow.place()
    shadow.settle("ran")
    await _drain()
    assert economy.wants == [] and _shadow_lines(caplog) == []


def test_real_from_result() -> None:
    from scripts import physiology_shadow as ps

    assert ps.real_from_result({"outreach": True, "reason": "sent"}) == "ran"
    assert ps.real_from_result({"outreach": False, "reason": "cooldown"}) == "blocked:cooldown"


# --- endogenous outreach ---------------------------------------------------------------------


def _outreach(**overrides):
    from scripts.endogenous_outreach import EndogenousOutreach

    base = dict(enabled=True, tick_interval_sec=300.0, min_cooldown_sec=0.0, daily_cap=4, quiet_start_hour=-1,
                quiet_end_hour=-1, timeout_sec=5.0, agent_lane_timeout_sec=5.0,
                notify_channel="orion:notify:in_app", fallback_session_id="orion_outreach",
                trigger_evaluator=lambda: object())
    base.update(overrides)
    return EndogenousOutreach(**base)


def _stub_tick(monkeypatch, published: list) -> None:
    from scripts.endogenous_outreach import EndogenousOutreach, OutreachContext

    async def fake_gather(self, session_id, open_prior_previews=None, open_prior_ids=None):
        return OutreachContext(curiosity_summaries=["sustained prediction error on node:x"], recent_turns=[],
                               presence=None)

    async def fake_generate(self, prompt, session_id, correlation_id, **_kw):
        return "something worth saying", {"stub": True}

    async def fake_act(bus, **kw):
        published.append(kw["correlation_id"])
        return True

    async def noop(self, **kw):
        return None

    from scripts import act_emit

    monkeypatch.setattr(EndogenousOutreach, "_gather_context", fake_gather)
    monkeypatch.setattr(EndogenousOutreach, "_generate", fake_generate)
    monkeypatch.setattr(EndogenousOutreach, "_publish_history", noop)
    monkeypatch.setattr(EndogenousOutreach, "_publish_notification", noop)
    monkeypatch.setattr(act_emit, "publish_outreach_act", fake_act)
    monkeypatch.setattr(EndogenousOutreach, "_named_ungrounded_terms", lambda self, *a, **k: [])


def _stable(result: dict) -> dict:
    """The result minus per-tick random ids and clocks."""
    drop = {"at", "correlation_id", "decision_id", "provenance"}
    return {k: v for k, v in result.items() if k not in drop}


async def _tick(outreach) -> dict:
    outreach._bus = object()
    result = await outreach.maybe_outreach()
    await _drain()
    return result


@pytest.mark.asyncio
async def test_outreach_sent_path_is_identical_for_every_answer_and_joins_the_act(economy, caplog, monkeypatch) -> None:
    caplog.set_level(logging.INFO, logger="orion-hub.physiology_shadow")
    results = []
    for answer in ANSWERS:
        published: list[str] = []
        _stub_tick(monkeypatch, published)
        economy.answer = answer
        economy.wants.clear()
        caplog.clear()
        outreach = _outreach()
        queue: asyncio.Queue = asyncio.Queue()
        outreach.register_connection("c1", queue, {"correlation_id": None, "kind": None})
        result = await _tick(outreach)
        assert result["outreach"] is True and result["reason"] == "sent" and not queue.empty()
        (want,) = economy.wants
        # The join: want_id == the outreach act's correlation id == the tick's correlation id.
        assert published == [result["correlation_id"]] == [want.want_id]
        f = _fields(_shadow_lines(caplog)[0])
        assert (f["family"], f["want_id"], f["real"], f["act_correlation_id"]) == (
            "hub.outreach", want.want_id, "ran", "-")
        results.append(_stable(result))
    # Baseline: the same tick with the shadow switched off (no bid at all) answers the same.
    from scripts.settings import settings

    monkeypatch.setattr(settings, "HUB_PHYSIOLOGY_SHADOW_BID_ENABLED", False, raising=False)
    economy.wants.clear()
    _stub_tick(monkeypatch, [])
    off = await _tick(_outreach())
    assert economy.wants == []
    results.append(_stable(off))
    assert all(r == results[0] for r in results)


@pytest.mark.asyncio
async def test_outreach_orion_pass_logs_the_real_reason(economy, caplog, monkeypatch) -> None:
    from scripts.endogenous_outreach import EndogenousOutreach

    caplog.set_level(logging.INFO, logger="orion-hub.physiology_shadow")
    _stub_tick(monkeypatch, [])

    async def passes(self, prompt, session_id, correlation_id, **_kw):
        return "PASS", {"stub": True}

    monkeypatch.setattr(EndogenousOutreach, "_generate", passes)
    result = await _tick(_outreach())
    assert result["reason"] == "orion_passed"
    assert _fields(_shadow_lines(caplog)[0])["real"] == "blocked:orion_passed"


@pytest.mark.asyncio
async def test_outreach_gate_blocked_tick_never_bids(economy, caplog, monkeypatch) -> None:
    caplog.set_level(logging.INFO, logger="orion-hub.physiology_shadow")
    _stub_tick(monkeypatch, [])
    result = await _tick(_outreach(enabled=False))
    assert result["reason"] == "disabled"
    assert economy.wants == [] and _shadow_lines(caplog) == []


@pytest.mark.asyncio
async def test_outreach_strain_held_tick_bids_and_is_unchanged(economy, caplog, monkeypatch) -> None:
    from scripts.endogenous_outreach import AROUSAL_STRAINED_REASON

    caplog.set_level(logging.INFO, logger="orion-hub.physiology_shadow")
    _stub_tick(monkeypatch, [])
    monkeypatch.setattr("scripts.endogenous_outreach.outreach_block_reason",
                        lambda inputs, **kw: AROUSAL_STRAINED_REASON)
    results = []
    for answer in ANSWERS:
        economy.answer = answer
        economy.wants.clear()
        caplog.clear()
        result = await _tick(_outreach(strain_report_enabled=False))
        assert result["outreach"] is False and result["reason"] == AROUSAL_STRAINED_REASON
        assert len(economy.wants) == 1
        assert _fields(_shadow_lines(caplog)[0])["real"] == f"blocked:{AROUSAL_STRAINED_REASON}"
        results.append(_stable(result))
    assert all(r == results[0] for r in results)


def _offline(monkeypatch, outreach, clock):
    """Many ticks: skip the per-tick decision-log write and sent-today DB recovery; fake clock."""
    from scripts import endogenous_outreach_decisions as dec

    monkeypatch.setattr(dec, "record_decision", lambda *a, **k: None)
    outreach._sent_today_recovered = True
    outreach._monotonic = lambda: clock["t"]
    return outreach


@pytest.mark.asyncio
async def test_outreach_strain_hold_bids_once_per_window_not_every_tick(economy, caplog, monkeypatch) -> None:
    """Regression (2026-10-30): a strain hold repeats on every 10 s tick. Live, that put ~360
    hub.outreach shadow bids an hour into want_ledger (one per tick) while real sends were a few
    a day. One held want is one bid per cooldown window."""
    from scripts import endogenous_outreach as eo

    caplog.set_level(logging.INFO, logger="orion-hub.physiology_shadow")
    _stub_tick(monkeypatch, [])
    monkeypatch.setattr(eo, "outreach_block_reason", lambda inputs, **kw: eo.AROUSAL_STRAINED_REASON)
    clock = {"t": 1000.0}
    economy.answer = ANSWERS[0]
    economy.wants.clear()
    outreach = _offline(monkeypatch, _outreach(strain_report_enabled=False, min_cooldown_sec=2700.0,
                                                tick_interval_sec=10.0), clock)
    reasons = set()
    for _ in range(360):  # one hour of 10 s ticks
        result = await _tick(outreach)
        reasons.add(result["reason"])
        clock["t"] += 10.0
    assert reasons == {eo.AROUSAL_STRAINED_REASON}  # the real outcome is unchanged on every tick
    # Before the fix: 360. After: t=0 and t=2700 s.
    assert len(economy.wants) == 2
    assert len({w.want_id for w in economy.wants}) == 2
    assert [_fields(line)["real"] for line in _shadow_lines(caplog)] == [
        f"blocked:{eo.AROUSAL_STRAINED_REASON}"] * 2


@pytest.mark.asyncio
async def test_outreach_strain_bid_window_has_a_floor_when_cooldown_is_zero(economy, monkeypatch) -> None:
    from scripts import endogenous_outreach as eo

    _stub_tick(monkeypatch, [])
    monkeypatch.setattr(eo, "outreach_block_reason", lambda inputs, **kw: eo.AROUSAL_STRAINED_REASON)
    clock = {"t": 0.0}
    economy.answer = ANSWERS[0]
    economy.wants.clear()
    outreach = _offline(monkeypatch, _outreach(strain_report_enabled=False, min_cooldown_sec=0.0), clock)
    for _ in range(120):  # 20 minutes of 10 s ticks
        await _tick(outreach)
        clock["t"] += 10.0
    assert eo.STRAIN_SHADOW_BID_MIN_WINDOW_SEC == 600.0
    assert len(economy.wants) == 2  # t=0 and t=600 s


@pytest.mark.asyncio
async def test_outreach_compose_still_bids_on_every_compose(economy, monkeypatch) -> None:
    """The fix touches only the strain hold. Each real compose is a new candidate and bids."""
    from scripts import endogenous_outreach as eo

    _stub_tick(monkeypatch, [])
    clock = {"t": 0.0}
    economy.answer = ANSWERS[0]
    economy.wants.clear()
    outreach = _offline(monkeypatch, _outreach(daily_cap=10), clock)
    for _ in range(3):
        result = await _tick(outreach)
        assert result["reason"] == "sent"
        clock["t"] += 10.0
    assert len(economy.wants) == 3


# --- retro leisure ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ANSWERS)
async def test_leisure_refusal_is_unchanged_and_logged_by_dispatch_id(economy, caplog, answer) -> None:
    from orion.schemas.retro_leisure import RetroLeisureStartRequestV1
    from scripts.retro_leisure_listener import RetroLeisureListener

    caplog.set_level(logging.INFO, logger="orion-hub.physiology_shadow")
    economy.answer = answer

    class _Runner:
        async def should_start(self, config):
            return "cabinet_hot"

        async def run_leisure_session(self, config, *, act_id=None):
            raise AssertionError("refused: never runs")

    listener = RetroLeisureListener(None, config_provider=lambda: None, pool_provider=lambda: None,
                                    runner_provider=lambda: _Runner())
    out = await listener.start(RetroLeisureStartRequestV1(dispatch_id="dispatch:t1", correlation={}))
    await _drain()
    assert (out.state, out.refusal) == ("refused", "cabinet_hot")
    (want,) = economy.wants
    assert (want.want_id, want.family, want.verb) == ("dispatch:t1", "hub.retro_leisure", "play")
    f = _fields(_shadow_lines(caplog)[0])
    assert (f["want_id"], f["real"]) == ("dispatch:t1", "blocked:cabinet_hot")


# --- retro co-play ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ANSWERS)
async def test_coplay_open_is_unchanged_and_bids_on_the_act_id(economy, caplog, monkeypatch, answer) -> None:
    from scripts import act_emit
    from scripts import retro_memory_export as rx

    caplog.set_level(logging.INFO, logger="orion-hub.physiology_shadow")
    economy.answer = answer
    opened = []

    async def fake_open(bus, *, key, retro_session_id, started_at=None):
        opened.append(key)
        return True

    monkeypatch.setattr(act_emit, "publish_coplay_opened", fake_open)
    sink = rx.make_session_start_sink(bus_provider=lambda: object())
    assert await sink("retro-1", 1792699320.0) is True
    await _drain()
    assert opened == ["retro-1:1792699320.000"]
    (want,) = economy.wants
    # want_id is the play act's correlation id: the handoff key's stable UUID (coplay_turn_id).
    assert (want.want_id, want.family) == (act_emit.coplay_turn_id("retro-1:1792699320.000"), "hub.retro_coplay")
    assert _fields(_shadow_lines(caplog)[0])["real"] == "ran"


@pytest.mark.asyncio
async def test_strain_report_door_never_claims_its_report_as_this_want(economy, caplog, monkeypatch) -> None:
    """Strain holds the want; the strain report door sends its own message on its own act id."""
    from scripts.endogenous_outreach import AROUSAL_STRAINED_REASON, EndogenousOutreach

    caplog.set_level(logging.INFO, logger="orion-hub.physiology_shadow")
    _stub_tick(monkeypatch, [])
    monkeypatch.setattr("scripts.endogenous_outreach.outreach_block_reason",
                        lambda inputs, **kw: AROUSAL_STRAINED_REASON)

    async def fake_report(self, *, force, compose_started_at):
        return {"outreach": True, "reason": "sent", "correlation_id": "strain-report-act"}

    monkeypatch.setattr(EndogenousOutreach, "_strain_report_once", fake_report)
    result = await _tick(_outreach(strain_report_enabled=True))
    assert result["correlation_id"] == "strain-report-act"
    f = _fields(_shadow_lines(caplog)[0])
    assert (f["real"], f["act_correlation_id"]) == (f"blocked:{AROUSAL_STRAINED_REASON}", "-")


@pytest.mark.asyncio
async def test_a_slow_economy_never_holds_a_producer(economy, caplog, monkeypatch) -> None:
    """The bid runs beside the producer: an answer that has not come back yet holds nothing,
    and the line is logged when it does come back."""
    import scripts.physiology_shadow as ps

    caplog.set_level(logging.INFO, logger="orion-hub.physiology_shadow")
    gate = asyncio.Event()

    async def slow_abid(want):
        economy.wants.append(want)
        await gate.wait()
        return _decision(want, "deferred")

    monkeypatch.setattr(ps, "abid", slow_abid)
    _stub_tick(monkeypatch, [])
    result = await asyncio.wait_for(_tick(_outreach()), timeout=5.0)
    assert result["reason"] == "sent" and len(economy.wants) == 1
    assert _shadow_lines(caplog) == []          # still waiting on the economy; the tick is done
    gate.set()
    await _drain()
    assert _fields(_shadow_lines(caplog)[0])["shadow"] == "deferred"


@pytest.mark.asyncio
async def test_leisure_start_logs_ran_on_the_dispatch_id(economy, caplog, monkeypatch) -> None:
    from orion.schemas.retro_leisure import RetroLeisureStartRequestV1
    from scripts import act_emit
    from scripts import retro_leisure_listener as rl

    caplog.set_level(logging.INFO, logger="orion-hub.physiology_shadow")
    monkeypatch.setattr(rl, "START_GRACE_SEC", 0.01)
    opened = []

    async def fake_open(bus, *, dispatch_id, session_id):
        opened.append(dispatch_id)
        return True

    monkeypatch.setattr(act_emit, "publish_play_opened", fake_open)
    hold = asyncio.Event()

    class _Runner:
        async def should_start(self, config):
            return None

        async def run_leisure_session(self, config, *, act_id=None):
            await hold.wait()

    listener = rl.RetroLeisureListener(None, config_provider=lambda: None, pool_provider=lambda: None,
                                       runner_provider=lambda: _Runner())
    out = await listener.start(RetroLeisureStartRequestV1(dispatch_id="dispatch:t2", correlation={}))
    await _drain()
    assert out.state == "started" and opened == ["dispatch:t2"]
    f = _fields(_shadow_lines(caplog)[0])
    assert (f["want_id"], f["real"]) == ("dispatch:t2", "ran")
    listener.session_task.cancel()


@pytest.mark.asyncio
async def test_coplay_without_a_bus_is_logged_blocked(economy, caplog) -> None:
    from scripts import retro_memory_export as rx

    caplog.set_level(logging.INFO, logger="orion-hub.physiology_shadow")
    sink = rx.make_session_start_sink(bus_provider=lambda: None)
    assert await sink("retro-2", 5.0) is False
    await _drain()
    assert _fields(_shadow_lines(caplog)[0])["real"] == "blocked:no_bus"
