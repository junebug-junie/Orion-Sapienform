"""A lane controller that cannot act on the pool's requests is loud, not silent.

Incident 2026-10-09 03:56 -> 2026-10-10 06:01: the circe controller ran code older than the
config it read, refused every request ``config_unloadable:ValidationError`` (136 times), the agent
seat granted nothing for 26 h, and the pool's /health said ok throughout. These pin: two such
refusals in a row degrade the seat (health, state payload, one alert); an answer that proves the
controller read its config clears it (one recovery alert); retryable refusals never trip it."""
from __future__ import annotations

import asyncio

from app.controller_alert import build_request
from app.controller_health import DEGRADE_AFTER, ControllerHealth, classify
from tests.test_holds_and_actuation import SEAT, actuations, boot, demand_gpu2, make, result, step
from tests.test_runtime import CFG, Clock, run
from tests.test_stage5_7_enforce import LOADED, enforce, statuses

STALE = "config_unloadable:ValidationError"


class Alerts:
    def __init__(self):
        self.sent: list[tuple[str, str, dict]] = []

    async def __call__(self, seat, state, view):
        self.sent.append((seat, state, view))


async def settle():
    for _ in range(3):
        await asyncio.sleep(0)


def wire(rt):
    rt.controller_alert = Alerts()
    return rt.controller_alert


async def refuse_next_load(rt, clock, home, reason):
    """Let the cooldown run out, take the retried load, refuse it."""
    await step(rt, clock, CFG.defaults.swap_cooldown_sec + 1, beat=[home.lease_id], every=30)
    msg = actuations(rt)[-1]
    assert msg.action == "load"
    await result(rt, msg, "refused", reason=reason)
    await settle()
    return msg


# --- pure ------------------------------------------------------------------------------------
def test_classify_names_only_reasons_a_retry_cannot_fix():
    assert classify(STALE) == "config_unreadable"
    assert classify("fence_state_unreadable:OSError") == "config_unreadable"
    assert classify("launch_digest_mismatch") == "config_mismatch"
    assert classify("no_launch_block:agent-gpu2") == "config_mismatch"
    assert classify("invalid_request:('max_holds',)") == "request_rejected"
    for transient in ("busy", "deadline_passed", "stale_generation", "upstream_not_idle:agent-gpu2",
                      "something_new", "", None):
        assert classify(transient) is None


def test_tracker_degrades_after_threshold_and_clears_on_answer():
    clock = Clock()
    h = ControllerHealth()
    assert h.on_refused(SEAT, STALE, host="circe", now=clock()) is None
    assert h.degraded() == {}
    clock.advance(600)
    t = h.on_refused(SEAT, STALE, host="circe", now=clock())
    assert t is not None and DEGRADE_AFTER == 2
    assert h.on_refused(SEAT, STALE, host="circe", now=clock()) is None   # already degraded: no re-alert
    v = h.view()[SEAT]
    assert v["degraded"] and v["refusals"] == 3 and v["host"] == "circe"
    assert "rebuild the controller on circe" in v["advice"].lower()
    assert h.on_answered(SEAT) is t
    assert h.view() == {} and h.on_answered(SEAT) is None


def test_build_request_is_an_error_card_with_the_fix_and_recovery_is_ack_free():
    h = ControllerHealth()
    clock = Clock()
    h.on_refused(SEAT, STALE, host="circe", now=clock())
    t = h.on_refused(SEAT, STALE, host="circe", now=clock())
    req = build_request(SEAT, "degraded", t.view(), source="orion-gpu-pool")
    assert req.severity == "error" and req.require_ack
    assert "circe" in req.reason and SEAT in req.reason
    assert "Rebuild the controller on circe" in req.message and STALE in req.message
    assert req.context["event"] == "controller_degraded" and req.context["refusals"] == 2
    ok = build_request(SEAT, "recovered", {**t.view(), "recovered_at": "x"}, source="orion-gpu-pool")
    assert ok.severity == "info" and not ok.require_ack


# --- through the runtime ---------------------------------------------------------------------
def test_repeated_config_unloadable_degrades_the_seat_then_a_success_clears_it():
    async def go():
        rt, clock = make()
        alerts = wire(rt)
        await boot(rt)
        home, _ = await demand_gpu2(rt, clock)
        [first] = actuations(rt)
        await result(rt, first, "refused", reason=STALE)
        await settle()
        assert rt.controller_health.degraded() == {} and alerts.sent == []   # one can be a mid-pull read

        await refuse_next_load(rt, clock, home, STALE)
        assert list(rt.controller_health.degraded()) == [SEAT]
        [(seat, state, view)] = alerts.sent
        assert (seat, state) == (SEAT, "degraded") and view["refusals"] == 2
        assert view["first_seen"] < view["last_seen"]

        # the state payload carries it on the seat's card (free `actuation` dict: no schema change)
        snap = await rt.snapshot()
        gpu2 = next(c for c in snap.cards if c.card == "gpu2")
        assert gpu2.actuation["controller_degraded"][SEAT]["reason"] == STALE
        assert gpu2.actuation["outcome"] == "refused"                    # the real action record is intact
        assert "controller_degraded" not in (rt.cards["gpu2"].swap_action or {})   # never persisted
        other = next(c for c in snap.cards if c.card != "gpu2")
        assert "controller_degraded" not in (other.actuation or {})

        # still refusing: counted, not re-alerted
        await refuse_next_load(rt, clock, home, STALE)
        assert len(alerts.sent) == 1 and rt.controller_health.view()[SEAT]["refusals"] == 3

        # controller rebuilt: the next retry is accepted -> cleared, one recovery alert
        await step(rt, clock, CFG.defaults.swap_cooldown_sec + 1, beat=[home.lease_id], every=30)
        msg = actuations(rt)[-1]
        await result(rt, msg, "accepted")
        await settle()
        assert rt.controller_health.degraded() == {} and rt.controller_health.view() == {}
        assert [s for _, s, _ in alerts.sent] == ["degraded", "recovered"]
        snap = await rt.snapshot()
        assert "controller_degraded" not in (next(c for c in snap.cards if c.card == "gpu2").actuation or {})
    run(go())


def test_retryable_refusals_never_degrade_the_seat():
    async def go():
        rt, clock = make()
        alerts = wire(rt)
        await boot(rt)
        home, _ = await demand_gpu2(rt, clock)
        await result(rt, actuations(rt)[-1], "refused", reason="busy")
        for reason in ("deadline_passed", "upstream_not_idle:agent-gpu2", "stale_generation", "busy"):
            await refuse_next_load(rt, clock, home, reason)
        assert rt.controller_health.view() == {} and alerts.sent == []
    run(go())


def test_a_retryable_refusal_between_two_stale_ones_does_not_reset_the_count():
    """busy/deadline_passed say nothing about the config either way: neutral."""
    async def go():
        rt, clock = make()
        alerts = wire(rt)
        await boot(rt)
        home, _ = await demand_gpu2(rt, clock)
        await result(rt, actuations(rt)[-1], "refused", reason=STALE)
        await refuse_next_load(rt, clock, home, "deadline_passed")
        await refuse_next_load(rt, clock, home, STALE)
        assert [s for _, s, _ in alerts.sent] == ["degraded"]
    run(go())


def test_boot_reconcile_refusal_counts_and_a_succeeded_status_clears():
    """enforce boot asks `status`; the stale controller refuses that too (it loads config first)."""
    async def go():
        rt, clock = enforce()
        alerts = wire(rt)
        await boot(rt)
        [st] = statuses(rt)
        await result(rt, st, "refused", reason=STALE)
        await settle()
        assert rt.controller_health.view()[SEAT]["refusals"] == 1 and alerts.sent == []
        home, _ = await demand_gpu2(rt, clock)
        load = [m for m in actuations(rt) if m.action == "load"][-1]
        await result(rt, load, "refused", reason=STALE)
        await settle()
        assert [s for _, s, _ in alerts.sent] == ["degraded"]
        await step(rt, clock, CFG.defaults.swap_cooldown_sec + 1, beat=[home.lease_id], every=30)
        again = actuations(rt)[-1]
        rt._world.up.add(SEAT)
        rt._world.up.discard("diffusion")
        await result(rt, again, "succeeded", observed=LOADED)
        await settle()
        assert rt.controller_health.view() == {}
        assert [s for _, s, _ in alerts.sent] == ["degraded", "recovered"]
    run(go())


def test_an_alert_that_raises_never_breaks_the_result_path():
    async def go():
        rt, clock = make()

        async def boom(seat, state, view):
            raise RuntimeError("notify down")

        rt.controller_alert = boom
        await boot(rt)
        home, _ = await demand_gpu2(rt, clock)
        await result(rt, actuations(rt)[-1], "refused", reason=STALE)
        await refuse_next_load(rt, clock, home, STALE)
        assert list(rt.controller_health.degraded()) == [SEAT]
        assert rt.cards["gpu2"].swap_state == "idle"
    run(go())


def test_health_reports_the_degraded_seat_plainly():
    import app.main as main

    async def go():
        rt, clock = make()
        await boot(rt)
        home, _ = await demand_gpu2(rt, clock)
        await result(rt, actuations(rt)[-1], "refused", reason=STALE)
        await refuse_next_load(rt, clock, home, STALE)
        saved = main._store, main.runtime
        main._store, main.runtime = None, rt
        try:
            return await main.health()
        finally:
            main._store, main.runtime = saved

    body = run(go())
    assert body["degraded"] == [SEAT]
    c = body["actuation"]["controller"][SEAT]
    assert c["degraded"] and c["kind"] == "config_unreadable" and c["first_seen"]
    assert "can't read its config" in c["advice"] and "Rebuild the controller on circe" in c["advice"]
