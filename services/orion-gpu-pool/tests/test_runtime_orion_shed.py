"""Orion's learned shed (attend-to-act A1) through the real runtime, with an in-memory ledger.

Covers acceptance checks 0, 4 (pool side), 9 (kill switch) and 10 (reflex takes over mid-shed),
against a fake pool -- never production."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from orion.gpu_pool.orion_shed import MemoryOrionShedLedger, OrionShedCaps, OrionShedController
from orion.schemas.gpu_pool import GpuPoolShedReasonRequestV1

from tests.test_runtime import acq, acq_r, boot, make, run
from tests.test_runtime_shed import incident


def pool(*, enabled=True, lever=True, ledger=None, caps=None, clock=None):
    rt, clock = make(clock=clock)
    rt.shed_enabled = lever
    rt.orion_shed = OrionShedController(
        board=rt.shed_board, ledger=ledger or MemoryOrionShedLedger(), caps=caps or OrionShedCaps(),
        enabled=enabled, lever_enabled=lambda: rt.shed_enabled, now=lambda: rt.now())
    return rt, clock


def req(action="set", dispatch_id="dispatch:a", ttl=900, **kw):
    return GpuPoolShedReasonRequestV1(action=action, dispatch_id=dispatch_id, ttl_sec=ttl, **kw)


def test_request_can_never_name_the_reflex_reason():
    with pytest.raises(ValidationError):
        GpuPoolShedReasonRequestV1(action="set", reason="cooling_incident", dispatch_id="d")


def test_check0_reflex_reason_wins_and_orion_reason_is_background_only():
    async def go():
        rt, _ = pool()
        await boot(rt)
        out = await rt.orion_shed_request(req())
        assert out.ok and out.state == "active"
        view = rt.shed_view()
        assert view.blocked == {"background": "orion_self_shed"}       # never system
        rt.on_incident(incident(rt))
        view = rt.shed_view()
        assert view.blocked == {"background": "cooling_incident", "system": "cooling_incident"}
        names = [r["name"] for r in view.reasons]          # precedence order: every reflex before Orion's
        assert names[-1] == "orion_self_shed" and names.index("cooling_incident") < names.index("orion_self_shed")
    run(go())


def test_set_blocks_new_background_only_running_work_finishes_and_ttl_expires():
    async def go():
        rt, clock = pool()
        await boot(rt)
        running = await rt.acquire(acq("metacog", priority="background"))
        assert running.status == "granted"
        out = await rt.orion_shed_request(req(ttl=600))
        assert out.background_live_at_start == 1 and out.ttl_sec == 600
        waiting = await rt.acquire(acq_r("metacog", priority="background"))   # retryable: waits (D3)
        sys_ = await rt.acquire(acq("metacog", priority="system"))
        chat = await rt.acquire(acq("chat", priority="interactive"))
        assert waiting.status == "queued" and chat.status == "granted"
        one_shot = await rt.acquire(acq("fast", priority="background"))       # D3: refused at once
        assert one_shot.status == "unavailable" and one_shot.reason == "shed:orion_self_shed"
        await rt.tick()
        assert (await rt.store.lease(running.lease_id))["status"] == "granted"   # nothing recalled
        assert not rt.bus.events("recalled")
        clock.advance(30)
        await rt.tick()
        rec = rt.orion_shed.active
        assert rec.grants_withheld >= 1 and rec.delayed_grant_sec > 0
        assert waiting.lease_id in rec.withheld_ids and sys_.lease_id not in rec.withheld_ids
        clock.advance(600)
        await rt.tick()
        assert rt.orion_shed.active is None
        status = await rt.orion_shed_request(req(action="status"))
        assert status.state == "expired" and status.ended_at is not None
        # the 630 s without heartbeats expired the request leases; what matters is that nothing
        # still names orion_self_shed once the TTL ended
        assert rt.shed_view().blocked == {}
    run(go())


def test_drained_at_is_the_first_tick_with_no_background_granted():
    async def go():
        rt, clock = pool()
        await boot(rt)
        held = await rt.acquire(acq("metacog", priority="background"))
        await rt.orion_shed_request(req())
        await rt.tick()
        assert rt.orion_shed.active.drained_at is None
        clock.advance(5)
        await rt.release(held.lease_id, "ok", None)
        await rt.tick()
        assert rt.orion_shed.active.drained_at == rt.now()
    run(go())


def test_caps_are_enforced_in_the_pool_gap_daily_and_one_at_a_time():
    async def go():
        caps = OrionShedCaps(max_ttl_sec=900, max_sec_per_day=1800, min_gap_sec=900)
        rt, clock = pool(caps=caps)
        await boot(rt)
        first = await rt.orion_shed_request(req(dispatch_id="d1", ttl=3000))
        assert first.ttl_sec == 900                                      # clipped to max TTL
        assert (await rt.orion_shed_request(req(dispatch_id="d2"))).refusal == "already_active"
        clock.advance(900)
        await rt.tick()
        clock.advance(600)
        assert (await rt.orion_shed_request(req(dispatch_id="d3"))).refusal == "min_gap"
        clock.advance(301)
        second = await rt.orion_shed_request(req(dispatch_id="d4"))
        assert second.state == "active"
        clock.advance(900)
        await rt.tick()
        clock.advance(901)
        assert (await rt.orion_shed_request(req(dispatch_id="d5"))).refusal == "daily_cap"
    run(go())


def test_set_is_idempotent_per_dispatch_id():
    async def go():
        rt, _ = pool()
        await boot(rt)
        a = await rt.orion_shed_request(req(dispatch_id="same"))
        b = await rt.orion_shed_request(req(dispatch_id="same"))
        assert a.shed_id == b.shed_id and b.state == "active"
    run(go())


def test_refusals_kill_switch_lever_off_reflex_active_and_ledger_down():
    async def go():
        rt, _ = pool(enabled=False)
        await boot(rt)
        assert (await rt.orion_shed_request(req())).refusal == "disabled"
        rt, _ = pool(lever=False)
        await boot(rt)
        assert (await rt.orion_shed_request(req())).refusal == "lever_disabled"
        rt, _ = pool()
        await boot(rt)
        rt.on_incident(incident(rt))
        assert (await rt.orion_shed_request(req())).refusal == "reflex_active"
        rt, _ = pool(ledger=MemoryOrionShedLedger(available=False))
        await boot(rt)
        assert (await rt.orion_shed_request(req())).refusal == "ledger_unavailable"
    run(go())


def test_check9_kill_switch_drill_restart_with_flag_off_cancels_and_next_lease_grants():
    async def go():
        ledger = MemoryOrionShedLedger()
        rt, clock = pool(ledger=ledger)
        await boot(rt)
        await rt.orion_shed_request(req())
        assert (await rt.acquire(acq_r("metacog", priority="background"))).status == "queued"
        # the operator flips GPU_POOL_ORION_SHED_ENABLED=false and restarts the pool
        rt2, _ = pool(enabled=False, ledger=ledger, clock=clock)
        rt2.store = rt.store
        await boot(rt2)
        row = next(iter(ledger.rows.values()))
        assert row["state"] == "cancelled" and row["detail"]["ended_because"] == "kill_switch_at_boot"
        assert rt2.shed_view().blocked == {}
        waiting = await rt2.acquire(acq("metacog", priority="background"))
        await rt2.tick()
        assert (await rt2.store.lease(waiting.lease_id))["status"] == "granted"
    run(go())


def test_restart_with_flag_on_restores_the_active_shed_and_keeps_the_daily_cap():
    async def go():
        ledger = MemoryOrionShedLedger()
        rt, clock = pool(ledger=ledger)
        await boot(rt)
        await rt.orion_shed_request(req())
        clock.advance(120)
        rt2, _ = pool(ledger=ledger, clock=clock)
        await boot(rt2)
        assert rt2.orion_shed.active is not None
        assert rt2.shed_view().blocked == {"background": "orion_self_shed"}
        assert rt2.orion_shed.health()["used_sec_24h"] == pytest.approx(120, abs=1)
    run(go())


def test_clear_verb_settles_cancelled():
    async def go():
        rt, _ = pool()
        await boot(rt)
        await rt.orion_shed_request(req())
        out = await rt.orion_shed_request(req(action="clear"))
        assert out.state == "cancelled" and rt.shed_view().blocked == {}
    run(go())


def test_check10_reflex_takes_over_mid_shed():
    async def go():
        rt, _ = pool()
        await boot(rt)
        await rt.orion_shed_request(req())
        sys_ = await rt.acquire(acq("metacog", priority="system"))
        assert sys_.status == "granted"                      # the learned action never sheds system
        did = await rt.handle_incident(incident(rt))
        assert did == "set+orion_shed_preempted"
        view = rt.shed_view()
        active = [r["name"] for r in view.reasons if r["active"]]
        assert active == ["cooling_incident"]                 # the pool shows only the reflex
        assert view.blocked == {"background": "cooling_incident", "system": "cooling_incident"}
        status = await rt.orion_shed_request(req(action="status"))
        assert status.state == "preempted_by_reflex"
        assert status.detail["preempted_by_reflex"] == f"cooling_incident:{incident(rt).incident_id}"
    run(go())


def test_ac_incident_without_a_shed_no_longer_preempts():
    """Thermal v2 (C7): only the reflex's own shed ends the learned action, not any open incident."""
    async def go():
        rt, _ = pool()
        await boot(rt)
        await rt.orion_shed_request(req())
        await rt.handle_incident(incident(rt, requested=False))
        assert rt.orion_shed.active is not None
    run(go())


def test_cpu_heat_incident_does_not_preempt():
    async def go():
        rt, _ = pool()
        await boot(rt)
        await rt.orion_shed_request(req())
        await rt.handle_incident(incident(rt, rule="cpu_heat"))
        assert rt.orion_shed.active is not None
    run(go())


def test_state_snapshot_carries_the_orion_shed_health():
    async def go():
        rt, _ = pool()
        await boot(rt)
        await rt.orion_shed_request(req())
        shed = (await rt.snapshot()).shed
        assert shed["orion_self_shed"]["active"]["state"] == "active"
        assert shed["orion_self_shed"]["caps"]["max_sec_per_day"] == 3600
    run(go())


def test_ledger_down_at_boot_recovers_on_the_next_set():
    async def go():
        ledger = MemoryOrionShedLedger(available=False)
        rt, _ = pool(ledger=ledger)
        await boot(rt)
        assert (await rt.orion_shed_request(req(dispatch_id="d1"))).refusal == "ledger_unavailable"
        ledger.available = True
        assert (await rt.orion_shed_request(req(dispatch_id="d2"))).state == "active"
    run(go())


def test_open_ac_incident_without_a_shed_does_not_refuse():
    """Thermal v2 (C7/C12): the old open-incident set is gone; an alert-only incident refuses nothing."""
    async def go():
        rt, _ = pool()
        await boot(rt)
        await rt.handle_incident(incident(rt, requested=False))
        assert (await rt.orion_shed_request(req())).state == "active"
    run(go())


# --- thermal controller v2 (D2): the per-tick reflex signal ----------------------------------

def reflex(rt, reason="cabinet_hot", active=True, valid=90, source="orion-hardware-watch:athena:cabinet"):
    from datetime import timedelta

    from orion.schemas.hardware_watch import HardwareWatchReflexShedV1
    now = rt.now()
    return HardwareWatchReflexShedV1(source_id=source, active=active, reason=reason if active else None,
                                     valid_until=now + timedelta(seconds=valid) if active else None,
                                     cabinet={"temp_c": 34.4, "thermal_state": "hot"}, emitted_at=now)


def test_v2_reflex_preempts_the_learned_shed_and_refuses_it_while_active():
    async def go():
        rt, clock = pool()
        await boot(rt)
        await rt.orion_shed_request(req())
        assert await rt.handle_reflex_shed(reflex(rt)) == "set+orion_shed_preempted"
        assert rt.shed_view().blocked == {"background": "cabinet_hot", "system": "cabinet_hot"}
        assert await rt.handle_reflex_shed(reflex(rt)) == "refreshed"          # per-tick re-send
        assert (await rt.orion_shed_request(req(dispatch_id="d2"))).refusal == "reflex_active"
        assert await rt.handle_reflex_shed(reflex(rt, active=False)) == "cleared"
        assert rt.shed_view().blocked == {}
        clock.advance(901)                                  # past the learned action's own min_gap
        assert (await rt.orion_shed_request(req(dispatch_id="d3"))).state == "active"
    run(go())


def test_v2_reflex_lapses_when_the_watcher_stops_sending():
    async def go():
        rt, clock = pool()
        await boot(rt)
        await rt.handle_reflex_shed(reflex(rt, reason="cabinet_unknown", valid=90))
        assert rt.shed_view().blocked == {"background": "cabinet_unknown"}
        clock.advance(91)
        assert rt.shed_view().blocked == {}
        assert rt.reflex_active() is None
    run(go())


def test_v2_unknown_escalating_to_hot_replaces_it_for_the_same_source():
    async def go():
        rt, _ = pool()
        await boot(rt)
        await rt.handle_reflex_shed(reflex(rt, reason="cabinet_unknown"))
        await rt.handle_reflex_shed(reflex(rt, reason="cabinet_hot"))
        active = [r["name"] for r in rt.shed_view().reasons if r["active"]]
        assert active == ["cabinet_hot"]
    run(go())


def test_v2_one_shot_interactive_is_never_shed_under_cabinet_hot():
    async def go():
        rt, _ = pool()
        rt.shed_enabled = True
        await boot(rt)
        await rt.handle_reflex_shed(reflex(rt))
        turn = await rt.acquire(acq("metacog", priority="interactive"))
        bg = await rt.acquire(acq("metacog", priority="system"))
        assert turn.status == "granted"
        assert bg.status == "unavailable" and bg.reason == "shed:cabinet_hot"     # D3: refused at once
    run(go())
