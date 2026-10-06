"""U4 through the real runtime: incident event -> shed signal -> no new background/system grants,
reported on the lease trace, visible in state; cleared on resolve; lapses without refresh."""
from __future__ import annotations

from datetime import timedelta

from orion.schemas.gpu_pool import GpuPoolStateV1
from orion.schemas.hardware_watch import HardwareWatchIncidentV1, HardwareWatchShedV1

from tests.test_runtime import acq, acq_r, boot, make, run

INC = "a1b2c3d4e5f6a1b2c3d4e5f6"


def incident(rt, status="open", requested=True, valid=300, rule="cooling", transition=None):
    now = rt.now()
    return HardwareWatchIncidentV1(
        incident_id=INC, rule=rule, subject="cabinet_ac", transition=transition or ("opened" if status == "open" else "resolved"),
        status=status, open_reason="low_power", opened_at=now,
        resolved_at=now if status == "resolved" else None,
        shed=HardwareWatchShedV1(requested=requested, reason="cabinet_elevated" if requested else None,
                                 requested_at=now, cabinet_temp_c=30.1,
                                 valid_until=now + timedelta(seconds=valid)))


def shed_pool():
    rt, clock = make()
    rt.shed_enabled = True
    return rt, clock


def test_open_cooling_incident_stops_new_background_and_system_grants():
    async def go():
        rt, clock = shed_pool()
        await boot(rt)
        assert rt.on_incident(incident(rt)) == "set"
        # retryable leases wait under shed; a one-shot request is refused at once (D3)
        bg = await rt.acquire(acq_r("metacog", priority="background"))
        sys = await rt.acquire(acq_r("metacog", priority="system"))
        chat = await rt.acquire(acq("chat", priority="interactive"))
        assert bg.status == "queued" and sys.status == "queued" and chat.status == "granted"
        one_shot = await rt.acquire(acq("metacog", priority="system"))
        assert one_shot.status == "unavailable" and one_shot.reason == "shed:cooling_incident"
        await rt.tick()
        shed = [e for e in rt.bus.events("queued") if (e.get("detail") or {}).get("shed")]
        assert {e["lease_id"] for e in shed} == {bg.lease_id, sys.lease_id}
        assert all(e["reason"] == "shed:cooling_incident" and e["detail"]["sources"] == [INC] for e in shed)
        await rt.tick()   # edge-triggered: not re-reported while it lasts
        assert len([e for e in rt.bus.events("queued") if (e.get("detail") or {}).get("shed")]) == 2
        state = await rt.snapshot()
        GpuPoolStateV1.model_validate(state.model_dump(mode="json"))
        assert state.shed["active_reason"] == "cooling_incident"
        assert state.shed["blocked"] == {"background": "cooling_incident", "system": "cooling_incident"}
        # resolve clears it: the waiting work is granted on the next tick, in its original order
        assert rt.on_incident(incident(rt, status="resolved")) == "cleared"
        await rt.tick()
        assert (await rt.store.lease(bg.lease_id))["status"] == "granted"
        assert (await rt.store.lease(sys.lease_id))["status"] == "granted"
        assert (await rt.snapshot()).shed["blocked"] == {}
    run(go())


def test_running_work_keeps_running_under_shed():
    async def go():
        rt, _ = shed_pool()
        await boot(rt)
        before = await rt.acquire(acq("metacog", priority="background"))
        assert before.status == "granted"
        rt.on_incident(incident(rt))
        await rt.tick()
        row = await rt.store.lease(before.lease_id)
        assert row["status"] == "granted" and not rt.bus.events("recalled")
    run(go())


def test_kill_switch_shows_the_signal_but_blocks_nothing():
    async def go():
        rt, _ = make()          # shed_enabled defaults to False (code default OFF)
        assert rt.shed_enabled is False
        await boot(rt)
        rt.on_incident(incident(rt))
        r = await rt.acquire(acq("metacog", priority="background"))
        assert r.status == "granted"
        shed = (await rt.snapshot()).shed
        assert shed["enabled"] is False and shed["blocked"] == {}
        row = next(r for r in shed["reasons"] if r["name"] == "cooling_incident")
        assert row["active"] and not row["effective"]
    run(go())


def test_signal_lapses_without_a_refresh_and_a_refresh_extends_it():
    async def go():
        rt, clock = shed_pool()
        await boot(rt)
        rt.on_incident(incident(rt, valid=300))
        clock.advance(240)
        rt.on_incident(incident(rt, valid=300, transition="refresh"))   # the watcher's 60 s refresh
        clock.advance(240)
        assert rt.shed_view().blocked
        clock.advance(61)
        assert rt.shed_view().blocked == {}
    run(go())


def test_valid_until_is_capped_so_a_bad_clock_cannot_latch_shedding():
    async def go():
        rt, clock = shed_pool()
        await boot(rt)
        rt.on_incident(incident(rt, valid=10 * 86400))
        clock.advance(901)
        assert rt.shed_view().blocked == {}
    run(go())


def test_shed_not_requested_or_other_rules_do_not_shed():
    async def go():
        rt, _ = shed_pool()
        await boot(rt)
        assert rt.on_incident(incident(rt, requested=False)) == "noop"
        assert rt.on_incident(incident(rt, rule="cpu_heat")) == "ignored"
        assert rt.shed_view().blocked == {}
        rt.on_incident(incident(rt))
        assert rt.on_incident(incident(rt, requested=False, transition="refresh")) == "cleared"
    run(go())
