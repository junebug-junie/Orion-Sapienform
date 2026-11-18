"""Warm standby through the real runtime (orion/gpu_pool/tests/test_warm_standby.py pins the
scheduler rule): the residents' quiet clocks start at boot, a granted diffusion lease restarts
them, and after standby_after_idle_sec of quiet the pool sends a `load` with reason=standby and
no agent lease waiting."""
from __future__ import annotations

from orion.gpu_pool.config import launch_digest
from orion.schemas.gpu_pool import GpuLeaseRequestV1
from tests.test_holds_and_actuation import SEAT, actuations, boot, make, step
from tests.test_runtime import CFG, WARM_CFG, run

QUIET = WARM_CFG.roles[SEAT].swap.standby_after_idle_sec


def test_warm_seat_loads_after_diffusion_quiet_and_never_before():
    async def go():
        rt, clock = make(cfg=WARM_CFG)
        await boot(rt)
        boot_at = clock()
        assert rt.cards["gpu2"].role_active_at["diffusion"] == boot_at   # clock starts at boot
        await step(rt, clock, QUIET - 60)
        assert actuations(rt) == []                                       # still settling
        await step(rt, clock, 120)
        [load] = actuations(rt)
        assert (load.role, load.action, load.reason) == (SEAT, "load", "standby")
    run(go())


def test_diffusion_activity_restarts_the_quiet_clock():
    async def go():
        rt, clock = make(cfg=WARM_CFG)
        await boot(rt)
        img = await rt.acquire(GpuLeaseRequestV1(verb="acquire", work_class="diffusion",
                                                 holder="durable-runs:reverie-visual-x", request_id="img:1",
                                                 kind="hold", priority="background", retryable=True))
        assert img.status == "granted" and img.grant.role == "diffusion"
        await step(rt, clock, QUIET + 60, beat=[img.lease_id])            # painting the whole time
        assert actuations(rt) == []
        assert (clock() - rt.cards["gpu2"].role_active_at["diffusion"]).total_seconds() <= 60
        await rt.release(img.lease_id, "ok")
        await step(rt, clock, QUIET - 60)
        assert actuations(rt) == []                                       # hysteresis after the painting
        await step(rt, clock, 120)
        assert [(a.action, a.reason) for a in actuations(rt)] == [("load", "standby")]
    run(go())


def test_cold_config_sends_nothing_without_demand():
    async def go():
        rt, clock = make()   # CFG: standby off
        await boot(rt)
        await step(rt, clock, QUIET * 2)
        assert actuations(rt) == []
    run(go())


def test_standby_does_not_change_the_launch_digest():
    """Pool (athena) and controller (circe) need no lockstep deploy for this field."""
    assert launch_digest(WARM_CFG, SEAT) == launch_digest(CFG, SEAT)
