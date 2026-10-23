"""Perception priority in the live runtime (body-economy design Patch A0).

Perception leases are never shed by heat; their own load is watched instead (perception_load.py).
"""
from __future__ import annotations

from datetime import datetime, timedelta

from app.controller_alert import build_perception_request
from orion.gpu_pool.shed import ShedSignal
from tests.test_runtime import acq, boot, make, run


def test_perception_lease_granted_while_cabinet_hot_blocks_system():
    async def go():
        rt, clock = make()
        rt.shed_enabled = True
        await boot(rt)
        rt.shed_board.set(ShedSignal(reason="cabinet_hot", source_id="hw", set_at=rt.now(),
                                     valid_until=rt.now() + timedelta(seconds=90)))
        sight = await rt.acquire(acq("metacog", priority="perception"))
        own = await rt.acquire(acq("metacog", priority="system"))
        assert sight.status == "granted"
        assert own.status == "unavailable" and own.reason == "shed:cabinet_hot"   # shed, not slot-full
    run(go())


def test_expired_perception_lease_counts_its_hold_once():
    """An expiry emits two names (expired + released); the hold must be counted once."""
    async def go():
        rt, clock = make()
        await boot(rt)
        lease = await rt.acquire(acq("metacog", priority="perception"))
        assert lease.status == "granted"
        clock.advance(rt.cfg.defaults.request_lease_ttl_sec + 5)
        await rt.tick()
        row = await rt.store.lease(lease.lease_id)
        assert row["status"] not in ("granted", "recalling")
        held = rt.perception_load.held_last_hour(rt.now())
        assert 0 < held <= rt.cfg.defaults.request_lease_ttl_sec + 5
    run(go())


def test_released_perception_hold_counts_toward_the_hour_and_alerts_once():
    async def go():
        rt, clock = make()
        sent = []

        async def alert(name, state, view):
            sent.append((name, state, view))
            return True
        rt.perception_alert = alert
        rt.perception_load.ceiling_sec = 100
        await boot(rt)
        lease = await rt.acquire(acq("metacog", priority="perception"))
        other = await rt.acquire(acq("metacog", priority="system"))
        clock.advance(150)
        await rt.release(lease.lease_id, "ok")
        await rt.release(other.lease_id, "ok")          # not perception: not counted
        assert round(rt.perception_load.held_last_hour(rt.now())) == 150
        await rt.tick()
        await rt.tick()
        await rt.drain_alerts()
        [(name, state, view)] = sent
        assert (name, state) == ("perception", "high") and view["held_sec_last_hour"] == 150
        req = build_perception_request(name, state, view, source="orion-gpu-pool")
        assert req.severity == "warning" and req.require_ack

        clock.advance(3700)
        await rt.tick()
        await rt.drain_alerts()
        assert [s for _, s, _ in sent] == ["high", "recovered"]
    run(go())
