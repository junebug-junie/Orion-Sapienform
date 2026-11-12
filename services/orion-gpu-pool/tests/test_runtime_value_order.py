"""Attention patch 3b in the pool runtime: a lease request's market net is stored on the lease, reaches
the scheduler's view, and a value reorder is reported as one pool_queue_reordered event."""
from __future__ import annotations

from orion.gpu_pool.scheduler import Reordered
from orion.gpu_pool.value_order import ValueOrderConfig

from tests.test_runtime import acq, boot, make, run


def test_the_requests_net_is_stored_on_the_lease_and_reaches_the_scheduler_view():
    async def go():
        rt, _ = make()
        rt.value_order = ValueOrderConfig()
        await boot(rt)
        r = await rt.acquire(acq("agent", kind="hold", want_id="run-1", want_net=0.74,
                                 want_family="hub.world_pulse_read"))
        row = await rt.store.lease(r.lease_id)
        assert (row["want_id"], row["want_net"], row["want_family"]) == ("run-1", 0.74, "hub.world_pulse_read")
        v = rt._view(row)
        assert (v.want_net, v.want_family) == (0.74, "hub.world_pulse_read")
        plain = await rt.store.lease((await rt.acquire(acq("fast"))).lease_id)
        assert rt._view(plain).want_net is None            # no bid: upkeep, arrival order
    run(go())


def test_a_reorder_is_reported_with_both_leases_both_nets_and_the_margin():
    async def go():
        rt, _ = make()
        rt.value_order = ValueOrderConfig(margin=0.1, aging_per_h=0.1)
        await boot(rt)
        r = await rt.acquire(acq("agent", kind="hold", want_id="run-2", want_net=0.9, want_family="durable.self_inquiry"))
        row = await rt.store.lease(r.lease_id)
        await rt._report_reordered(Reordered(r.lease_id, "earlier", "agent", 0.9, 0.2, "durable.self_inquiry",
                                             "hub.world_pulse_read"), row)
        (ev,) = rt.bus.events("pool_queue_reordered")
        d = ev["detail"]
        assert ev["lease_id"] == r.lease_id and ev["role"] == "agent" and ev["reason"] == "value_order"
        assert (d["jumped_lease_id"], d["net"], d["jumped_net"], d["margin"]) == ("earlier", 0.9, 0.2, 0.1)
        assert (d["family"], d["jumped_family"], d["want_id"]) == ("durable.self_inquiry", "hub.world_pulse_read", "run-2")
    run(go())


def test_an_operator_class_lease_that_carries_a_bid_is_refused():
    """Leased evals (2026-11-12): operator work never bids in Orion's want market."""
    async def go():
        rt, _ = make()
        await boot(rt)
        r = await rt.acquire(acq("operator_agent", kind="hold", priority="background", want_id="x", want_net=0.5,
                                 want_family="eval"))
        assert (r.status, r.reason, r.lease_id) == ("unavailable", "operator_class_never_bids", None)
        ok = await rt.acquire(acq("operator_agent", kind="hold", priority="background"))
        assert ok.status in ("granted", "queued") and ok.lease_id
    run(go())
