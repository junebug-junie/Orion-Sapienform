"""Stage 7.3: the pool says when a role's hold limit is clamped below its max_holds (fewer
discovered slots than configured), on /health and once in the log -- never silently."""
from __future__ import annotations

import asyncio
import logging

from orion.gpu_pool.scheduler import RoleLive

from tests.test_runtime import boot, make


def test_hold_limits_on_health_and_an_edge_triggered_clamp_log(caplog):
    import app.main as main

    rt, _ = make()
    asyncio.run(boot(rt))
    seat = "agent-gpu2"
    # Unloaded seat (no slots): reported, but not news -- no warning.
    assert rt.hold_limits()[seat] == {"max_holds": 2, "reserve_one_off_slots": 0, "slots": 0,
                                      "effective": 0, "reason": "no_slots"}
    assert set(rt.hold_limits()) == {seat}          # every other role keeps the default of one

    caplog.set_level(logging.INFO, logger="orion-gpu-pool")
    rt.roles[seat] = RoleLive(seat, True, 1, 131072)   # e.g. the 1-slot Q4 rollback profile loaded
    rt._report_hold_caps()
    rt._report_hold_caps()                              # edge-triggered: said once
    clamped = [r for r in caplog.records if "gpu_pool_max_holds_clamped" in r.getMessage()]
    assert len(clamped) == 1 and clamped[0].levelno == logging.WARNING
    assert "effective=1" in clamped[0].getMessage() and "discovered slots 1" in clamped[0].getMessage()

    rt.roles[seat] = RoleLive(seat, True, 2, 131072)   # Bonsai, 2 x 131072
    rt._report_hold_caps()
    assert any("gpu_pool_max_holds_in_force role=agent-gpu2" in r.getMessage() for r in caplog.records)
    assert rt.hold_limits()[seat]["effective"] == 2 and rt.hold_limits()[seat]["reason"] is None

    saved = main.runtime
    main.runtime = rt
    try:
        body = asyncio.run(main.health())
    finally:
        main.runtime = saved
    assert body["holds"][seat]["effective"] == 2
