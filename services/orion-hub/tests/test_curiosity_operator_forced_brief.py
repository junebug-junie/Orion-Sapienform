"""Body economy Patch D: Hub "Run now" (tick(force=True)) marks the durable brief operator_forced, so
durable-runs never holds an operator's run on the economy. A scheduled run never carries the key
(exclude_none), so an older consumer never sees it from one."""
from __future__ import annotations

import asyncio

from orion.schemas.durable_run import DurableRunRequestV1

from test_curiosity_investigation import _CortexBus, _loop


def _sent(bus) -> dict:
    assert len(bus.rpc_calls) == 1
    return bus.rpc_calls[0][1].payload["context"]["metadata"]["durable_run"]


def test_forced_run_brief_is_marked_operator_forced() -> None:
    bus = _CortexBus()
    assert asyncio.run(_loop(bus, kickoff_via_cortex=True).tick(force=True)) == "dispatched"
    raw = _sent(bus)
    assert raw["brief"]["operator_forced"] is True
    assert DurableRunRequestV1.model_validate(raw).brief.operator_forced is True


def test_scheduled_run_brief_never_carries_the_key() -> None:
    bus = _CortexBus()
    assert asyncio.run(_loop(bus, kickoff_via_cortex=True).tick()) == "dispatched"
    assert "operator_forced" not in _sent(bus)["brief"]
