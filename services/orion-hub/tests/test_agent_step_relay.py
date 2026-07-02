"""AgentStepRelay fans agent_step events to per-correlation queues."""
from __future__ import annotations

import asyncio

import pytest

from scripts.agent_step_relay import AgentStepRelay


@pytest.mark.asyncio
async def test_relay_dispatches_to_registered_queue():
    relay = AgentStepRelay(channel="orion:context_exec:event")
    q: asyncio.Queue = asyncio.Queue()
    relay.register_queue("corr-1", q)

    await relay._dispatch_payload(
        kind="context.exec.agent_step.v1",
        payload={"correlation_id": "corr-1", "step_index": 0, "tool_id": "python_interpreter"},
    )
    item = q.get_nowait()
    assert item["kind"] == "agent_step"
    assert item["step"]["step_index"] == 0

    relay.unregister_queue("corr-1", q)


@pytest.mark.asyncio
async def test_relay_ignores_non_step_kinds_and_unknown_corr():
    relay = AgentStepRelay(channel="orion:context_exec:event")
    q: asyncio.Queue = asyncio.Queue()
    relay.register_queue("corr-1", q)

    await relay._dispatch_payload(kind="context.exec.finished.v1",
                                  payload={"correlation_id": "corr-1"})
    await relay._dispatch_payload(kind="context.exec.agent_step.v1",
                                  payload={"correlation_id": "other"})
    assert q.empty()
