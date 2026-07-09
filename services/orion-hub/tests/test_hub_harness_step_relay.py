"""HarnessStepRelay fans harness.run.step events to per-correlation queues."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from scripts.harness_step_relay import HarnessStepRelay


@pytest.mark.asyncio
async def test_harness_step_relay_dispatches_matching_correlation() -> None:
    relay = HarnessStepRelay(channel="orion:harness:run:step")
    queue: asyncio.Queue = asyncio.Queue()
    relay.register_queue("corr-1", queue)

    step_event = MagicMock()
    step_event.correlation_id = "corr-1"
    step_event.step_index = 0
    step_event.step = {"type": "assistant", "raw": {"type": "assistant"}}

    await relay._dispatch_step(step_event)

    item = queue.get_nowait()
    assert item["kind"] == "claude_step"
    assert item["mode"] == "orion"
    assert item["correlation_id"] == "corr-1"
    assert item["step_index"] == 0


@pytest.mark.asyncio
async def test_harness_step_relay_ignores_unregistered_correlation() -> None:
    relay = HarnessStepRelay(channel="orion:harness:run:step")
    step_event = MagicMock()
    step_event.correlation_id = "missing"
    step_event.step_index = 0
    step_event.step = {"type": "assistant"}
    await relay._dispatch_step(step_event)
