from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

from orion.hub.harness_step_stream import relay_harness_run_steps


@pytest.mark.asyncio
async def test_relay_harness_run_steps_forwards_matching_correlation() -> None:
    frames: list[dict] = []
    stop = asyncio.Event()
    delivered = False

    async def _send(frame: dict) -> None:
        frames.append(frame)
        stop.set()

    class _Decoded:
        def __init__(self, payload: dict):
            self.ok = True
            self.envelope = MagicMock(payload=payload)

    class _FakePubSub:
        async def get_message(self, **kwargs):
            nonlocal delivered
            if delivered:
                await asyncio.sleep(0.05)
                return None
            delivered = True
            return {"type": "message", "data": b"x"}

        async def listen(self):
            msg = await self.get_message(ignore_subscribe_messages=True, timeout=0.5)
            if msg:
                yield msg

    @asynccontextmanager
    async def _subscribe(_channel: str):
        yield _FakePubSub()

    bus = MagicMock()
    bus.codec.decode.return_value = _Decoded(
        {
            "schema_version": "harness.run.step.v1",
            "correlation_id": "corr-1",
            "step_index": 0,
            "step": {"type": "assistant", "raw": {"type": "assistant"}},
        }
    )
    bus.subscribe = _subscribe
    bus.iter_messages = lambda pubsub: pubsub.listen()

    task = asyncio.create_task(
        relay_harness_run_steps(
            bus,
            correlation_id="corr-1",
            channel="orion:harness:run:step",
            send_frame=_send,
            stop_event=stop,
        )
    )
    await asyncio.wait_for(stop.wait(), timeout=2.0)
    stop.set()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert len(frames) == 1
    assert frames[0]["kind"] == "claude_step"
    assert frames[0]["mode"] == "orion"
    assert frames[0]["correlation_id"] == "corr-1"
