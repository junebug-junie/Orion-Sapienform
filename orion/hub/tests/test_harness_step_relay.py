from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

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
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get_message(self, **kwargs):
            nonlocal delivered
            if delivered:
                await asyncio.sleep(0.05)
                return None
            delivered = True
            return {"type": "message", "data": b"x"}

    bus = AsyncMock()
    bus.codec.decode.return_value = _Decoded(
        {
            "schema_version": "harness.run.step.v1",
            "correlation_id": "corr-1",
            "step_index": 0,
            "step": {"type": "assistant", "raw": {"type": "assistant"}},
        }
    )
    bus.subscribe.return_value = _FakePubSub()

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
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(frames) == 1
    assert frames[0]["kind"] == "claude_step"
    assert frames[0]["mode"] == "orion"
    assert frames[0]["correlation_id"] == "corr-1"
