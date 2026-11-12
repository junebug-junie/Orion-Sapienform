"""The pre-turn check worker must not make one turn's check wait behind another's.

Regression for 2026-11-12: the worker handled checks one at a time, so Juniper's
2.4 s check sat 180 s behind world-pulse checks and the Hub timed it out.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from app import main as exec_main


def test_pre_turn_appraisal_worker_is_concurrent() -> None:
    assert exec_main.pre_turn_appraisal_svc.concurrent_handlers is True


class _FakeBus:
    enabled = True
    redis = object()

    def __init__(self, envelopes: list[object]) -> None:
        self._envelopes = envelopes
        self.drained = asyncio.Event()
        self.codec = SimpleNamespace(
            decode=lambda data: SimpleNamespace(ok=True, envelope=data, error=None)
        )

    @asynccontextmanager
    async def subscribe(self, channel: str):
        yield object()

    async def iter_messages(self, pubsub):
        for env in self._envelopes:
            yield {"channel": "orion:cortex:pre_turn_appraisal:request", "data": env}
        self.drained.set()
        await asyncio.Event().wait()  # stay subscribed like a live pubsub


def test_slow_check_does_not_block_a_later_fast_check(monkeypatch) -> None:
    svc = exec_main.pre_turn_appraisal_svc
    release_slow = asyncio.Event()
    fast_done = asyncio.Event()

    async def fake_handle(*, channel: str, env) -> None:
        if env == "slow":
            await release_slow.wait()
        else:
            fast_done.set()

    async def scenario() -> None:
        bus = _FakeBus(["slow", "fast"])
        monkeypatch.setattr(svc, "bus", bus, raising=False)
        monkeypatch.setattr(svc, "_handle_decoded_envelope", fake_handle)
        monkeypatch.setattr(svc, "_stop", asyncio.Event(), raising=False)
        runner = asyncio.create_task(svc._run())
        try:
            # Serial handling would hang here: "fast" waits until "slow" is released.
            await asyncio.wait_for(fast_done.wait(), timeout=2.0)
            assert not release_slow.is_set()
        finally:
            release_slow.set()
            runner.cancel()
            for task in list(svc._inflight_tasks):
                task.cancel()
            await asyncio.gather(runner, *svc._inflight_tasks, return_exceptions=True)

    asyncio.run(scenario())
