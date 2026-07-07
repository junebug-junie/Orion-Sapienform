from __future__ import annotations

import asyncio

from app.services import emit_runtime
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

RUN_RESULT_CHANNEL = "orion:world_pulse:run:result"


def _env(kind: str) -> BaseEnvelope:
    return BaseEnvelope(
        kind=kind,
        source=ServiceRef(name="orion-world-pulse", node="athena", version="0.1.0"),
        payload={"run": {"run_id": "run-1"}},
    )


class FakeQueue:
    calls: list[dict] = []

    def __init__(self, url: str) -> None:
        self.url = url
        FakeQueue.calls = []

    async def connect(self) -> None:
        pass

    async def enqueue(self, stream, envelope, maxlen=None, **kwargs) -> str:
        FakeQueue.calls.append({"stream": stream, "kind": envelope.kind, "maxlen": maxlen})
        return "1-0"

    async def close(self) -> None:
        pass


def test_enqueue_run_result_stream_selects_run_result_envelope(monkeypatch) -> None:
    monkeypatch.setattr(emit_runtime, "RedisStreamWorkQueue", FakeQueue)
    monkeypatch.setattr(emit_runtime.settings, "wp_run_result_stream_key", "orion:stream:world_pulse:run:result")
    monkeypatch.setattr(emit_runtime.settings, "wp_run_result_stream_maxlen", 1000)

    envelopes = [
        ("orion:world_pulse:digest:created", _env("world.pulse.digest.created.v1")),
        (RUN_RESULT_CHANNEL, _env("world.pulse.run.result.v1")),
    ]

    asyncio.run(emit_runtime._enqueue_run_result_stream(envelopes))

    assert len(FakeQueue.calls) == 1
    assert FakeQueue.calls[0]["stream"] == "orion:stream:world_pulse:run:result"
    assert FakeQueue.calls[0]["kind"] == "world.pulse.run.result.v1"
    assert FakeQueue.calls[0]["maxlen"] == 1000


def test_enqueue_skipped_when_no_run_result_envelope(monkeypatch) -> None:
    monkeypatch.setattr(emit_runtime, "RedisStreamWorkQueue", FakeQueue)
    FakeQueue.calls = []  # early-return path never constructs the queue, so reset explicitly
    envelopes = [("orion:world_pulse:digest:created", _env("world.pulse.digest.created.v1"))]

    asyncio.run(emit_runtime._enqueue_run_result_stream(envelopes))

    assert FakeQueue.calls == []


def test_enqueue_failure_is_swallowed(monkeypatch) -> None:
    class BoomQueue(FakeQueue):
        async def enqueue(self, stream, envelope, maxlen=None, **kwargs) -> str:
            raise RuntimeError("redis down")

    monkeypatch.setattr(emit_runtime, "RedisStreamWorkQueue", BoomQueue)
    envelopes = [(RUN_RESULT_CHANNEL, _env("world.pulse.run.result.v1"))]

    # Must not raise: a stream failure never fails the world-pulse run.
    asyncio.run(emit_runtime._enqueue_run_result_stream(envelopes))
