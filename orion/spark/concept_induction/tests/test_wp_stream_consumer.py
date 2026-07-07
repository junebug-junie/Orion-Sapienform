from __future__ import annotations

from typing import Optional

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.work_queue import (
    PendingMessage,
    ReadGroupBatch,
    StreamMessage,
    WorkQueueDecodeError,
)
from orion.spark.concept_induction.wp_stream_consumer import (
    WorldPulseStreamConsumer,
    extract_run_id,
)

STREAM = "orion:stream:world_pulse:run:result"
GROUP = "cg:concept-induction"
DLQ = "orion:stream:world_pulse:run:result:dlq"


def _envelope(run_id: str = "run-1") -> BaseEnvelope:
    return BaseEnvelope(
        kind="world.pulse.run.result.v1",
        source=ServiceRef(name="orion-world-pulse", node="athena", version="0.1.0"),
        payload={"run": {"run_id": run_id}},
    )


def _msg(message_id: str, run_id: str = "run-1") -> StreamMessage:
    return StreamMessage(
        stream=STREAM,
        message_id=message_id,
        envelope=_envelope(run_id),
        raw_fields={},
        enqueued_at_ms=None,
        delivery_count=None,
    )


class FakeQueue:
    """In-memory stand-in for RedisStreamWorkQueue exercising the consumer contract."""

    def __init__(self) -> None:
        self.ensure_group_calls: list[tuple] = []
        self.acked: list[str] = []
        self.dlq: list[dict] = []
        self.dlq_malformed: list[dict] = []
        self._new: list[list[StreamMessage]] = []
        self._new_decode_errors: list[list[WorkQueueDecodeError]] = []
        self._autoclaim: list[ReadGroupBatch] = []
        self._pending: list[PendingMessage] = []

    # --- configuration helpers ---
    def queue_new(self, messages, decode_errors=None) -> None:
        self._new.append(list(messages))
        self._new_decode_errors.append(list(decode_errors or []))

    def queue_autoclaim(self, batch: ReadGroupBatch) -> None:
        self._autoclaim.append(batch)

    def set_pending(self, pending) -> None:
        self._pending = list(pending)

    # --- RedisStreamWorkQueue surface used by the consumer ---
    async def ensure_group(self, stream, group, start_id="$", mkstream=True) -> None:
        self.ensure_group_calls.append((stream, group, start_id, mkstream))

    async def read_group(self, stream, group, consumer, count=1, block_ms=5000) -> ReadGroupBatch:
        if self._new:
            msgs = self._new.pop(0)
            errs = self._new_decode_errors.pop(0)
            return ReadGroupBatch(messages=msgs, decode_errors=errs)
        return ReadGroupBatch(messages=[], decode_errors=[])

    async def autoclaim(self, stream, group, consumer, min_idle_ms, start_id="0-0", count=10):
        if self._autoclaim:
            return "0-0", self._autoclaim.pop(0)
        return "0-0", ReadGroupBatch(messages=[], decode_errors=[])

    async def pending_range(self, stream, group, start="-", end="+", count=10, consumer=None):
        return list(self._pending)

    async def ack(self, stream, group, message_id) -> int:
        self.acked.append(message_id)
        return 1

    async def send_to_dlq(self, dlq_stream, message, error, reason, attempt) -> str:
        self.dlq.append({"id": message.message_id, "reason": reason, "attempt": attempt})
        return "dlq-1"

    async def send_malformed_to_dlq(self, dlq_stream, *, original_stream, original_message_id, raw_fields, error, reason) -> str:
        self.dlq_malformed.append({"id": original_message_id, "reason": reason})
        return "dlq-m-1"


class Tracker:
    def __init__(self, processed: Optional[set] = None) -> None:
        self.processed: set[str] = set(processed or set())
        self.handled: list[str] = []
        self.raise_on: set[str] = set()

    async def handler(self, envelope: BaseEnvelope) -> None:
        run_id = extract_run_id(envelope)
        if run_id in self.raise_on:
            raise RuntimeError(f"boom {run_id}")
        self.handled.append(run_id)

    def is_processed(self, run_id: str) -> bool:
        return run_id in self.processed

    def mark_processed(self, run_id: str) -> None:
        self.processed.add(run_id)


def _consumer(queue: FakeQueue, tracker: Tracker, **kwargs) -> WorldPulseStreamConsumer:
    return WorldPulseStreamConsumer(
        queue=queue,
        stream=STREAM,
        group=GROUP,
        consumer="athena:1",
        dlq_stream=DLQ,
        handler=tracker.handler,
        is_processed=tracker.is_processed,
        mark_processed=tracker.mark_processed,
        max_attempts=kwargs.pop("max_attempts", 5),
    )


def test_extract_run_id_from_payload() -> None:
    assert extract_run_id(_envelope("run-xyz")) == "run-xyz"


def test_extract_run_id_falls_back_to_correlation_id() -> None:
    env = BaseEnvelope(
        kind="world.pulse.run.result.v1",
        source=ServiceRef(name="x", node="athena", version="0.1.0"),
        payload={"no_run": True},
    )
    assert extract_run_id(env) == str(env.correlation_id)


@pytest.mark.asyncio
async def test_drain_new_happy_path_handles_marks_and_acks() -> None:
    queue = FakeQueue()
    tracker = Tracker()
    queue.queue_new([_msg("1-0", "run-1")])
    consumer = _consumer(queue, tracker)

    handled = await consumer.run_once()

    assert handled == 1
    assert tracker.handled == ["run-1"]
    assert tracker.is_processed("run-1") is True
    assert queue.acked == ["1-0"]
    assert queue.ensure_group_calls  # group ensured with start_id="$"
    assert queue.ensure_group_calls[0][2] == "$"


@pytest.mark.asyncio
async def test_duplicate_run_is_skipped_but_acked() -> None:
    queue = FakeQueue()
    tracker = Tracker(processed={"run-1"})
    queue.queue_new([_msg("1-0", "run-1")])
    consumer = _consumer(queue, tracker)

    await consumer.run_once()

    assert tracker.handled == []  # handler NOT called for an already-processed run
    assert queue.acked == ["1-0"]  # but still acked so it leaves the PEL


@pytest.mark.asyncio
async def test_handler_failure_leaves_message_unacked() -> None:
    queue = FakeQueue()
    tracker = Tracker()
    tracker.raise_on = {"run-boom"}
    queue.queue_new([_msg("1-0", "run-boom")])
    consumer = _consumer(queue, tracker)

    await consumer.run_once()

    assert queue.acked == []  # not acked => redelivered later
    assert tracker.is_processed("run-boom") is False  # not marked processed


@pytest.mark.asyncio
async def test_decode_error_goes_to_dlq_and_acked() -> None:
    queue = FakeQueue()
    tracker = Tracker()
    err = WorkQueueDecodeError(stream=STREAM, message_id="9-0", error="bad", raw_fields={"envelope": b"x"})
    queue.queue_new([], decode_errors=[err])
    consumer = _consumer(queue, tracker)

    await consumer.run_once()

    assert queue.dlq_malformed and queue.dlq_malformed[0]["id"] == "9-0"
    assert queue.acked == ["9-0"]


@pytest.mark.asyncio
async def test_reclaim_reprocesses_when_under_max_attempts() -> None:
    queue = FakeQueue()
    tracker = Tracker()
    queue.queue_autoclaim(ReadGroupBatch(messages=[_msg("1-0", "run-r")], decode_errors=[]))
    queue.set_pending([PendingMessage(stream=STREAM, message_id="1-0", consumer="dead", idle_ms=200000, deliveries=2)])
    consumer = _consumer(queue, tracker, max_attempts=5)

    await consumer.run_once()

    assert tracker.handled == ["run-r"]
    assert queue.acked == ["1-0"]
    assert queue.dlq == []


@pytest.mark.asyncio
async def test_reclaim_sends_to_dlq_when_over_max_attempts() -> None:
    queue = FakeQueue()
    tracker = Tracker()
    queue.queue_autoclaim(ReadGroupBatch(messages=[_msg("1-0", "run-poison")], decode_errors=[]))
    queue.set_pending([PendingMessage(stream=STREAM, message_id="1-0", consumer="dead", idle_ms=200000, deliveries=6)])
    consumer = _consumer(queue, tracker, max_attempts=5)

    await consumer.run_once()

    assert tracker.handled == []  # poison message never handled again
    assert queue.dlq and queue.dlq[0]["id"] == "1-0"
    assert queue.acked == ["1-0"]  # acked after DLQ so it leaves the PEL
