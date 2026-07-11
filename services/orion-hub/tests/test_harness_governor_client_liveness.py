"""HarnessGovernorClient.run should extend its RPC wait while the governor is still
alive (per harness_step_relay liveness), and give up promptly once it stalls.
"""
from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, HUB_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.core.bus.codec import OrionCodec  # noqa: E402
from orion.schemas.cognition.answer_contract import AnswerContract  # noqa: E402
from orion.schemas.context_exec import ContextExecPermissionV1  # noqa: E402
from orion.schemas.harness_finalize import HarnessRunRequestV1, HarnessRunV1  # noqa: E402
from orion.schemas.thought import StanceHarnessSliceV1, ThoughtEventV1  # noqa: E402

from scripts.harness_governor_client import HarnessGovernorClient  # noqa: E402
from scripts.settings import settings  # noqa: E402

_CORR_ID = "00000000-0000-4000-8000-000000000301"


def _request() -> HarnessRunRequestV1:
    thought = ThoughtEventV1(
        event_id="t-1",
        correlation_id=_CORR_ID,
        session_id="sess-1",
        created_at=datetime.now(timezone.utc),
        imperative="Answer directly.",
        tone="neutral",
        strain_refs=[],
        evidence_refs=[],
        disposition="proceed",
        stance_harness_slice=StanceHarnessSliceV1(
            task_mode="direct_response",
            conversation_frame="mixed",
            answer_strategy="direct",
        ),
    )
    return HarnessRunRequestV1(
        correlation_id=_CORR_ID,
        thought_event=thought,
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )


class _FakePubSub:
    """Mimics redis.asyncio's PubSub.get_message(timeout=X): waits up to `timeout`
    seconds for a message and returns None on timeout, without ever losing a message
    to task-cancellation the way an outer asyncio.wait_for around a fresh read would.
    """

    def __init__(self, bus: "_FakeBus") -> None:
        self._bus = bus

    async def get_message(self, ignore_subscribe_messages: bool = False, timeout: float = 0.0) -> dict | None:
        assert self._bus._published_at is not None
        if self._bus._reply_payload is None:
            await asyncio.sleep(timeout)
            return None
        remaining_to_reply = self._bus._reply_after_sec - (
            asyncio.get_event_loop().time() - self._bus._published_at
        )
        if remaining_to_reply > timeout:
            await asyncio.sleep(timeout)
            return None
        if remaining_to_reply > 0:
            await asyncio.sleep(remaining_to_reply)
        envelope = BaseEnvelope(
            kind="harness.run.v1",
            source=ServiceRef(name="test", version="0"),
            correlation_id=_CORR_ID,
            payload=self._bus._reply_payload,
        )
        return {"data": self._bus.codec.encode(envelope)}


class _FakeBus:
    """Stands in for OrionBusAsync: records publishes, delivers a reply at a fixed
    wall-clock point (`reply_after_sec` after publish) regardless of how many times
    the caller re-enters get_message to keep waiting.
    """

    def __init__(self, *, reply_after_sec: float, reply_payload: dict | None) -> None:
        self.codec = OrionCodec()
        self.publish_calls = 0
        self._reply_after_sec = reply_after_sec
        self._reply_payload = reply_payload
        self._published_at: float | None = None

    async def publish(self, channel: str, envelope: BaseEnvelope) -> None:
        self.publish_calls += 1
        if self._published_at is None:
            self._published_at = asyncio.get_event_loop().time()

    @asynccontextmanager
    async def subscribe(self, *channels: str, patterns: bool = False):
        yield _FakePubSub(self)


def _run_payload() -> dict:
    return HarnessRunV1(
        correlation_id=_CORR_ID,
        final_text="done",
        finalize_ran=True,
        step_count=3,
        compliance_verdict="completed",
        grounding_status="grounded",
    ).model_dump(mode="json")


@pytest.mark.asyncio
async def test_run_extends_wait_while_governor_stays_alive() -> None:
    # Reply arrives after the wait would normally have already fired once.
    poll_sec = 0.05
    bus = _FakeBus(reply_after_sec=poll_sec * 2.5, reply_payload=_run_payload())
    client = HarnessGovernorClient(bus)

    result = await client.run(
        _request(),
        correlation_id=_CORR_ID,
        timeout_sec=poll_sec,
        liveness_check=lambda _within_sec: True,
    )

    assert result is not None
    assert result.final_text == "done"
    assert bus.publish_calls == 1  # never re-publishes the request while retrying


@pytest.mark.asyncio
async def test_run_gives_up_immediately_when_not_alive() -> None:
    poll_sec = 0.05
    bus = _FakeBus(reply_after_sec=poll_sec * 10, reply_payload=_run_payload())
    client = HarnessGovernorClient(bus)

    result = await client.run(
        _request(),
        correlation_id=_CORR_ID,
        timeout_sec=poll_sec,
        liveness_check=lambda _within_sec: False,
    )

    assert result is None


@pytest.mark.asyncio
async def test_run_without_liveness_check_preserves_old_single_wait_behavior() -> None:
    poll_sec = 0.05
    bus = _FakeBus(reply_after_sec=poll_sec * 10, reply_payload=_run_payload())
    client = HarnessGovernorClient(bus)

    result = await client.run(
        _request(),
        correlation_id=_CORR_ID,
        timeout_sec=poll_sec,
        liveness_check=None,
    )

    assert result is None


@pytest.mark.asyncio
async def test_run_respects_max_wait_ceiling_even_if_alive() -> None:
    poll_sec = 0.05
    # Reply never arrives; liveness always true, but max_wait should still cut it off.
    bus = _FakeBus(reply_after_sec=10.0, reply_payload=_run_payload())
    client = HarnessGovernorClient(bus)
    original_max_wait = settings.HUB_HARNESS_GOVERNOR_RPC_MAX_WAIT_SEC
    settings.HUB_HARNESS_GOVERNOR_RPC_MAX_WAIT_SEC = poll_sec * 3
    try:
        result = await client.run(
            _request(),
            correlation_id=_CORR_ID,
            timeout_sec=poll_sec,
            liveness_check=lambda _within_sec: True,
        )
    finally:
        settings.HUB_HARNESS_GOVERNOR_RPC_MAX_WAIT_SEC = original_max_wait

    assert result is None


@pytest.mark.asyncio
async def test_liveness_check_receives_fixed_window_not_shrinking_poll_sec() -> None:
    """Regression test: liveness_check must always be called with the fixed
    HUB_HARNESS_GOVERNOR_LIVENESS_WINDOW_SEC, never with poll_sec — poll_sec shrinks
    over the loop (min(poll_sec, remaining)) for reasons unrelated to how recently the
    governor actually emitted a step, so using it as the recency threshold makes the
    check too lenient early on and too strict near the max-wait ceiling.
    """
    poll_sec = 0.05
    bus = _FakeBus(reply_after_sec=poll_sec * 10, reply_payload=_run_payload())
    client = HarnessGovernorClient(bus)
    seen_windows: list[float] = []

    original_window = settings.HUB_HARNESS_GOVERNOR_LIVENESS_WINDOW_SEC
    original_max_wait = settings.HUB_HARNESS_GOVERNOR_RPC_MAX_WAIT_SEC
    settings.HUB_HARNESS_GOVERNOR_LIVENESS_WINDOW_SEC = 7.0
    settings.HUB_HARNESS_GOVERNOR_RPC_MAX_WAIT_SEC = poll_sec * 3
    try:
        def _record(within_sec: float) -> bool:
            seen_windows.append(within_sec)
            return True

        result = await client.run(
            _request(),
            correlation_id=_CORR_ID,
            timeout_sec=poll_sec,
            liveness_check=_record,
        )
    finally:
        settings.HUB_HARNESS_GOVERNOR_LIVENESS_WINDOW_SEC = original_window
        settings.HUB_HARNESS_GOVERNOR_RPC_MAX_WAIT_SEC = original_max_wait

    assert result is None  # max_wait ceiling still cuts it off despite alive=True
    assert seen_windows, "liveness_check was never called"
    assert all(within_sec == 7.0 for within_sec in seen_windows)
    assert poll_sec not in seen_windows


class _FakeWorkerBus:
    """Stands in for an OrionBusAsync forked with start_rpc_worker=True: run() should
    reuse the shared _pending_rpc/_rpc_subscribe machinery instead of opening its own
    ad-hoc subscribe() connection, to avoid pinning one dedicated Redis connection per
    concurrent turn.
    """

    def __init__(self, *, rpc_worker_task: "asyncio.Task", reply_after_sec: float, reply_payload: dict) -> None:
        self.codec = OrionCodec()
        self.publish_calls = 0
        self.subscribe_calls = 0
        self._rpc_worker_task = rpc_worker_task
        self._rpc_lock = asyncio.Lock()
        self._pending_rpc: dict[tuple[str, str], asyncio.Future] = {}
        self._reply_after_sec = reply_after_sec
        self._reply_payload = reply_payload

    async def _rpc_subscribe(self, reply_channel: str) -> None:
        self.subscribe_calls += 1

    async def subscribe(self, *_args, **_kwargs):  # pragma: no cover - must not be used
        raise AssertionError("worker path must not open an ad-hoc subscribe() connection")

    async def publish(self, channel: str, envelope: BaseEnvelope) -> None:
        self.publish_calls += 1
        asyncio.get_event_loop().call_later(self._reply_after_sec, self._resolve_pending)

    def _resolve_pending(self) -> None:
        envelope = BaseEnvelope(
            kind="harness.run.v1",
            source=ServiceRef(name="test", version="0"),
            correlation_id=_CORR_ID,
            payload=self._reply_payload,
        )
        msg = {"data": self.codec.encode(envelope)}
        for fut in list(self._pending_rpc.values()):
            if not fut.done():
                fut.set_result(msg)


@pytest.mark.asyncio
async def test_run_uses_shared_worker_connection_when_available() -> None:
    poll_sec = 0.05
    worker_task = asyncio.ensure_future(asyncio.sleep(1000))
    try:
        bus = _FakeWorkerBus(
            rpc_worker_task=worker_task,
            reply_after_sec=poll_sec * 2.5,
            reply_payload=_run_payload(),
        )
        client = HarnessGovernorClient(bus)

        result = await client.run(
            _request(),
            correlation_id=_CORR_ID,
            timeout_sec=poll_sec,
            liveness_check=lambda _within_sec: True,
        )

        assert result is not None
        assert result.final_text == "done"
        assert bus.publish_calls == 1
        assert bus.subscribe_calls == 1
        assert bus._pending_rpc == {}  # cleaned up in the finally block
    finally:
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task
