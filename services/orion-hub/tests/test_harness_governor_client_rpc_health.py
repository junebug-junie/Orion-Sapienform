"""hub -> harness-governor hand-rolled RPC records ``governor:<mode>`` per-hop RPC
health and, on timeout, emits the same rpc_transport_timeout grammar atom rpc_request()
does (2026-09-24 metacog-capture / transport-EWMA spec, A0 mesh coverage)."""
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

from orion.core.bus.async_service import OrionBusAsync  # noqa: E402
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.core.bus.codec import OrionCodec  # noqa: E402
from orion.schemas.cognition.answer_contract import AnswerContract  # noqa: E402
from orion.schemas.context_exec import ContextExecPermissionV1  # noqa: E402
from orion.schemas.harness_finalize import HarnessRunRequestV1, HarnessRunV1  # noqa: E402
from orion.schemas.thought import StanceHarnessSliceV1, ThoughtEventV1  # noqa: E402

from scripts.harness_governor_client import HarnessGovernorClient, governor_hop_key  # noqa: E402

_CORR = "00000000-0000-4000-8000-000000000901"


def _request(mode: str | None = "orion") -> HarnessRunRequestV1:
    thought = ThoughtEventV1(
        event_id="t-1",
        correlation_id=_CORR,
        session_id="sess-1",
        created_at=datetime.now(timezone.utc),
        imperative="Answer directly.",
        tone="neutral",
        strain_refs=[],
        evidence_refs=[],
        disposition="proceed",
        stance_harness_slice=StanceHarnessSliceV1(
            task_mode="direct_response", conversation_frame="mixed", answer_strategy="direct"
        ),
    )
    return HarnessRunRequestV1(
        correlation_id=_CORR,
        thought_event=thought,
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
        mode=mode,
    )


class _PubSub:
    def __init__(self, bus: "_Bus") -> None:
        self._bus = bus

    async def get_message(self, ignore_subscribe_messages: bool = False, timeout: float = 0.0):
        if self._bus.reply_payload is None:
            await asyncio.sleep(timeout)
            return None
        env = BaseEnvelope(
            kind="harness.run.v1",
            source=ServiceRef(name="test", version="0"),
            correlation_id=_CORR,
            payload=self._bus.reply_payload,
        )
        return {"data": self._bus.codec.encode(env)}


class _Bus:
    """Ad-hoc-subscribe path fake with a REAL aggregator behind record_hop_*."""

    def __init__(self, reply_payload: dict | None) -> None:
        self.codec = OrionCodec()
        self.reply_payload = reply_payload
        self._real = OrionBusAsync("redis://unused:6379/0")
        self.grammar_calls: list[dict] = []

    async def publish(self, channel: str, envelope: BaseEnvelope) -> None:
        return None

    @asynccontextmanager
    async def subscribe(self, *channels: str, patterns: bool = False):
        yield _PubSub(self)

    def record_hop_success(self, hop: str, elapsed_ms: float) -> None:
        self._real.record_hop_success(hop, elapsed_ms)

    def record_hop_timeout(self, hop: str, elapsed_ms: float | None = None) -> None:
        self._real.record_hop_timeout(hop, elapsed_ms)

    async def emit_rpc_timeout_grammar(self, **kwargs) -> None:
        self.grammar_calls.append(kwargs)


def _payload(**overrides) -> dict:
    base = dict(
        correlation_id=_CORR,
        final_text="done",
        finalize_ran=True,
        step_count=1,
        compliance_verdict="completed",
        grounding_status="grounded",
    )
    base.update(overrides)
    return HarnessRunV1(**base).model_dump(mode="json")


def test_hop_key_uses_mode() -> None:
    assert governor_hop_key(_request("orion")) == "governor:orion"
    assert governor_hop_key(_request("agent")) == "governor:agent"
    assert governor_hop_key(_request(None)) == "governor:unknown"


@pytest.mark.asyncio
async def test_reply_records_governor_hop_success_no_grammar() -> None:
    bus = _Bus(_payload())
    result = await HarnessGovernorClient(bus).run(_request("agent"), correlation_id=_CORR, timeout_sec=0.5)
    assert result is not None
    snap = bus._real.get_rpc_health_snapshot()
    hop = snap.channel_latency["governor:agent"]
    assert (hop.success_count, hop.timeout_count) == (1, 0)
    assert snap.success_count == 0  # pooled untouched
    assert bus.grammar_calls == []


@pytest.mark.asyncio
async def test_timeout_records_governor_hop_timeout_and_emits_grammar_atom() -> None:
    bus = _Bus(None)
    result = await HarnessGovernorClient(bus).run(
        _request("orion"), correlation_id=_CORR, timeout_sec=0.05, liveness_check=lambda _s: False
    )
    assert result is None
    hop = bus._real.get_rpc_health_snapshot().channel_latency["governor:orion"]
    assert (hop.success_count, hop.timeout_count) == (0, 1)
    assert len(bus.grammar_calls) == 1
    call = bus.grammar_calls[0]
    assert call["corr"] == _CORR
    assert call["reply_channel"].endswith(_CORR)
    assert call["timeout_elapsed_ms"] >= 40.0
    assert call["timeout_sec"] == pytest.approx(call["timeout_elapsed_ms"] / 1000.0)


@pytest.mark.asyncio
async def test_bus_without_hop_api_still_completes_the_run() -> None:
    """Old fakes/wrappers lacking record_hop_*/emit_rpc_timeout_grammar must not break a turn."""

    class _Bare(_Bus):
        record_hop_success = None  # type: ignore[assignment]
        record_hop_timeout = None  # type: ignore[assignment]
        emit_rpc_timeout_grammar = None  # type: ignore[assignment]

    assert await HarnessGovernorClient(_Bare(_payload())).run(_request(), correlation_id=_CORR, timeout_sec=0.5)
    assert await HarnessGovernorClient(_Bare(None)).run(
        _request(), correlation_id=_CORR, timeout_sec=0.05, liveness_check=lambda _s: False
    ) is None


@pytest.mark.asyncio
async def test_real_bus_emit_rpc_timeout_grammar_delegates() -> None:
    bus = OrionBusAsync("redis://unused:6379/0")
    seen: dict = {}

    async def _fake(**kwargs):
        seen.update(kwargs)

    bus._emit_rpc_timeout_grammar = _fake  # type: ignore[assignment]
    await bus.emit_rpc_timeout_grammar(
        request_channel="a", reply_channel="b", corr="c", timeout_sec=1.0, timeout_elapsed_ms=1000.0
    )
    assert seen["request_channel"] == "a" and seen["timeout_elapsed_ms"] == 1000.0


@pytest.mark.asyncio
async def test_hub_main_rpc_health_publish_is_gated_and_drains_rpc_bus(monkeypatch) -> None:
    import scripts.main as hub_main
    from unittest.mock import AsyncMock

    rpc = OrionBusAsync("redis://unused:6379/0")
    main_bus = OrionBusAsync("redis://unused:6379/0")
    rpc.publish = AsyncMock()  # type: ignore[assignment]
    monkeypatch.setattr(hub_main, "rpc_bus", rpc)
    monkeypatch.setattr(hub_main, "bus", main_bus)
    monkeypatch.setattr(hub_main, "_rpc_health_task", None)

    monkeypatch.setattr(hub_main.settings, "RPC_HEALTH_PUBLISH_ENABLED", False)
    hub_main._start_rpc_health_publish()
    assert hub_main._rpc_health_task is None  # default off

    monkeypatch.setattr(hub_main.settings, "RPC_HEALTH_PUBLISH_ENABLED", True)
    monkeypatch.setattr(hub_main.settings, "RPC_HEALTH_PUBLISH_INTERVAL_SEC", 0.02)
    monkeypatch.setattr(hub_main.settings, "RPC_HEALTH_CHANNEL_LATENCY_ENABLED", True)
    rpc.record_hop_success("governor:orion", 1000.0)
    main_bus.record_hop_timeout("governor:agent", None)
    hub_main._start_rpc_health_publish()
    try:
        for _ in range(100):
            if rpc.publish.await_count:
                break
            await asyncio.sleep(0.01)
    finally:
        await hub_main._stop_rpc_health_publish()
    assert hub_main._rpc_health_task is None
    channel, env = rpc.publish.await_args_list[0].args
    assert channel == "orion:rpc_health:snapshot"
    assert env.payload["instance"] == "main"
    assert set(env.payload["channel_latency"]) == {"governor:orion", "governor:agent"}
