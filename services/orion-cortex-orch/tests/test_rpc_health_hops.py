"""cortex-orch per-hop RPC health: chat-lane verb path records ``verb:<name>`` and
metacog dispatch labels its hop ``log_orion_metacognition`` (2026-09-24 spec, A0)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, APP_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import app.orchestrator as orchestrator
from app.clients import CortexExecClient
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.cortex.schemas import ExecutionPlan, ExecutionStep, PlanExecutionArgs, PlanExecutionRequest

from types import SimpleNamespace

from orion.core.verbs.models import VerbResultV1
from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, LLMMessage, RecallDirective

SOURCE = ServiceRef(name="cortex-orch", version="0", node="n")


def _client_request() -> CortexClientRequest:
    return CortexClientRequest(
        mode="agent",
        route_intent="none",
        verb=None,
        packs=["executive_pack"],
        options={"supervised": True, "force_agent_chain": False},
        recall=RecallDirective(enabled=False, required=False, mode="hybrid", profile=None),
        context=CortexClientContext(
            messages=[LLMMessage(role="user", content="hi")],
            raw_user_text="hi",
            user_message="hi",
            session_id="sid-1",
            user_id="user-1",
            trace_id="trace-1",
            metadata={},
        ),
    )


class _SubscribeCtx:
    def __init__(self, channel: str):
        self.pubsub = SimpleNamespace(channel=channel)

    async def __aenter__(self):
        return self.pubsub

    async def __aexit__(self, *exc):
        return False


class _FakeBus:
    """Replies to the hand-rolled orion:verb:request publish with a matching VerbResultV1."""

    def __init__(self) -> None:
        self.codec = SimpleNamespace(decode=lambda data: SimpleNamespace(ok=True, envelope=data, error=None))
        self.published: list[tuple[str, BaseEnvelope]] = []

    def subscribe(self, channel: str):
        return _SubscribeCtx(channel)

    async def publish(self, channel: str, env: BaseEnvelope):
        self.published.append((channel, env))

    async def iter_messages(self, pubsub):
        env = next(e for c, e in self.published if c == "orion:verb:request")
        yield {
            "data": BaseEnvelope(
                kind="verb.result",
                source=ServiceRef(name="cortex-exec", version="0", node="n"),
                correlation_id=env.correlation_id,
                payload=VerbResultV1(
                    verb="legacy.plan", ok=True, output={}, request_id=env.payload["request_id"]
                ).model_dump(mode="json"),
            )
        }


def _plan_request(*args: Any, **kwargs: Any) -> PlanExecutionRequest:
    return PlanExecutionRequest(
        plan=ExecutionPlan(
            verb_name="agent_runtime",
            label="agent-runtime",
            description="",
            category="agentic",
            priority="normal",
            interruptible=True,
            can_interrupt_others=False,
            timeout_ms=1000,
            max_recursion_depth=1,
            steps=[ExecutionStep(verb_name="agent_runtime", step_name="context_exec", order=0, services=["ContextExecService"])],
            metadata={"mode": "agent"},
        ),
        args=PlanExecutionArgs(request_id="trace-1", extra={"mode": "agent", "supervised": True}),
        context={"mode": "agent", "packs": ["executive_pack"]},
    )


class _RecordingBus(_FakeBus):
    """_FakeBus plus a real RpcHealthAggregator behind the public record_hop_* API."""

    def __init__(self) -> None:
        super().__init__()
        self._real = OrionBusAsync("redis://unused:6379/0")

    def record_hop_success(self, hop: str, elapsed_ms: float) -> None:
        self._real.record_hop_success(hop, elapsed_ms)

    def record_hop_timeout(self, hop: str, elapsed_ms: float | None = None) -> None:
        self._real.record_hop_timeout(hop, elapsed_ms)

    def snapshot(self):
        return self._real.get_rpc_health_snapshot()


class _SilentBus(_RecordingBus):
    async def iter_messages(self, pubsub):  # never replies -> timeout
        await asyncio.sleep(10)
        yield {}  # pragma: no cover


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(orchestrator, "build_plan_request", _plan_request)

    async def _no_state(*a, **k):
        return None

    monkeypatch.setattr(orchestrator, "_maybe_fetch_state", _no_state)
    # Force the hand-rolled orion:verb:request path (the chat lane path) regardless of
    # the lane resolver's decision for this request.
    settings = orchestrator.get_settings()
    monkeypatch.setattr(settings, "exec_lane_routing_enabled", False)


def test_chat_lane_verb_path_records_verb_hop_success() -> None:
    bus = _RecordingBus()
    result = asyncio.run(
        orchestrator.call_verb_runtime(
            bus, source=SOURCE, client_request=_client_request(), correlation_id="11111111-1111-1111-1111-000000000001", timeout_sec=5.0
        )
    )
    assert result.ok is True
    snap = bus.snapshot()
    hop = snap.channel_latency["verb:agent_runtime"]
    assert (hop.success_count, hop.timeout_count) == (1, 0)
    assert hop.max_ms is not None
    assert snap.success_count == 0  # pooled fields untouched by hand-rolled hops


def test_chat_lane_verb_path_records_verb_hop_timeout() -> None:
    bus = _SilentBus()
    with pytest.raises(TimeoutError):
        asyncio.run(
            orchestrator.call_verb_runtime(
                bus, source=SOURCE, client_request=_client_request(), correlation_id="11111111-1111-1111-1111-000000000002", timeout_sec=0.05
            )
        )
    hop = bus.snapshot().channel_latency["verb:agent_runtime"]
    assert (hop.success_count, hop.timeout_count) == (0, 1)


def test_verb_hop_record_failure_never_breaks_the_turn() -> None:
    """A bus without record_hop_* (older fake/wrapper) must not break call_verb_runtime."""
    result = asyncio.run(
        orchestrator.call_verb_runtime(
            _FakeBus(), source=SOURCE, client_request=_client_request(), correlation_id="11111111-1111-1111-1111-000000000003", timeout_sec=5.0
        )
    )
    assert result.ok is True


def test_metacog_dispatch_passes_health_label(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    async def _execute_plan(self, **kwargs):
        captured.update(kwargs)
        captured["request_channel"] = self.request_channel
        return {}

    monkeypatch.setattr(CortexExecClient, "execute_plan", _execute_plan)
    env = BaseEnvelope(
        kind="orion.metacog.trigger.v1",
        source=ServiceRef(name="equilibrium", version="0", node="n"),
        payload={"trigger_kind": "transport", "reason": "t", "zen_state": "unknown", "pressure": 0.1},
    )
    asyncio.run(orchestrator.dispatch_metacog_trigger(object(), source=SOURCE, env=env))
    assert captured["health_label"] == "log_orion_metacognition" == orchestrator.METACOG_HEALTH_LABEL


def test_exec_client_forwards_health_label_to_rpc_request() -> None:
    seen: dict[str, Any] = {}

    class _Bus:
        async def rpc_request(self, channel, env, **kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop")

    client = CortexExecClient(_Bus(), request_channel="orion:cortex:exec:request:background", result_prefix="r")
    req = _plan_request()
    with pytest.raises(RuntimeError):
        asyncio.run(client.execute_plan(source=SOURCE, req=req, correlation_id="11111111-1111-1111-1111-000000000004", timeout_sec=1.0, health_label="x"))
    assert seen["health_label"] == "x"

    seen.clear()
    with pytest.raises(RuntimeError):
        asyncio.run(client.execute_plan(source=SOURCE, req=req, correlation_id="11111111-1111-1111-1111-000000000004", timeout_sec=1.0))
    assert "health_label" not in seen  # unlabelled calls keep the old call shape


def test_main_publish_loop_wiring_folds_metacog_bus_and_sets_instance() -> None:
    """Static guard: importing app.main starts real chassis objects, so check the call
    site's source. Without the hop-only fold, the metacog hop (recorded on the
    equilibrium Hunter's bus) would never be published."""
    import ast

    tree = ast.parse((APP_ROOT / "app" / "main.py").read_text())
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "rpc_health_publish_loop"
    ]
    assert len(calls) == 1
    kw = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    assert kw["instance"] == "'main'"
    assert kw["include_channel_latency"] == "s.rpc_health_channel_latency_enabled"
    # svc.bus: verb:* hops recorded via _bus_for_rpc() before the fork is ready.
    assert kw["hop_only_bus_getters"] == "[lambda: equilibrium_hunter.bus, lambda: svc.bus]"
    assert kw["bus_getter"] == "_bus_for_rpc"
