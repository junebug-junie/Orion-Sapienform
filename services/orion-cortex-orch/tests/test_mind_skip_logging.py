from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

_guard = Path(__file__).resolve().parent / "_orch_import_guard.py"
_spec = importlib.util.spec_from_file_location("_orch_guard_mind_skip", _guard)
assert _spec and _spec.loader
_guard_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_guard_mod)

ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def _orch_prep() -> None:
    _guard_mod.ensure_orion_cortex_orch_app()


def test_call_verb_runtime_logs_and_marks_skip_when_mind_enabled_missing(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _orch_prep()
    from app.orchestrator import call_verb_runtime
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from orion.core.verbs.models import VerbResultV1
    from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, LLMMessage
    from orion.schemas.cortex.schemas import ExecutionPlan, ExecutionStep, PlanExecutionArgs, PlanExecutionRequest

    class _PubSub:
        def __init__(self, channel: str):
            self.channel = channel

    class _SubscribeCtx:
        def __init__(self, channel: str):
            self.pubsub = _PubSub(channel)

        async def __aenter__(self):
            return self.pubsub

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakeCodec:
        def decode(self, data):
            return SimpleNamespace(ok=True, envelope=data, error=None)

    class _FakeBus:
        def __init__(self):
            self.codec = _FakeCodec()
            self.published: list[tuple[str, BaseEnvelope]] = []

        def subscribe(self, channel: str):
            self.subscribed = channel
            return _SubscribeCtx(channel)

        async def publish(self, channel: str, env: BaseEnvelope):
            self.published.append((channel, env))

        async def iter_messages(self, pubsub: _PubSub):
            _channel, env = self.published[0]
            reply_env = BaseEnvelope(
                kind="verb.result",
                source=ServiceRef(name="cortex-exec", version="0", node="n"),
                correlation_id=env.correlation_id,
                payload=VerbResultV1(
                    verb="legacy.plan",
                    ok=True,
                    output={"result": {"status": "success", "final_text": "ok", "metadata": {}}},
                    request_id=env.payload["request_id"],
                ).model_dump(mode="json"),
            )
            yield {"data": reply_env}

    captured: dict = {}

    def _fake_build_plan_request(client_request, correlation_id, router_metadata=None):
        plan = PlanExecutionRequest(
            plan=ExecutionPlan(
                verb_name="chat_general",
                steps=[ExecutionStep(verb_name="chat_general", step_name="noop", order=0, services=[])],
            ),
            args=PlanExecutionArgs(request_id="trace-1"),
            context={"metadata": {}},
        )
        captured["plan"] = plan
        return plan

    monkeypatch.setattr("app.orchestrator.build_plan_request", _fake_build_plan_request)

    async def _no_state(*args, **kwargs):
        return None

    monkeypatch.setattr("app.orchestrator._maybe_fetch_state", _no_state)
    monkeypatch.setattr(
        "app.orchestrator.get_settings",
        lambda: SimpleNamespace(exec_lane_routing_enabled=False),
    )

    corr = str(uuid4())
    with caplog.at_level(logging.INFO, logger="orion.cortex.orch"):
        asyncio.run(
            call_verb_runtime(
                _FakeBus(),
                source=ServiceRef(name="cortex-orch", version="0", node="n"),
                client_request=CortexClientRequest(
                    verb="chat_general",
                    mode="brain",
                    context=CortexClientContext(
                        messages=[LLMMessage(role="user", content="hi")],
                        session_id="s",
                        trace_id="t",
                        user_message="hi",
                        metadata={},
                    ),
                ),
                correlation_id=corr,
                timeout_sec=5.0,
            )
        )

    meta = captured["plan"].context.get("metadata") or {}
    assert meta.get("mind_requested") is False
    assert meta.get("mind_skip_reason") == "mind_enabled_not_true"
    assert any(
        "mind_skipped" in r.message and "mind_enabled_not_true" in r.message and corr in r.message
        for r in caplog.records
    )


def test_call_verb_runtime_sets_mind_requested_when_mind_enabled_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _orch_prep()
    from app.orchestrator import call_verb_runtime
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from orion.core.verbs.models import VerbResultV1
    from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, LLMMessage
    from orion.schemas.cortex.schemas import ExecutionPlan, ExecutionStep, PlanExecutionArgs, PlanExecutionRequest

    class _PubSub:
        def __init__(self, channel: str):
            self.channel = channel

    class _SubscribeCtx:
        def __init__(self, channel: str):
            self.pubsub = _PubSub(channel)

        async def __aenter__(self):
            return self.pubsub

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakeCodec:
        def decode(self, data):
            return SimpleNamespace(ok=True, envelope=data, error=None)

    class _FakeBus:
        def __init__(self):
            self.codec = _FakeCodec()
            self.published: list[tuple[str, BaseEnvelope]] = []

        def subscribe(self, channel: str):
            return _SubscribeCtx(channel)

        async def publish(self, channel: str, env: BaseEnvelope):
            self.published.append((channel, env))

        async def iter_messages(self, pubsub: _PubSub):
            _channel, env = self.published[0]
            reply_env = BaseEnvelope(
                kind="verb.result",
                source=ServiceRef(name="cortex-exec", version="0", node="n"),
                correlation_id=env.correlation_id,
                payload=VerbResultV1(
                    verb="legacy.plan",
                    ok=True,
                    output={"result": {"status": "success", "final_text": "ok", "metadata": {}}},
                    request_id=env.payload["request_id"],
                ).model_dump(mode="json"),
            )
            yield {"data": reply_env}

    captured: dict = {}

    def _fake_build_plan_request(client_request, correlation_id, router_metadata=None):
        plan = PlanExecutionRequest(
            plan=ExecutionPlan(
                verb_name="chat_general",
                steps=[ExecutionStep(verb_name="chat_general", step_name="noop", order=0, services=[])],
            ),
            args=PlanExecutionArgs(request_id="trace-1"),
            context={"metadata": {}},
        )
        captured["plan"] = plan
        return plan

    async def _noop_prepare(*args, **kwargs):
        return None

    async def _fake_mind_http(*args, **kwargs):
        from orion.mind.v1 import MindRunResultV1

        return MindRunResultV1(mind_run_id=uuid4(), ok=True, mind_quality="contract_only")

    monkeypatch.setattr("app.orchestrator.build_plan_request", _fake_build_plan_request)
    monkeypatch.setattr("app.orchestrator._maybe_fetch_state", lambda *a, **k: asyncio.sleep(0))
    monkeypatch.setattr("app.mind_runtime.prepare_plan_context_for_mind_projection", _noop_prepare)
    monkeypatch.setattr("app.orchestrator.build_mind_run_request", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr("app.orchestrator.call_orion_mind_http", _fake_mind_http)
    monkeypatch.setattr("app.orchestrator.publish_mind_run_artifact", _noop_prepare)
    monkeypatch.setattr("app.orchestrator.merge_mind_brief_into_plan_metadata", lambda *a, **k: None)
    async def _no_substrate(*args, **kwargs):
        return None

    monkeypatch.setattr("app.orchestrator.fetch_substrate_telemetry_facet_for_mind", _no_substrate)
    monkeypatch.setattr(
        "app.orchestrator.get_settings",
        lambda: SimpleNamespace(exec_lane_routing_enabled=False),
    )

    asyncio.run(
        call_verb_runtime(
            _FakeBus(),
            source=ServiceRef(name="cortex-orch", version="0", node="n"),
            client_request=CortexClientRequest(
                verb="chat_general",
                mode="brain",
                context=CortexClientContext(
                    messages=[LLMMessage(role="user", content="hi")],
                    session_id="s",
                    trace_id="t",
                    user_message="hi",
                    metadata={"mind_enabled": True},
                ),
            ),
            correlation_id=str(uuid4()),
            timeout_sec=5.0,
        )
    )

    meta = captured["plan"].context.get("metadata") or {}
    assert meta.get("mind_requested") is True
    assert "mind_skip_reason" not in meta
