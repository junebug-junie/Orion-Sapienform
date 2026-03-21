from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import app.orchestrator as orchestrator
from app.orchestrator import build_plan_request, build_verb_request, call_verb_runtime
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.verbs.models import VerbResultV1
from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, LLMMessage, RecallDirective
from orion.schemas.cortex.schemas import ExecutionPlan, ExecutionStep, PlanExecutionArgs, PlanExecutionRequest


def _client_request() -> CortexClientRequest:
    return CortexClientRequest(
        mode="agent",
        route_intent="none",
        verb=None,
        packs=["executive_pack", "delivery_pack"],
        options={"supervised": True, "force_agent_chain": False},
        recall=RecallDirective(enabled=False, required=False, mode="hybrid", profile=None),
        context=CortexClientContext(
            messages=[LLMMessage(role="user", content="deploy Orion on Discord")],
            raw_user_text="deploy Orion on Discord",
            user_message="deploy Orion on Discord",
            session_id="sid-1",
            user_id="user-1",
            trace_id="trace-1",
            metadata={},
        ),
    )


def test_build_verb_request_uses_dedicated_reply_channel() -> None:
    source = ServiceRef(name="cortex-orch", version="0", node="n")
    req = _client_request()
    corr = "11111111-1111-1111-1111-111111111234"
    plan_req = PlanExecutionRequest(
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
            steps=[ExecutionStep(verb_name="agent_runtime", step_name="planner_react", order=0, services=["PlannerReactService"])],
            metadata={"mode": "agent"},
        ),
        args=PlanExecutionArgs(request_id="trace-1", extra={"mode": "agent"}),
        context={"mode": "agent", "packs": req.packs},
    )

    verb_req, env = build_verb_request(
        client_request=req,
        plan_request=plan_req,
        source=source,
        correlation_id=corr,
    )

    assert verb_req.request_id
    assert env.reply_to == f"orion:verb:result:{corr}:{verb_req.request_id}"
    assert env.kind == "verb.request"


def test_no_recall_agent_request_still_builds_planner_and_agent_chain_steps() -> None:
    plan_req = build_plan_request(_client_request(), "corr-no-recall-shape")

    assert plan_req.plan.verb_name == "agent_runtime"
    assert [step.step_name for step in plan_req.plan.steps] == ["planner_react", "agent_chain"]
    assert plan_req.args.extra["supervised"] is True
    assert plan_req.args.extra["recall"]["enabled"] is False


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
        channel, env = self.published[0]
        assert channel == "orion:verb:request"
        reply_channel = env.reply_to
        assert reply_channel == pubsub.channel
        reply_env = BaseEnvelope(
            kind="verb.result",
            source=ServiceRef(name="cortex-exec", version="0", node="n"),
            correlation_id=env.correlation_id,
            payload=VerbResultV1(
                verb="legacy.plan",
                ok=True,
                output={"result": {"status": "success", "steps": [], "final_text": "ok", "metadata": {}}},
                request_id=env.payload["request_id"],
            ).model_dump(mode="json"),
        )
        yield {"data": reply_env}


def test_call_verb_runtime_waits_on_dedicated_reply_channel(monkeypatch) -> None:
    source = ServiceRef(name="cortex-orch", version="0", node="n")
    bus = _FakeBus()
    corr = "11111111-1111-1111-1111-111111111345"

    monkeypatch.setattr(orchestrator, "build_plan_request", lambda *args, **kwargs: PlanExecutionRequest(
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
            steps=[ExecutionStep(verb_name="agent_runtime", step_name="planner_react", order=0, services=["PlannerReactService"])],
            metadata={"mode": "agent"},
        ),
        args=PlanExecutionArgs(request_id="trace-1", extra={"mode": "agent", "supervised": True}),
        context={"mode": "agent", "packs": ["executive_pack", "delivery_pack"], "output_mode": "implementation_guide", "response_profile": "technical_delivery"},
    ))

    async def _fake_maybe_fetch_state(*args, **kwargs):
        return None

    monkeypatch.setattr(orchestrator, "_maybe_fetch_state", _fake_maybe_fetch_state)

    result = asyncio.run(
        call_verb_runtime(
            bus,
            source=source,
            client_request=_client_request(),
            correlation_id=corr,
            timeout_sec=5.0,
        )
    )

    assert result.ok is True
    assert bus.subscribed.startswith(f"orion:verb:result:{corr}:")
