import asyncio
from types import SimpleNamespace

from app.decision_router import DecisionRouter
from app.orchestrator import build_plan_request
from app import main as orch_main
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.verbs.models import VerbResultV1
from orion.schemas.cortex.contracts import CortexClientRequest
from orion.schemas.cortex.schemas import PlanExecutionRequest


def _req(mode: str = "auto", text: str = "please refactor this module") -> CortexClientRequest:
    return CortexClientRequest.model_validate(
        {
            "mode": mode,
            "route_intent": "auto",
            "packs": [],
            "options": {"route_intent": "auto"},
            "recall": {"enabled": True, "required": False, "mode": "hybrid", "profile": None},
            "context": {
                "messages": [{"role": "user", "content": text}],
                "raw_user_text": text,
                "user_message": text,
                "metadata": {},
            },
        }
    )


class _FakeBus:
    codec = None


def test_hub_auto_depth0_simple_question_heuristic():
    router = DecisionRouter(_FakeBus())
    req = _req(text="what time is it?")
    routed = asyncio.run(router.route(req, correlation_id="c1", source=ServiceRef(name="orch", version="0", node="n")))
    assert routed.decision.execution_depth == 0
    assert routed.request.verb == "chat_general"


def test_hub_auto_depth1_analysis_heuristic():
    router = DecisionRouter(_FakeBus())
    req = _req(text="analyze this text and extract intent")
    routed = asyncio.run(router.route(req, correlation_id="c2", source=ServiceRef(name="orch", version="0", node="n")))
    assert routed.decision.execution_depth == 1
    assert routed.request.verb == "analyze_text"


def test_hub_auto_depth2_engineering_heuristic_and_plan_shape():
    router = DecisionRouter(_FakeBus())
    req = _req(text="debug this docker compose stack trace and fix it")
    routed = asyncio.run(router.route(req, correlation_id="c3", source=ServiceRef(name="orch", version="0", node="n")))
    assert routed.decision.execution_depth == 2
    assert routed.request.verb == "agent_runtime"
    plan_req = build_plan_request(routed.request, "corr")
    assert [s.step_name for s in plan_req.plan.steps] == ["planner_react", "agent_chain"]


def test_build_plan_request_preserves_delivery_pack_in_args_extra():
    req = _req(mode="agent", text="write me a deployment runbook for this service")
    req.packs = ["executive_pack"]

    plan_req = build_plan_request(req, "corr-delivery")

    assert "delivery_pack" in plan_req.context["packs"]
    assert plan_req.args.extra["packs"] == plan_req.context["packs"]


def test_agent_mode_preserves_supervised_and_force_agent_chain_flags_in_exec_args():
    req = _req(mode="agent", text="generate code scaffolding for this feature")
    req.packs = ["executive_pack"]
    req.options["supervised"] = True
    req.options["force_agent_chain"] = True
    req.options["diagnostic"] = True

    plan_req = build_plan_request(req, "corr-agent-flags")

    assert plan_req.args.extra["supervised"] is True
    assert plan_req.args.extra["force_agent_chain"] is True
    assert plan_req.args.extra["diagnostic"] is True
    assert plan_req.context["metadata"]["output_mode_decision"]["output_mode"] == "code_delivery"
    assert "delivery_pack" in plan_req.context["packs"]


def test_auto_delivery_prompt_routes_to_agent_runtime_and_preserves_output_mode_signal():
    router = DecisionRouter(_FakeBus())
    req = _req(text="compare Docker Compose versus Kubernetes and recommend which to deploy")

    routed = asyncio.run(router.route(req, correlation_id="c-auto-delivery", source=ServiceRef(name="orch", version="0", node="n")))
    plan_req = build_plan_request(routed.request, "corr-auto-delivery", router_metadata=routed.decision.model_dump(mode="json"))

    assert routed.request.mode == "agent"
    assert routed.request.verb == "agent_runtime"
    assert routed.request.options["output_mode"] == "comparative_analysis"
    assert routed.request.options["response_profile"] == "reflective_depth"
    assert plan_req.args.extra["output_mode_decision"]["output_mode"] == "comparative_analysis"
    assert plan_req.context["metadata"]["output_mode_decision"]["response_profile"] == "reflective_depth"
    assert "delivery_pack" in plan_req.context["packs"]


def test_non_auto_introspect_spark_never_touches_router(monkeypatch):
    called = {"router": 0}

    class _NeverRouter:
        def __init__(self, *_a, **_k):
            called["router"] += 1

    async def _fake_call_verb_runtime(*args, **kwargs):
        req = kwargs["client_request"]
        return VerbResultV1(verb=req.verb or "unknown", ok=True, output={"result": {"status": "success", "steps": []}}, request_id="r")

    monkeypatch.setattr(orch_main, "DecisionRouter", _NeverRouter)
    monkeypatch.setattr(orch_main, "call_verb_runtime", _fake_call_verb_runtime)
    monkeypatch.setattr(orch_main, "svc", SimpleNamespace(bus=object()))
    monkeypatch.setattr(orch_main, "is_active", lambda *_args, **_kwargs: True)

    env = BaseEnvelope(
        kind="cortex.orch.request",
        source=ServiceRef(name="spark-introspector", version="0", node="n"),
        correlation_id="11111111-1111-1111-1111-111111111111",
        payload={
            "mode": "brain",
            "verb": "introspect_spark",
            "route_intent": "none",
            "packs": [],
            "options": {},
            "recall": {"enabled": False, "required": False, "mode": "hybrid", "profile": None},
            "context": {"messages": [{"role": "user", "content": "x"}], "raw_user_text": "x", "user_message": "x", "metadata": {}},
        },
    )
    asyncio.run(orch_main.handle(env))
    assert called["router"] == 0


def test_chat_general_personality_file_survives_plan_request_serialization():
    req = _req(mode="brain", text="hello")
    req.verb = "chat_general"
    plan_req = build_plan_request(req, "corr-personality")
    assert plan_req.plan.metadata.get("personality_file") == "orion/cognition/personality/orion_identity.yaml"

    serialized = plan_req.model_dump(mode="json")
    restored = PlanExecutionRequest.model_validate(serialized)
    assert restored.plan.metadata.get("personality_file") == "orion/cognition/personality/orion_identity.yaml"
