import asyncio
import json
from types import SimpleNamespace

import pytest

from app.decision_router import DecisionRouter, _instruction_runbook_heuristic
from app.orchestrator import build_plan_request
from app import main as orch_main
from orion.substrate import mutation_control_surface
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.verbs.models import VerbResultV1
from orion.schemas.cortex.contracts import AutoDepthDecisionV1, CortexClientRequest
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


def test_auto_router_clamps_short_instruction_question_to_agent_for_implementation_guide():
    """Short `?` heuristic is depth 0, but output_mode implementation_guide must still use planner+agent lane."""
    router = DecisionRouter(_FakeBus())
    req = _req(text="how do I configure redis for caching?")
    routed = asyncio.run(router.route(req, correlation_id="c-impl-short", source=ServiceRef(name="orch", version="0", node="n")))
    assert routed.request.options["output_mode"] == "implementation_guide"
    assert routed.decision.execution_depth == 2
    assert "output_mode_tool_lane" in routed.decision.reason
    assert routed.request.mode == "agent"
    assert routed.request.verb == "agent_runtime"


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


def test_auto_router_respects_live_routing_threshold_gate(monkeypatch):
    router = DecisionRouter(_FakeBus())
    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.95, actor="test_router_gate")
    monkeypatch.setattr(
        DecisionRouter,
        "heuristic_router",
        lambda self, req, shortlist: AutoDepthDecisionV1(
            execution_depth=2,
            primary_verb=None,
            confidence=0.85,
            reason="heuristic:engineering",
            source="heuristic",
        ),
    )
    req = _req(text="debug this docker compose stack trace and fix it")
    routed = asyncio.run(router.route(req, correlation_id="c-threshold-gate", source=ServiceRef(name="orch", version="0", node="n")))
    assert routed.request.verb == "chat_general"
    assert routed.decision.execution_depth == 0
    assert "routing_threshold_gate" in routed.decision.reason


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


def test_auto_router_treats_short_menu_followup_as_topic_selection_chat_general():
    router = DecisionRouter(_FakeBus())
    assistant_menu = (
        "Cool — let's align on a path. Here's how we might proceed:\n"
        "- Deep dive into one of the three axes (Adaptive Learning, Action System, or Mesh Continuity)\n"
        "- Explore a specific use case\n"
        "- Shift to a new focus"
    )
    req = CortexClientRequest.model_validate(
        {
            "mode": "auto",
            "route_intent": "auto",
            "packs": [],
            "options": {"route_intent": "auto"},
            "recall": {"enabled": True, "required": False, "mode": "hybrid", "profile": None},
            "context": {
                "messages": [
                    {"role": "assistant", "content": assistant_menu},
                    {"role": "user", "content": "hm mesh continuity"},
                ],
                "raw_user_text": "hm mesh continuity",
                "user_message": "hm mesh continuity",
                "metadata": {},
            },
        }
    )

    routed = asyncio.run(router.route(req, correlation_id="c-topic-followup", source=ServiceRef(name="orch", version="0", node="n")))

    assert routed.decision.execution_depth == 0
    assert routed.decision.reason == "heuristic:menu_topic_selection_followup"
    assert routed.request.mode == "brain"
    assert routed.request.verb == "chat_general"
    assert routed.request.options["menu_topic_selection"]["enabled"] is True
    assert routed.request.options["menu_topic_selection"]["selected_topic"] == "Mesh Continuity"


def test_auto_router_menu_followup_variants_preserve_chat_lane():
    router = DecisionRouter(_FakeBus())
    assistant_menu = (
        "Cool — let's align on a path. Here's how we might proceed:\n"
        "- Deep dive into one of the three axes (Adaptive Learning, Action System, or Mesh Continuity)\n"
        "- Explore a specific use case\n"
        "- Shift to a new focus"
    )
    variants = [
        "mesh continuity",
        "hmm, mesh continuity",
        "let's do mesh continuity",
        "adaptive learning",
        "action system",
        "the first one",
        "that second one",
        "sure, deep dive on mesh continuity",
    ]
    for idx, user_text in enumerate(variants):
        req = CortexClientRequest.model_validate(
            {
                "mode": "auto",
                "route_intent": "auto",
                "packs": [],
                "options": {"route_intent": "auto"},
                "recall": {"enabled": True, "required": False, "mode": "hybrid", "profile": None},
                "context": {
                    "messages": [
                        {"role": "assistant", "content": assistant_menu},
                        {"role": "user", "content": user_text},
                    ],
                    "raw_user_text": user_text,
                    "user_message": user_text,
                    "metadata": {},
                },
            }
        )
        routed = asyncio.run(router.route(req, correlation_id=f"c-topic-followup-{idx}", source=ServiceRef(name="orch", version="0", node="n")))
        assert routed.request.verb == "chat_general"
        assert routed.decision.reason == "heuristic:menu_topic_selection_followup"
        assert "hm_mesh_continuity" not in json.dumps(routed.request.model_dump(mode="json"))


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", False),
        ("how does it work", False),
        ("how do we ship", True),
        ("deploy to prod", True),
        ("walkthrough please", True),
    ],
)
def test_instruction_runbook_how_do_word_boundary(text: str, expected: bool) -> None:
    assert _instruction_runbook_heuristic(text) is expected


def test_auto_route_engineering_how_do_still_uses_agent_runtime():
    router = DecisionRouter(_FakeBus())
    req = _req(text="How do I fix this docker compose error?")
    routed = asyncio.run(router.route(req, correlation_id="c-howdo-docker", source=ServiceRef(name="orch", version="0", node="n")))
    assert routed.decision.execution_depth == 2
    assert routed.request.verb == "agent_runtime"


def test_auto_route_prune_containers_stays_depth0_chat_general():
    router = DecisionRouter(_FakeBus())
    req = _req(text="Prune stopped containers")
    routed = asyncio.run(router.route(req, correlation_id="c-prune", source=ServiceRef(name="orch", version="0", node="n")))
    assert routed.decision.execution_depth == 0
    assert routed.request.verb == "chat_general"


def test_auto_route_workflow_planner_agent_chain_goes_agent_runtime():
    router = DecisionRouter(_FakeBus())
    req = _req(text="Help me design a workflow for log triage with planner and agent chain")
    routed = asyncio.run(router.route(req, correlation_id="c-workflow", source=ServiceRef(name="orch", version="0", node="n")))
    assert routed.decision.execution_depth == 2
    assert routed.request.verb == "agent_runtime"
    assert routed.decision.reason in ("heuristic:engineering", "heuristic:planner_agent_chain_design")


def test_explicit_skill_verb_request_skips_auto_router(monkeypatch):
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
        source=ServiceRef(name="cortex-gateway", version="0", node="n"),
        correlation_id="22222222-2222-2222-2222-222222222222",
        payload={
            "mode": "brain",
            "verb": "skills.runtime.docker_prune_stopped_containers.v1",
            "route_intent": "none",
            "packs": [],
            "options": {},
            "recall": {"enabled": False, "required": False, "mode": "hybrid", "profile": None},
            "context": {
                "messages": [{"role": "user", "content": "Prune stopped containers"}],
                "raw_user_text": "Prune stopped containers",
                "user_message": "Prune stopped containers",
                "metadata": {},
            },
        },
    )
    asyncio.run(orch_main.handle(env))
    assert called["router"] == 0
