import asyncio

import pytest

from app.decision_router import DecisionRouter
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.cortex.contracts import CortexClientRequest


def _req(text: str, **options) -> CortexClientRequest:
    opts = {"route_intent": "auto", **options}
    return CortexClientRequest.model_validate(
        {
            "mode": "auto",
            "route_intent": "auto",
            "packs": [],
            "options": opts,
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


def test_belief_provenance_routes_context_exec():
    router = DecisionRouter(_FakeBus())
    req = _req("Where did the Denver claim come from?")
    routed = asyncio.run(router.route(req, correlation_id="c-belief", source=ServiceRef(name="orch", version="0", node="n")))
    assert routed.decision.execution_depth == 2
    assert routed.request.options.get("agent_runtime_engine") == "context_exec"
    assert routed.request.options.get("context_exec_mode") == "belief_provenance"


def test_trace_autopsy_routes_context_exec():
    router = DecisionRouter(_FakeBus())
    req = _req("Why did corr 123 fail open?")
    routed = asyncio.run(router.route(req, correlation_id="c-trace", source=ServiceRef(name="orch", version="0", node="n")))
    assert routed.decision.execution_depth == 2
    assert routed.request.options.get("agent_runtime_engine") == "context_exec"
    assert routed.request.options.get("context_exec_mode") == "trace_autopsy"


def test_repo_grounded_answer_contract_routes_context_exec():
    router = DecisionRouter(_FakeBus())
    req = _req(
        "Ground this in my repo",
        answer_contract={
            "request_kind": "repo_technical",
            "requires_repo_grounding": True,
            "allow_unverified_specifics": False,
            "preferred_render_style": "answer",
        },
    )
    routed = asyncio.run(router.route(req, correlation_id="c-repo", source=ServiceRef(name="orch", version="0", node="n")))
    assert routed.decision.execution_depth == 2
    assert routed.request.options.get("agent_runtime_engine") == "context_exec"


def test_simple_chat_no_context_exec():
    router = DecisionRouter(_FakeBus())
    req = _req("what time is it?")
    routed = asyncio.run(router.route(req, correlation_id="c-simple", source=ServiceRef(name="orch", version="0", node="n")))
    assert routed.decision.execution_depth == 0
    assert routed.request.options.get("agent_runtime_engine") is None
