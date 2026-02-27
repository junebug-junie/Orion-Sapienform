import asyncio
import json
from types import SimpleNamespace

from app.decision_router import DecisionRouter
from app.orchestrator import build_plan_request
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.cortex.contracts import CortexClientRequest


def _req(mode: str = "auto", text: str = "please refactor this module") -> CortexClientRequest:
    return CortexClientRequest.model_validate(
        {
            "mode": mode,
            "packs": [],
            "options": {},
            "recall": {"enabled": True, "required": False, "mode": "hybrid", "profile": None},
            "context": {
                "messages": [{"role": "user", "content": text}],
                "raw_user_text": text,
                "user_message": text,
                "metadata": {},
            },
        }
    )


class _FakeCodec:
    def decode(self, data):
        return SimpleNamespace(ok=True, envelope=BaseEnvelope.model_validate(data))


class _FakeBus:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.codec = _FakeCodec()

    async def rpc_request(self, channel, env, reply_channel, timeout_sec):
        return {
            "data": {
                "kind": "llm.chat.result",
                "source": ServiceRef(name="llm", version="0", node="n").model_dump(mode="json"),
                "correlation_id": env.correlation_id,
                "payload": {"content": self.response_text},
            }
        }


def test_auto_mode_rewritten_by_heuristic():
    bus = _FakeBus("{}")
    router = DecisionRouter(bus)
    req = _req(text="please implement and run tests")
    routed = asyncio.run(router.route(req, correlation_id="c1", source=ServiceRef(name="orch", version="0", node="n")))
    assert routed.request.mode == "agent"
    assert routed.request.verb == "agent_runtime"
    assert routed.decision.source in {"heuristic", "fallback"}


def test_llm_router_output_is_clamped():
    bad_decision = json.dumps(
        {
            "route_mode": "chat",
            "verb": "chat_general",
            "packs": ["unknown_pack"],
            "recall": {"enabled": True, "required": True, "profile": "x"},
            "confidence": 1.7,
            "reason": "bad",
            "source": "llm",
        }
    )
    bus = _FakeBus(bad_decision)
    router = DecisionRouter(bus)
    router.settings.auto_router_llm_enabled = True
    req = _req(text="hello there")
    routed = asyncio.run(router.route(req, correlation_id="11111111-1111-1111-1111-111111111111", source=ServiceRef(name="orch", version="0", node="n")))
    assert routed.request.mode == "brain"
    assert routed.request.verb == "chat_general"
    assert routed.request.packs == ["executive_pack"]
    assert routed.decision.confidence == 1.0
    assert routed.decision.source == "llm"


def test_build_plan_request_carries_router_metadata():
    req = _req(mode="agent", text="run build")
    req.verb = "agent_runtime"
    meta = {"route_mode": "chat", "source": "heuristic"}
    plan_req = build_plan_request(req, "corr-1", router_metadata=meta)
    assert plan_req.context["metadata"]["auto_route"] == meta
    assert plan_req.plan.metadata["auto_route"] == meta
