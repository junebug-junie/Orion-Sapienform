from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_ROOT) in sys.path:
    sys.path.remove(str(APP_ROOT))
sys.path.insert(0, str(APP_ROOT))
for module_name in [name for name in list(sys.modules) if name == "app" or name.startswith("app.")]:
    del sys.modules[module_name]

from app import main as exec_main
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.verbs.models import VerbResultV1
from orion.schemas.cortex.schemas import PlanExecutionArgs, PlanExecutionRequest, StepExecutionResult


class _Bus:
    def __init__(self) -> None:
        self.published: list[tuple[str, BaseEnvelope]] = []

    async def publish(self, channel: str, env: BaseEnvelope) -> None:
        self.published.append((channel, env))


def test_legacy_plan_verb_runtime_publishes_cognition_trace(monkeypatch) -> None:
    bus = _Bus()
    corr = "54e630da-3995-4c98-8faf-518b687f2160"
    monkeypatch.setattr(exec_main, "svc", SimpleNamespace(bus=bus))
    monkeypatch.setattr(
        exec_main,
        "verb_runtime",
        SimpleNamespace(
            handle_request=lambda *args, **kwargs: asyncio.sleep(
                0,
                result=VerbResultV1(
                    verb="legacy.plan",
                    ok=True,
                    output={
                        "result": {
                            "verb_name": "chat_general",
                            "request_id": corr,
                            "status": "success",
                            "blocked": False,
                            "steps": [
                                StepExecutionResult(
                                    status="success",
                                    verb_name="chat_general",
                                    step_name="collect_metacog_context",
                                    order=0,
                                    result={},
                                    latency_ms=10,
                                ).model_dump(mode="json"),
                                StepExecutionResult(
                                    status="success",
                                    verb_name="chat_general",
                                    step_name="llm_chat_general",
                                    order=1,
                                    result={},
                                    latency_ms=20,
                                ).model_dump(mode="json"),
                            ],
                            "mode": "brain",
                            "final_text": "hello from chat",
                            "memory_used": True,
                            "recall_debug": {"profile": "chat.general.v1"},
                            "error": None,
                        }
                    },
                    request_id="req-chat",
                ),
            )
        ),
    )

    plan_payload = PlanExecutionRequest(
        plan={
            "verb_name": "chat_general",
            "steps": [],
        },
        args=PlanExecutionArgs(
            request_id="req-chat",
            extra={"session_id": "sess-1", "message_id": "msg-1"},
        ),
        context={"trace_id": "trace-chat"},
    )

    env = BaseEnvelope(
        kind="verb.request",
        source=ServiceRef(name="cortex-orch", version="0", node="n"),
        correlation_id=corr,
        reply_to=f"orion:verb:result:{corr}:req-chat",
        payload={
            "trigger": "legacy.plan",
            "schema_id": "PlanExecutionRequest",
            "payload": plan_payload.model_dump(mode="json"),
            "request_id": "req-chat",
            "caller": "cortex-orch",
            "meta": {"mode": "brain", "verb": "chat_general"},
        },
    )

    asyncio.run(exec_main.handle_verb_request(env))

    channels = [channel for channel, _ in bus.published]
    assert exec_main.settings.channel_cognition_trace_pub in channels

    trace_channel, trace_env = next(
        (channel, published_env)
        for channel, published_env in bus.published
        if channel == exec_main.settings.channel_cognition_trace_pub
    )
    assert trace_channel == "orion:cognition:trace"
    assert trace_env.kind == "cognition.trace"
    assert trace_env.payload["correlation_id"] == corr
    assert trace_env.payload["verb"] == "chat_general"
    assert len(trace_env.payload["steps"]) == 2
    assert trace_env.payload["steps"][0]["step_name"] == "collect_metacog_context"
