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
from orion.schemas.cortex.schemas import PlanExecutionArgs, PlanExecutionRequest


class _Bus:
    def __init__(self) -> None:
        self.published: list[tuple[str, BaseEnvelope]] = []

    async def publish(self, channel: str, env: BaseEnvelope) -> None:
        self.published.append((channel, env))


def test_legacy_plan_dream_cycle_publishes_dream_result_event(monkeypatch) -> None:
    bus = _Bus()
    corr = "11111111-1111-1111-1111-111111111456"
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
                            "verb_name": "dream_cycle",
                            "request_id": corr,
                            "status": "success",
                            "blocked": False,
                            "steps": [],
                            "mode": "brain",
                            "final_text": '{"tldr":"Moonlit archive","themes":["memory"],"symbols":{"moon":"continuity"},"narrative":"Moonlit archive."}',
                            "memory_used": True,
                            "recall_debug": {"profile": "dream.v1"},
                            "error": None,
                        }
                    },
                    request_id="req-dream",
                ),
            )
        ),
    )

    plan_payload = PlanExecutionRequest(
        plan={
            "verb_name": "dream_cycle",
            "steps": [],
        },
        args=PlanExecutionArgs(
            request_id="req-dream",
            extra={"mode": "standard"},
        ),
        context={
            "trace_id": "trace-dream",
            "metadata": {
                "dream_trigger": {
                    "mode": "standard",
                    "profile": "dream.v1",
                    "source": "test",
                },
                "dream_mode": "standard",
            },
        },
    )

    env = BaseEnvelope(
        kind="verb.request",
        source=ServiceRef(name="cortex-orch", version="0", node="n"),
        correlation_id=corr,
        reply_to=f"orion:verb:result:{corr}:req-dream",
        payload={
            "trigger": "legacy.plan",
            "schema_id": "PlanExecutionRequest",
            "payload": plan_payload.model_dump(mode="json"),
            "request_id": "req-dream",
            "caller": "cortex-orch",
            "meta": {"mode": "brain", "verb": "dream_cycle"},
        },
    )

    asyncio.run(exec_main.handle_verb_request(env))

    channels = [channel for channel, _ in bus.published]
    assert f"orion:verb:result:{corr}:req-dream" in channels
    assert "orion:verb:result" in channels
    assert exec_main.settings.channel_dream_log in channels

    dream_channel, dream_env = next(
        (channel, published_env)
        for channel, published_env in bus.published
        if channel == exec_main.settings.channel_dream_log
    )
    assert dream_channel == "orion:dream:log"
    assert dream_env.kind == "dream.result.v1"
    assert dream_env.payload["tldr"] == "Moonlit archive"
    assert dream_env.payload["narrative"] == "Moonlit archive."
