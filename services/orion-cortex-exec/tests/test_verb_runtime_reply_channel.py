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


class _Bus:
    def __init__(self) -> None:
        self.published: list[tuple[str, BaseEnvelope]] = []

    async def publish(self, channel: str, env: BaseEnvelope) -> None:
        self.published.append((channel, env))


def test_exec_verb_runtime_replies_to_dedicated_reply_channel(monkeypatch) -> None:
    bus = _Bus()
    corr = "11111111-1111-1111-1111-111111111123"
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
                    output={"result": {"status": "success", "steps": [], "final_text": "ok"}},
                    request_id="req-123",
                ),
            )
        ),
    )

    env = BaseEnvelope(
        kind="verb.request",
        source=ServiceRef(name="cortex-orch", version="0", node="n"),
        correlation_id=corr,
        reply_to=f"orion:verb:result:{corr}:req-123",
        payload={
            "trigger": "legacy.plan",
            "schema_id": "PlanExecutionRequest",
            "payload": {"plan": {"verb_name": "agent_runtime"}, "args": {"request_id": "req-123"}},
            "request_id": "req-123",
            "caller": "cortex-orch",
            "meta": {"mode": "agent"},
        },
    )

    asyncio.run(exec_main.handle_verb_request(env))

    channels = [channel for channel, _ in bus.published]
    assert channels[0] == f"orion:verb:result:{corr}:req-123"
    assert "orion:verb:result" in channels
