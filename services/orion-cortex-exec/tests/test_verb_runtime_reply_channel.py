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


def _verb_env(corr: str) -> BaseEnvelope:
    return BaseEnvelope(
        kind="verb.request",
        source=ServiceRef(name="cortex-orch", version="0", node="n"),
        correlation_id=corr,
        reply_to=f"orion:verb:result:{corr}:req-9",
        trace={"trace_id": "t", "rpc_degraded": "echo-from-request"},
        payload={
            "trigger": "legacy.plan",
            "schema_id": "PlanExecutionRequest",
            "payload": {"plan": {"verb_name": "agent_runtime"}, "args": {"request_id": "req-9"}},
            "request_id": "req-9",
            "caller": "cortex-orch",
            "meta": {"mode": "agent"},
        },
    )


def _run_verb(monkeypatch, *, ok: bool, note: str | None) -> BaseEnvelope:
    from orion.core.bus.rpc_degraded import note_degraded

    bus = _Bus()
    monkeypatch.setattr(exec_main, "svc", SimpleNamespace(bus=bus))

    async def _handle(*args, **kwargs):
        if note:
            note_degraded(note)
        return VerbResultV1(verb="legacy.plan", ok=ok, output={"result": {"status": "success", "steps": []}},
                            request_id="req-9")

    monkeypatch.setattr(exec_main, "verb_runtime", SimpleNamespace(handle_request=_handle))
    corr = "11111111-1111-1111-1111-111111111124"
    asyncio.run(exec_main.handle_verb_request(_verb_env(corr)))
    return bus.published[0][1]


def test_verb_reply_is_stamped_degraded_when_handling_noted_it(monkeypatch) -> None:
    """2026-10-13: orion:verb:request is a Hunter (no chassis stamp), so the handler
    stamps its own verb.result -- and never echoes the request's marker."""
    from orion.core.bus.rpc_degraded import degraded_reason

    reply = _run_verb(monkeypatch, ok=True, note="llm_gateway:gpu_pool_unavailable")
    assert degraded_reason(reply) == "llm_gateway:gpu_pool_unavailable"
    # a verb that failed after real work is not degraded: its wait was real
    assert degraded_reason(_run_verb(monkeypatch, ok=False, note=None)) is None
    clean = _run_verb(monkeypatch, ok=True, note=None)
    assert degraded_reason(clean) is None and "rpc_degraded" not in clean.trace
    assert clean.trace.get("trace_id") == "t"
