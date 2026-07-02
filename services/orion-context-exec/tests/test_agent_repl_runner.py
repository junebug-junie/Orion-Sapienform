"""Tests for the agent_repl mode + runner dispatch."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

CTX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (str(REPO_ROOT), str(CTX_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def test_agent_repl_is_a_valid_mode():
    from orion.schemas.context_exec import ContextExecRequestV1

    req = ContextExecRequestV1(text="what does orion-hub do?", mode="agent_repl")
    assert req.mode == "agent_repl"


@pytest.mark.asyncio
async def test_llm_chat_route_forwards_messages_and_stop(monkeypatch):
    from app import llm_tools
    from orion.core.bus.bus_schemas import LLMMessage

    captured = {}

    class FakeCodec:
        def decode(self, data):
            class D:
                ok = True

                class envelope:
                    payload = {"content": "ok"}

            return D()

    class FakeBus:
        codec = FakeCodec()

        async def rpc_request(self, channel, env, *, reply_channel, timeout_sec):
            captured["payload"] = env.payload
            return {"data": b"x"}

    monkeypatch.setattr(llm_tools.settings, "orion_bus_enabled", True, raising=False)

    msgs = [
        LLMMessage(role="system", content="you are an agent"),
        LLMMessage(role="user", content="find the bug"),
    ]
    result = await llm_tools.llm_chat_route(
        FakeBus(),
        prompt="find the bug",
        route="agent",
        messages=msgs,
        stop=["<end_code>"],
    )
    assert result["ok"] is True
    payload = captured["payload"]
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    assert payload["options"]["stop"] == ["<end_code>"]


@pytest.mark.asyncio
async def test_organ_runtime_llm_chat_passes_messages_and_stop(monkeypatch):
    from app import organ_runtime as orm
    from orion.core.bus.bus_schemas import LLMMessage
    from orion.schemas.context_exec import ContextExecRequestV1

    seen = {}

    async def fake_route(bus, **kwargs):
        seen.update(kwargs)
        return {"ok": True, "content": "hi"}

    monkeypatch.setattr(orm.llm_tools, "llm_chat_route", fake_route)

    rt = orm.OrganRuntime(
        bus=object(),
        request=ContextExecRequestV1(text="q", mode="agent_repl"),
        run_id="r1",
        llm_route="agent",
    )
    msgs = [LLMMessage(role="user", content="q")]
    await rt.llm_chat("q", route="agent", messages=msgs, stop=["<end_code>"])
    assert seen["messages"] == msgs
    assert seen["stop"] == ["<end_code>"]
    assert seen["route"] == "agent"


@pytest.mark.asyncio
async def test_event_emitter_agent_step_publishes():
    from app.events import ContextExecEventEmitter

    published = []

    class FakeBus:
        async def publish(self, channel, env):
            published.append((channel, env.kind, env.payload))

    import app.events as ev
    ev.settings.orion_bus_enabled = True

    emitter = ContextExecEventEmitter(FakeBus(), correlation_id="corr-1")
    await emitter.agent_step(
        run_id="r1",
        mode="agent_repl",
        step_index=0,
        thought="I will grep the repo",
        tool_id="python_interpreter",
        tool_args="repo_grep('runtime')",
        observation="services/orion-hub/... matched",
        duration_ms=1234,
        is_final=False,
    )
    assert published, "no event published"
    channel, kind, payload = published[0]
    assert kind == "context.exec.agent_step.v1"
    assert payload["step_index"] == 0
    assert payload["tool_id"] == "python_interpreter"
    assert payload["correlation_id"] == "corr-1"


@pytest.mark.asyncio
async def test_run_agent_repl_returns_final_answer_as_final_text(monkeypatch):
    from app import runner as runner_mod
    from app.runner import ContextExecRunner
    from app.rlm_engine import RLMEngine
    from orion.schemas.context_exec import (
        ContextExecRequestV1,
        context_exec_permissions_for_llm_profile,
    )

    class StubEngine(RLMEngine):
        engine_name = "smolcode"

        async def run(self, request, namespace, *, organ_runtime=None,
                      step_callbacks=None, max_steps=None, per_step_timeout=None):
            # Simulate one emitted step via the callback, then a final answer.
            if step_callbacks:
                class _Step:
                    step_number = 0
                    is_final_answer = False
                    model_output = "let me look"
                    code_action = "repo_list('services')"
                    observations = "orion-hub/"
                    error = None

                    class timing:
                        duration = 0.5
                step_callbacks[0](_Step())
            return {"summary": "orion-hub is the operator UI + chat gateway.",
                    "engine": "smolcode", "mode": request.mode}

    r = ContextExecRunner(engine=StubEngine())

    async def fake_resolve(profile):
        from app.llm_profile_resolver import LLMProfileSelection
        return LLMProfileSelection(requested=profile, selected="agent", route_used="agent")

    monkeypatch.setattr(runner_mod, "resolve_llm_profile", fake_resolve)

    req = ContextExecRequestV1(
        text="what does orion-hub do?",
        mode="agent_repl",
        permissions=context_exec_permissions_for_llm_profile("agent"),
        llm_profile="agent",
    )
    run = await r.run(req)
    assert run.mode == "agent_repl"
    assert run.status == "ok"
    assert run.final_text == "orion-hub is the operator UI + chat gateway."
    # step callback populated the visible trace
    assert any(s.callable == "python_interpreter" for s in run.verb_trace)
