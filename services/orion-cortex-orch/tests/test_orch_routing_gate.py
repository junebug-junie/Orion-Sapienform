from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import asyncio
import pytest

from app import main as orch_main
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.verbs.models import VerbResultV1


def _env(*, source_name: str, mode: str, verb: str | None, route_intent: str = "none", text: str = "hi") -> BaseEnvelope:
    payload = {
        "mode": mode,
        "route_intent": route_intent,
        "verb": verb,
        "packs": [],
        "options": {"route_intent": route_intent} if route_intent != "none" else {},
        "recall": {"enabled": False, "required": False, "mode": "hybrid", "profile": None},
        "context": {
            "messages": [{"role": "user", "content": text}],
            "raw_user_text": text,
            "user_message": text,
            "metadata": {},
        },
    }
    return BaseEnvelope(
        kind="cortex.orch.request",
        source=ServiceRef(name=source_name, version="0", node="n"),
        correlation_id=str(uuid4()),
        payload=payload,
    )


def test_internal_brain_request_skips_auto_router(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"router": 0, "verb": None, "mode": None}

    async def _fake_call_verb_runtime(*args, **kwargs):
        req = kwargs["client_request"]
        called["verb"] = req.verb
        called["mode"] = req.mode
        return VerbResultV1(verb=req.verb or "unknown", ok=True, output={"result": {"status": "success", "steps": [], "final_text": "ok", "metadata": {}}}, request_id="r1")

    class _NeverRouter:
        def __init__(self, *_args, **_kwargs):
            called["router"] += 1

        async def route(self, *_args, **_kwargs):
            raise AssertionError("DecisionRouter should not be invoked for internal brain requests")

    monkeypatch.setattr(orch_main, "DecisionRouter", _NeverRouter)
    monkeypatch.setattr(orch_main, "call_verb_runtime", _fake_call_verb_runtime)
    monkeypatch.setattr(orch_main, "svc", SimpleNamespace(bus=object()))
    monkeypatch.setattr(orch_main, "is_active", lambda *_args, **_kwargs: True)

    res = asyncio.run(orch_main.handle(_env(source_name="spark-introspector", mode="brain", verb="introspect_spark", route_intent="auto")))
    assert res.payload.ok is True
    assert called["router"] == 0
    assert called["verb"] == "introspect_spark"


def test_mode_auto_backward_compat_allowlisted_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"router": 0}

    async def _fake_call_verb_runtime(*args, **kwargs):
        req = kwargs["client_request"]
        return VerbResultV1(verb=req.verb or "unknown", ok=True, output={"result": {"status": "success", "steps": [], "final_text": "ok", "metadata": {}}}, request_id="r3")

    class _FakeRouter:
        def __init__(self, *_args, **_kwargs):
            pass

        async def route(self, req, **_kwargs):
            calls["router"] += 1
            rewritten = req.model_copy(deep=True)
            rewritten.mode = "brain"
            rewritten.verb = "chat_general"
            rewritten.options["execution_depth"] = 0
            decision = {"execution_depth": 0, "primary_verb": None, "confidence": 0.7, "reason": "legacy_auto", "source": "heuristic"}
            return SimpleNamespace(request=rewritten, decision=SimpleNamespace(model_dump=lambda mode="json": decision, execution_depth=0, primary_verb=None, source="heuristic", confidence=0.7))

    monkeypatch.setattr(orch_main, "DecisionRouter", _FakeRouter)
    monkeypatch.setattr(orch_main, "call_verb_runtime", _fake_call_verb_runtime)
    monkeypatch.setattr(orch_main, "svc", SimpleNamespace(bus=object()))
    monkeypatch.setattr(orch_main, "is_active", lambda *_args, **_kwargs: True)

    asyncio.run(orch_main.handle(_env(source_name="cortex-gateway", mode="auto", verb=None, route_intent="none")))
    asyncio.run(orch_main.handle(_env(source_name="spark-introspector", mode="auto", verb=None, route_intent="none")))
    assert calls["router"] == 1


def test_auto_depth2_agent_runtime_not_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"executed": 0, "verb": None, "mode": None}

    async def _fake_call_verb_runtime(*args, **kwargs):
        req = kwargs["client_request"]
        calls["executed"] += 1
        calls["verb"] = req.verb
        calls["mode"] = req.mode
        return VerbResultV1(
            verb=req.verb or "unknown",
            ok=True,
            output={"result": {"status": "success", "steps": [], "final_text": "ok", "metadata": {}}},
            request_id="r-depth2",
        )

    class _Depth2Router:
        def __init__(self, *_args, **_kwargs):
            pass

        async def route(self, req, **_kwargs):
            rewritten = req.model_copy(deep=True)
            rewritten.mode = "agent"
            rewritten.verb = "agent_runtime"
            rewritten.options["execution_depth"] = 2
            decision = {
                "execution_depth": 2,
                "primary_verb": "agent_runtime",
                "confidence": 0.91,
                "reason": "deep_task",
                "source": "heuristic",
            }
            return SimpleNamespace(
                request=rewritten,
                decision=SimpleNamespace(
                    model_dump=lambda mode="json": decision,
                    execution_depth=2,
                    primary_verb="agent_runtime",
                    source="heuristic",
                    confidence=0.91,
                ),
            )

    monkeypatch.setattr(orch_main, "DecisionRouter", _Depth2Router)
    monkeypatch.setattr(orch_main, "call_verb_runtime", _fake_call_verb_runtime)
    monkeypatch.setattr(orch_main, "svc", SimpleNamespace(bus=object()))
    # Simulate restrictive activation list: runtime verb must still pass in agent mode
    monkeypatch.setattr(orch_main, "is_active", lambda *_args, **_kwargs: False)

    res = asyncio.run(orch_main.handle(_env(source_name="cortex-gateway", mode="auto", verb=None, route_intent="auto", text="build this feature")))
    assert res.payload.ok is True
    assert calls["executed"] == 1
    assert calls["mode"] == "agent"
    assert calls["verb"] == "agent_runtime"


def test_brain_yaml_verb_still_gated_when_inactive(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_call_verb_runtime(*args, **kwargs):
        raise AssertionError("call_verb_runtime should not be reached for inactive brain verb")

    monkeypatch.setattr(orch_main, "call_verb_runtime", _fake_call_verb_runtime)
    monkeypatch.setattr(orch_main, "svc", SimpleNamespace(bus=object()))
    monkeypatch.setattr(orch_main, "is_active", lambda *_args, **_kwargs: False)

    res = asyncio.run(orch_main.handle(_env(source_name="spark-introspector", mode="brain", verb="chat_general", route_intent="none", text="hello")))
    assert res.payload.ok is False
    assert res.payload.error and res.payload.error.get("message") == "inactive_verb:chat_general"


def test_timeout_terminalizes_with_fallback_verb(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _timeout(*args, **kwargs):
        raise TimeoutError("RPC timeout waiting on orion:verb:result")

    monkeypatch.setattr(orch_main, "call_verb_runtime", _timeout)
    monkeypatch.setattr(orch_main, "svc", SimpleNamespace(bus=object()))
    monkeypatch.setattr(orch_main, "is_active", lambda *_args, **_kwargs: True)

    res = asyncio.run(orch_main.handle(_env(source_name="spark-introspector", mode="brain", verb=None, route_intent="none", text="hello")))
    assert res.payload.ok is False
    assert res.payload.verb == "chat_general"
    assert res.payload.error and res.payload.error.get("category") == "timeout"
    assert res.payload.metadata and res.payload.metadata.get("orch_timeout_terminalized") is True
