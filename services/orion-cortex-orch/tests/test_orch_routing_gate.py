from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import asyncio

import pytest

from app import main as orch_main
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.verbs.models import VerbResultV1


def _env(*, source_name: str, mode: str, verb: str | None, route_intent: str = "none") -> BaseEnvelope:
    payload = {
        "mode": mode,
        "route_intent": route_intent,
        "verb": verb,
        "packs": [],
        "options": {},
        "recall": {"enabled": False, "required": False, "mode": "hybrid", "profile": None},
        "context": {
            "messages": [{"role": "user", "content": "hi"}],
            "raw_user_text": "hi",
            "user_message": "hi",
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
        return VerbResultV1(
            verb=req.verb or "unknown",
            ok=True,
            output={"result": {"status": "success", "steps": [], "final_text": "ok", "metadata": {}}},
            request_id="r1",
        )

    class _NeverRouter:
        def __init__(self, *_args, **_kwargs):
            called["router"] += 1

        async def route(self, *_args, **_kwargs):
            raise AssertionError("DecisionRouter should not be invoked for internal brain requests")

    monkeypatch.setattr(orch_main, "DecisionRouter", _NeverRouter)
    monkeypatch.setattr(orch_main, "call_verb_runtime", _fake_call_verb_runtime)
    monkeypatch.setattr(orch_main, "svc", SimpleNamespace(bus=object()))
    monkeypatch.setattr(orch_main, "is_active", lambda *_args, **_kwargs: True)

    res = asyncio.run(orch_main.handle(
        _env(source_name="spark-introspector", mode="brain", verb="introspect_spark", route_intent="none")
    ))
    assert res.payload.ok is True
    assert called["router"] == 0
    assert called["verb"] == "introspect_spark"
    assert called["mode"] == "brain"


def test_hub_auto_route_rewrites_to_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {"mode": None, "verb": None}

    async def _fake_call_verb_runtime(*args, **kwargs):
        req = kwargs["client_request"]
        captured["mode"] = req.mode
        captured["verb"] = req.verb
        return VerbResultV1(
            verb=req.verb or "unknown",
            ok=True,
            output={"result": {"status": "success", "steps": [], "final_text": "ok", "metadata": {}}},
            request_id="r2",
        )

    class _FakeRouter:
        def __init__(self, *_args, **_kwargs):
            pass

        async def route(self, req, **_kwargs):
            rewritten = req.model_copy(deep=True)
            rewritten.mode = "agent"
            rewritten.verb = "agent_runtime"
            rewritten.packs = ["executive_pack"]
            decision = {
                "route_mode": "agent",
                "verb": "agent_runtime",
                "packs": ["executive_pack"],
                "recall": {"enabled": False, "required": False, "profile": None},
                "confidence": 0.9,
                "reason": "test",
                "source": "llm",
            }
            return SimpleNamespace(request=rewritten, decision=SimpleNamespace(model_dump=lambda mode="json": decision, source="llm", confidence=0.9))

    monkeypatch.setattr(orch_main, "DecisionRouter", _FakeRouter)
    monkeypatch.setattr(orch_main, "call_verb_runtime", _fake_call_verb_runtime)
    monkeypatch.setattr(orch_main, "svc", SimpleNamespace(bus=object()))
    monkeypatch.setattr(orch_main, "is_active", lambda *_args, **_kwargs: True)

    res = asyncio.run(orch_main.handle(
        _env(source_name="cortex-gateway", mode="brain", verb=None, route_intent="auto")
    ))
    assert res.payload.ok is True
    assert captured["mode"] == "agent"
    assert captured["verb"] == "agent_runtime"


def test_mode_auto_backward_compat_allowlisted_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"router": 0}

    async def _fake_call_verb_runtime(*args, **kwargs):
        req = kwargs["client_request"]
        return VerbResultV1(
            verb=req.verb or "unknown",
            ok=True,
            output={"result": {"status": "success", "steps": [], "final_text": "ok", "metadata": {}}},
            request_id="r3",
        )

    class _FakeRouter:
        def __init__(self, *_args, **_kwargs):
            pass

        async def route(self, req, **_kwargs):
            calls["router"] += 1
            rewritten = req.model_copy(deep=True)
            rewritten.mode = "brain"
            rewritten.verb = "chat_general"
            decision = {
                "route_mode": "chat",
                "verb": "chat_general",
                "packs": ["executive_pack"],
                "recall": {"enabled": False, "required": False, "profile": None},
                "confidence": 0.7,
                "reason": "legacy_auto",
                "source": "heuristic",
            }
            return SimpleNamespace(request=rewritten, decision=SimpleNamespace(model_dump=lambda mode="json": decision, source="heuristic", confidence=0.7))

    monkeypatch.setattr(orch_main, "DecisionRouter", _FakeRouter)
    monkeypatch.setattr(orch_main, "call_verb_runtime", _fake_call_verb_runtime)
    monkeypatch.setattr(orch_main, "svc", SimpleNamespace(bus=object()))
    monkeypatch.setattr(orch_main, "is_active", lambda *_args, **_kwargs: True)

    asyncio.run(orch_main.handle(_env(source_name="cortex-gateway", mode="auto", verb=None, route_intent="none")))
    asyncio.run(orch_main.handle(_env(source_name="spark-introspector", mode="auto", verb=None, route_intent="none")))

    assert calls["router"] == 1
