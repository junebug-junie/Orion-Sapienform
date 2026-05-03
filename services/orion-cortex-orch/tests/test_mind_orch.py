from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

_guard = Path(__file__).resolve().parent / "_orch_import_guard.py"
_spec = importlib.util.spec_from_file_location("_orch_guard_boot", _guard)
assert _spec and _spec.loader
_guard_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_guard_mod)

import asyncio

ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def _orch_prep() -> None:
    _guard_mod.ensure_orion_cortex_orch_app()


from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.verbs.models import VerbResultV1
from orion.mind.v1 import MindHandoffBriefV1, MindRunResultV1
from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, LLMMessage, RecallDirective


def test_mind_enabled_calls_http_and_merges_metadata(monkeypatch) -> None:
    _orch_prep()
    import app.orchestrator as orchestrator
    from app.orchestrator import call_verb_runtime

    async def fake_maybe_fetch_state(*args, **kwargs):
        return None

    mr = MindRunResultV1(
        mind_run_id=uuid4(),
        ok=True,
        brief=MindHandoffBriefV1(
            machine_contract={"mind.route_kind": "brain"},
            stance_payload={
                "conversation_frame": "technical",
                "user_intent": "u",
                "self_relevance": "s",
                "juniper_relevance": "j",
                "answer_strategy": "a",
                "stance_summary": "st",
            },
        ),
    )
    called: dict = {}

    async def fake_call_mind(req):
        called["req"] = req
        return mr

    async def fake_publish(*args, **kwargs):
        called["publish"] = True

    monkeypatch.setattr(orchestrator, "_maybe_fetch_state", fake_maybe_fetch_state)
    monkeypatch.setattr(orchestrator, "call_orion_mind_http", fake_call_mind)
    monkeypatch.setattr(orchestrator, "publish_mind_run_artifact", fake_publish)

    class _PubSub:
        def __init__(self, channel: str) -> None:
            self.channel = channel

    class _SubCtx:
        def __init__(self, channel: str) -> None:
            self.pubsub = _PubSub(channel)

        async def __aenter__(self):
            return self.pubsub

        async def __aexit__(self, *a):
            return False

    class _FakeBus:
        def __init__(self) -> None:
            self.published: list = []
            self.codec = SimpleNamespace(decode=lambda d: SimpleNamespace(ok=True, envelope=d, error=None))

        def subscribe(self, channel: str):
            return _SubCtx(channel)

        async def publish(self, channel: str, env: BaseEnvelope) -> None:
            self.published.append((channel, env))

        async def iter_messages(self, pubsub: _PubSub):
            _ch, env = self.published[0]
            yield {"data": BaseEnvelope(
                kind="verb.result",
                source=ServiceRef(name="cortex-exec", version="0", node="n"),
                correlation_id=env.correlation_id,
                payload=VerbResultV1(
                    verb="legacy.plan",
                    ok=True,
                    output={"result": {"status": "success", "steps": [], "final_text": "ok", "metadata": {}}},
                    request_id=env.payload["request_id"],
                ).model_dump(mode="json"),
            )}

    req = CortexClientRequest(
        mode="brain",
        route_intent="none",
        verb="chat_general",
        packs=[],
        options={},
        recall=RecallDirective(enabled=False, required=False, mode="hybrid", profile=None),
        context=CortexClientContext(
            messages=[LLMMessage(role="user", content="hi")],
            raw_user_text="hi",
            user_message="hi",
            session_id="s1",
            user_id="u1",
            trace_id="t1",
            metadata={"mind_enabled": True},
        ),
    )

    bus = _FakeBus()
    source = ServiceRef(name="cortex-orch", version="0", node="n")
    captured: dict = {}
    _orig_build = orchestrator.build_verb_request

    def _capture_build_verb(**kwargs):
        plan_request = kwargs["plan_request"]
        captured["metadata"] = dict((plan_request.context or {}).get("metadata") or {})
        return _orig_build(**kwargs)

    monkeypatch.setattr(orchestrator, "build_verb_request", _capture_build_verb)
    asyncio.run(
        call_verb_runtime(
            bus,
            source=source,
            client_request=req,
            correlation_id="11111111-1111-1111-1111-111111111111",
            timeout_sec=30.0,
        )
    )
    assert "req" in called
    assert called.get("publish") is True
    assert called["req"].policy.n_loops_max >= 1
    meta = captured.get("metadata") or {}
    assert meta.get("mind_skip_stance_synthesis") is True
    assert meta.get("mind_run_ok") is True
    assert meta.get("mind.route_kind") == "brain"
    assert isinstance(meta.get("mind_handoff"), dict)


def test_mind_disabled_skips_http(monkeypatch) -> None:
    _orch_prep()
    import app.orchestrator as orchestrator
    from app.orchestrator import call_verb_runtime

    async def fake_maybe_fetch_state(*args, **kwargs):
        return None

    called: dict = {}

    async def fake_call_mind(req):
        called["mind"] = True
        raise AssertionError("mind HTTP should not be called")

    monkeypatch.setattr(orchestrator, "_maybe_fetch_state", fake_maybe_fetch_state)
    monkeypatch.setattr(orchestrator, "call_orion_mind_http", fake_call_mind)

    class _PubSub:
        def __init__(self, channel: str) -> None:
            self.channel = channel

    class _SubCtx:
        def __init__(self, channel: str) -> None:
            self.pubsub = _PubSub(channel)

        async def __aenter__(self):
            return self.pubsub

        async def __aexit__(self, *a):
            return False

    class _FakeBus:
        def __init__(self) -> None:
            self.published: list = []
            self.codec = SimpleNamespace(decode=lambda d: SimpleNamespace(ok=True, envelope=d, error=None))

        def subscribe(self, channel: str):
            return _SubCtx(channel)

        async def publish(self, channel: str, env: BaseEnvelope) -> None:
            self.published.append((channel, env))

        async def iter_messages(self, pubsub: _PubSub):
            _ch, env = self.published[0]
            yield {"data": BaseEnvelope(
                kind="verb.result",
                source=ServiceRef(name="cortex-exec", version="0", node="n"),
                correlation_id=env.correlation_id,
                payload=VerbResultV1(
                    verb="legacy.plan",
                    ok=True,
                    output={"result": {"status": "success", "steps": [], "final_text": "ok", "metadata": {}}},
                    request_id=env.payload["request_id"],
                ).model_dump(mode="json"),
            )}

    req = CortexClientRequest(
        mode="brain",
        route_intent="none",
        verb="chat_general",
        packs=[],
        options={},
        recall=RecallDirective(enabled=False, required=False, mode="hybrid", profile=None),
        context=CortexClientContext(
            messages=[LLMMessage(role="user", content="hi")],
            raw_user_text="hi",
            user_message="hi",
            session_id="s1",
            user_id="u1",
            trace_id="t1",
            metadata={"mind_enabled": False},
        ),
    )

    bus = _FakeBus()
    source = ServiceRef(name="cortex-orch", version="0", node="n")
    asyncio.run(
        call_verb_runtime(
            bus,
            source=source,
            client_request=req,
            correlation_id="22222222-2222-2222-2222-222222222222",
            timeout_sec=30.0,
        )
    )
    assert "mind" not in called


def test_merge_skips_stance_when_payload_invalid(monkeypatch) -> None:
    _orch_prep()
    import app.orchestrator as orchestrator
    from app.orchestrator import call_verb_runtime

    mr = MindRunResultV1(
        mind_run_id=uuid4(),
        ok=True,
        brief=MindHandoffBriefV1(
            machine_contract={"mind.route_kind": "brain"},
            stance_payload={"conversation_frame": "___invalid___"},
        ),
    )

    async def fake_maybe_fetch_state(*args, **kwargs):
        return None

    async def fake_call_mind(req):
        return mr

    async def fake_publish(*args, **kwargs):
        pass

    monkeypatch.setattr(orchestrator, "_maybe_fetch_state", fake_maybe_fetch_state)
    monkeypatch.setattr(orchestrator, "call_orion_mind_http", fake_call_mind)
    monkeypatch.setattr(orchestrator, "publish_mind_run_artifact", fake_publish)

    class _PubSub:
        def __init__(self, channel: str) -> None:
            self.channel = channel

    class _SubCtx:
        def __init__(self, channel: str) -> None:
            self.pubsub = _PubSub(channel)

        async def __aenter__(self):
            return self.pubsub

        async def __aexit__(self, *a):
            return False

    class _FakeBus:
        def __init__(self) -> None:
            self.published: list = []
            self.codec = SimpleNamespace(decode=lambda d: SimpleNamespace(ok=True, envelope=d, error=None))

        def subscribe(self, channel: str):
            return _SubCtx(channel)

        async def publish(self, channel: str, env: BaseEnvelope) -> None:
            self.published.append((channel, env))

        async def iter_messages(self, pubsub: _PubSub):
            _ch, env = self.published[0]
            yield {"data": BaseEnvelope(
                kind="verb.result",
                source=ServiceRef(name="cortex-exec", version="0", node="n"),
                correlation_id=env.correlation_id,
                payload=VerbResultV1(
                    verb="legacy.plan",
                    ok=True,
                    output={"result": {"status": "success", "steps": [], "final_text": "ok", "metadata": {}}},
                    request_id=env.payload["request_id"],
                ).model_dump(mode="json"),
            )}

    req = CortexClientRequest(
        mode="brain",
        route_intent="none",
        verb="chat_general",
        packs=[],
        options={},
        recall=RecallDirective(enabled=False, required=False, mode="hybrid", profile=None),
        context=CortexClientContext(
            messages=[LLMMessage(role="user", content="hi")],
            raw_user_text="hi",
            user_message="hi",
            session_id="s1",
            user_id="u1",
            trace_id="t1",
            metadata={"mind_enabled": True},
        ),
    )

    bus = _FakeBus()
    source = ServiceRef(name="cortex-orch", version="0", node="n")
    captured: dict = {}
    _orig_build = orchestrator.build_verb_request

    def _capture_build_verb(**kwargs):
        plan_request = kwargs["plan_request"]
        captured["metadata"] = dict((plan_request.context or {}).get("metadata") or {})
        return _orig_build(**kwargs)

    monkeypatch.setattr(orchestrator, "build_verb_request", _capture_build_verb)
    asyncio.run(
        call_verb_runtime(
            bus,
            source=source,
            client_request=req,
            correlation_id="33333333-3333-3333-3333-333333333333",
            timeout_sec=30.0,
        )
    )
    meta = captured.get("metadata") or {}
    assert meta.get("mind_stance_payload_invalid") is True
    assert meta.get("mind_skip_stance_synthesis") is False
