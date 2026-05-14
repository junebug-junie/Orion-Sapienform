from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest

import app.orchestrator as orchestrator
from app.settings import Settings


def _spark_req() -> CortexClientRequest:
    return CortexClientRequest(
        mode="brain",
        verb_name="introspect_spark",
        packs=[],
        options={"timeout_sec": 5.0},
        context=CortexClientContext(
            messages=[],
            raw_user_text="x",
            user_message="x",
            session_id="s",
            user_id="u",
            trace_id="trace-spark",
            metadata={},
        ),
    )


def _settings_routing_on() -> Settings:
    return Settings.model_construct(
        exec_lane_routing_enabled=True,
        channel_exec_request="orion:cortex:exec:request",
        channel_exec_request_chat="orion:cortex:exec:request:chat",
        channel_exec_request_spark="orion:cortex:exec:request:spark",
        channel_exec_request_background="orion:cortex:exec:request:background",
        channel_exec_result_prefix="orion:exec:result",
        orion_state_enabled=False,
    )


def _settings_routing_off() -> Settings:
    return Settings.model_construct(
        exec_lane_routing_enabled=False,
        channel_exec_request="orion:cortex:exec:request",
        channel_exec_request_chat="orion:cortex:exec:request:chat",
        channel_exec_request_spark="orion:cortex:exec:request:spark",
        channel_exec_request_background="orion:cortex:exec:request:background",
        channel_exec_result_prefix="orion:exec:result",
        orion_state_enabled=False,
    )


def test_exec_request_channel_for_lane_legacy_when_disabled() -> None:
    s = _settings_routing_off()
    assert s.exec_request_channel_for_lane("spark") == "orion:cortex:exec:request"
    assert s.exec_request_channel_for_lane("chat") == "orion:cortex:exec:request"


def test_exec_request_channel_for_lane_when_enabled() -> None:
    s = _settings_routing_on()
    assert s.exec_request_channel_for_lane("chat") == "orion:cortex:exec:request:chat"
    assert s.exec_request_channel_for_lane("spark") == "orion:cortex:exec:request:spark"
    assert s.exec_request_channel_for_lane("background") == "orion:cortex:exec:request:background"


def test_call_verb_runtime_uses_direct_exec_for_spark_when_routing_on(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_call_cortex_exec(_bus, *, exec_request_channel: str, plan_request=None, **kwargs):  # type: ignore[no-untyped-def]
        captured["exec_request_channel"] = exec_request_channel
        captured["plan_request"] = plan_request
        return {
            "status": "success",
            "verb_name": "introspect_spark",
            "final_text": "ok",
            "steps": [],
            "memory_used": False,
            "recall_debug": {},
            "metadata": {},
        }

    monkeypatch.setattr(orchestrator, "call_cortex_exec", fake_call_cortex_exec)
    monkeypatch.setattr(orchestrator, "get_settings", _settings_routing_on)

    async def _none(*_a, **_k):
        return None

    monkeypatch.setattr(orchestrator, "_maybe_fetch_state", _none)

    bus = MagicMock()
    from orion.core.bus.bus_schemas import ServiceRef

    async def _run() -> None:
        res = await orchestrator.call_verb_runtime(
            bus,
            source=ServiceRef(name="cortex-orch", version="0", node="n"),
            client_request=_spark_req(),
            correlation_id="corr-1",
            timeout_sec=5.0,
        )
        assert captured.get("exec_request_channel") == "orion:cortex:exec:request:spark"
        pr = captured.get("plan_request")
        assert pr is not None, "direct exec must pass the enriched PlanExecutionRequest"
        ctx = getattr(pr, "context", None) or {}
        meta = ctx.get("metadata") if isinstance(ctx.get("metadata"), dict) else {}
        assert meta.get("execution_lane") == "spark"
        assert res.ok is True
        assert res.verb == "legacy.plan"
        assert isinstance(res.request_id, str) and len(res.request_id) == 36

    asyncio.run(_run())


def test_direct_exec_passes_router_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: direct lane RPC must reuse the same plan as the verb path (router + lane metadata)."""
    captured: dict[str, object] = {}

    async def fake_call_cortex_exec(_bus, *, exec_request_channel: str, plan_request=None, **kwargs):  # type: ignore[no-untyped-def]
        captured["exec_request_channel"] = exec_request_channel
        captured["plan_request"] = plan_request
        return {
            "status": "success",
            "verb_name": "introspect_spark",
            "final_text": "ok",
            "steps": [],
            "memory_used": False,
            "recall_debug": {},
            "metadata": {},
        }

    monkeypatch.setattr(orchestrator, "call_cortex_exec", fake_call_cortex_exec)
    monkeypatch.setattr(orchestrator, "get_settings", _settings_routing_on)

    async def _none(*_a, **_k):
        return None

    monkeypatch.setattr(orchestrator, "_maybe_fetch_state", _none)

    bus = MagicMock()
    from orion.core.bus.bus_schemas import ServiceRef

    router_meta = {"picked_model": "test-router-marker"}

    async def _run() -> None:
        await orchestrator.call_verb_runtime(
            bus,
            source=ServiceRef(name="cortex-orch", version="0", node="n"),
            client_request=_spark_req(),
            correlation_id="corr-router",
            timeout_sec=5.0,
            router_metadata=router_meta,
        )
        pr = captured["plan_request"]
        assert pr is not None
        ctx = getattr(pr, "context", None) or {}
        meta = ctx.get("metadata") if isinstance(ctx.get("metadata"), dict) else {}
        assert meta.get("auto_route") == router_meta
        assert meta.get("execution_lane") == "spark"

    asyncio.run(_run())


def test_exec_request_channel_for_lane_unknown_warns(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    caplog.set_level(logging.WARNING, logger="orion.cortex.orch.settings")
    s = _settings_routing_on()
    assert s.exec_request_channel_for_lane("typo_lane") == "orion:cortex:exec:request:background"
    assert any("exec_request_channel_for_lane_unknown" in r.message for r in caplog.records)
