"""Runner integration tests for llm_profile → route binding."""

from __future__ import annotations

from typing import Any

import pytest

from app.llm_tools import llm_chat_route
from app.runner import ContextExecRunner
from app.rlm_engine import RLMEngine
from orion.schemas.context_exec import ContextExecRequestV1


@pytest.fixture(autouse=True)
def _disable_gateway_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.settings import settings as cfg

    monkeypatch.setattr(cfg, "context_exec_llm_gateway_url", "")
    monkeypatch.setattr(cfg, "context_exec_llm_profile_fallback_enabled", False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("chat", "chat"),
        ("quick", "quick"),
        ("agent", "agent"),
        ("metacog", "metacog"),
    ],
)
async def test_runner_runtime_debug_profile_route(profile: str, expected: str) -> None:
    runner = ContextExecRunner()
    req = ContextExecRequestV1(text="denver probe", mode="general_investigation", llm_profile=profile)
    run = await runner.run(req)
    dbg = run.runtime_debug
    assert dbg["llm_profile_requested"] == expected
    assert dbg["llm_profile_selected"] == expected
    assert dbg["route_used"] == expected
    assert dbg.get("fallback_used") is not True


@pytest.mark.asyncio
async def test_runner_runtime_debug_default_when_omitted() -> None:
    runner = ContextExecRunner()
    req = ContextExecRequestV1(text="denver probe", mode="general_investigation")
    run = await runner.run(req)
    assert run.runtime_debug["llm_profile_requested"] is None
    assert run.runtime_debug["llm_profile_selected"] == "chat"
    assert run.runtime_debug["route_used"] == "chat"


class _RouteProbeEngine(RLMEngine):
    engine_name = "probe"

    async def run(self, request, namespace, *, organ_runtime=None) -> Any:
        namespace.llm.subcall("route probe")
        namespace.set_local("report", {"summary": "probe", "mode": request.mode})
        namespace.FINAL_VAR("report")
        return namespace.get_final()


@pytest.mark.asyncio
async def test_engine_subcall_records_route_for_profile() -> None:
    runner = ContextExecRunner(engine=_RouteProbeEngine())
    req = ContextExecRequestV1(text="probe", mode="general_investigation", llm_profile="quick")
    run = await runner.run(req)
    assert run.status == "ok"
    assert run.runtime_debug["route_used"] == "quick"


@pytest.mark.asyncio
async def test_llm_chat_route_uses_route_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeBus:
        enabled = True

        class codec:
            @staticmethod
            def decode(data: bytes) -> Any:
                class _Decoded:
                    ok = True

                    class envelope:
                        payload = {"content": "ok", "role": "assistant"}

                return _Decoded()

        async def rpc_request(self, channel: str, env: Any, **kwargs: Any) -> dict[str, Any]:
            captured["channel"] = channel
            captured["route"] = env.payload.get("route")
            return {"data": b"{}"}

    result = await llm_chat_route(_FakeBus(), prompt="hi", route="metacog")
    assert captured["route"] == "metacog"
    assert result["route"] == "metacog"
    assert result["ok"] is True
