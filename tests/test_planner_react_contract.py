from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import asyncio

import pytest

from orion.core.bus.bus_schemas import LLMMessage
from orion.schemas.agents.schemas import ContextBlock, Goal, PlannerRequest, ToolDef


def _load_planner_api_module():
    root = Path(__file__).resolve().parents[1]
    app_dir = root / "services" / "orion-planner-react" / "app"

    pkg = types.ModuleType("orion_planner_react")
    pkg.__path__ = [str(app_dir.parent)]
    subpkg = types.ModuleType("orion_planner_react.app")
    subpkg.__path__ = [str(app_dir)]
    sys.modules.setdefault("orion_planner_react", pkg)
    sys.modules.setdefault("orion_planner_react.app", subpkg)

    settings_spec = importlib.util.spec_from_file_location("orion_planner_react.app.settings", app_dir / "settings.py")
    settings_mod = importlib.util.module_from_spec(settings_spec)
    assert settings_spec and settings_spec.loader
    settings_spec.loader.exec_module(settings_mod)
    sys.modules["orion_planner_react.app.settings"] = settings_mod

    api_spec = importlib.util.spec_from_file_location("orion_planner_react.app.api", app_dir / "api.py")
    api_mod = importlib.util.module_from_spec(api_spec)
    assert api_spec and api_spec.loader
    api_spec.loader.exec_module(api_mod)
    return api_mod


api = _load_planner_api_module()


def test_delegate_mode_repairs_missing_action_with_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_call_planner_llm(**_: object) -> dict:
        return {"thought": "x", "finish": False, "action": None, "final_answer": None}

    async def fake_connect(self) -> None:  # noqa: ANN001
        return None

    async def fake_close(self) -> None:  # noqa: ANN001
        return None

    async def fake_rpc(self, *args, **kwargs):  # noqa: ANN001
        return {"data": b""}

    monkeypatch.setattr(api, "_call_planner_llm", fake_call_planner_llm)
    monkeypatch.setattr(api.OrionBusAsync, "connect", fake_connect)
    monkeypatch.setattr(api.OrionBusAsync, "close", fake_close)
    monkeypatch.setattr(api.OrionBusAsync, "rpc_request", fake_rpc)

    req = PlannerRequest(
        goal=Goal(description="Help me decide next step"),
        context=ContextBlock(conversation_history=[LLMMessage(role="user", content="Need help now")]),
        toolset=[
            ToolDef(tool_id="triage", description="triage", input_schema={"properties": {"text": {"type": "string"}}}),
            ToolDef(tool_id="plan_action", description="plan", input_schema={"properties": {"goal": {"type": "string"}}}),
            ToolDef(tool_id="analyze_text", description="analyze", input_schema={"properties": {"text": {"type": "string"}}}),
        ],
        preferences={"delegate_tool_execution": True},
    )

    res = asyncio.run(api.run_react_loop(req))

    assert res.stop_reason == "delegate"
    assert res.trace
    assert res.trace[-1].action is not None
    assert res.trace[-1].action["tool_id"] in {"triage", "plan_action", "analyze_text"}


def test_delegate_mode_falls_back_when_planner_call_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_call_planner_llm(**_: object) -> dict:
        raise RuntimeError("Planner LLM returned non-JSON: '{\"thought\":\"x\"'")

    async def fake_connect(self) -> None:  # noqa: ANN001
        return None

    async def fake_close(self) -> None:  # noqa: ANN001
        return None

    async def fake_rpc(self, *args, **kwargs):  # noqa: ANN001
        return {"data": b""}

    monkeypatch.setattr(api, "_call_planner_llm", fake_call_planner_llm)
    monkeypatch.setattr(api.OrionBusAsync, "connect", fake_connect)
    monkeypatch.setattr(api.OrionBusAsync, "close", fake_close)
    monkeypatch.setattr(api.OrionBusAsync, "rpc_request", fake_rpc)

    req = PlannerRequest(
        goal=Goal(description="Triage this"),
        context=ContextBlock(conversation_history=[LLMMessage(role="user", content="Need triage")]),
        toolset=[ToolDef(tool_id="triage", description="triage", input_schema={})],
        preferences={"delegate_tool_execution": True},
    )

    res = asyncio.run(api.run_react_loop(req))

    assert res.status == "ok"
    assert res.stop_reason == "delegate"
    assert res.trace[-1].action is not None
    assert res.trace[-1].action["tool_id"] == "triage"
