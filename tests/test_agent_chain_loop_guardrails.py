from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from uuid import uuid4

from orion.schemas.agents.schemas import AgentChainRequest


def _load_agent_api_module():
    root = Path(__file__).resolve().parents[1]
    app_dir = root / "services" / "orion-agent-chain" / "app"

    pkg = types.ModuleType("orion_agent_chain")
    pkg.__path__ = [str(app_dir.parent)]
    subpkg = types.ModuleType("orion_agent_chain.app")
    subpkg.__path__ = [str(app_dir)]
    sys.modules.setdefault("orion_agent_chain", pkg)
    sys.modules.setdefault("orion_agent_chain.app", subpkg)

    settings_spec = importlib.util.spec_from_file_location("orion_agent_chain.app.settings", app_dir / "settings.py")
    settings_mod = importlib.util.module_from_spec(settings_spec)
    assert settings_spec and settings_spec.loader
    settings_spec.loader.exec_module(settings_mod)
    sys.modules["orion_agent_chain.app.settings"] = settings_mod

    rpc_spec = importlib.util.spec_from_file_location("orion_agent_chain.app.planner_rpc", app_dir / "planner_rpc.py")
    rpc_mod = importlib.util.module_from_spec(rpc_spec)
    assert rpc_spec and rpc_spec.loader
    rpc_spec.loader.exec_module(rpc_mod)
    sys.modules["orion_agent_chain.app.planner_rpc"] = rpc_mod

    exec_spec = importlib.util.spec_from_file_location("orion_agent_chain.app.tool_executor", app_dir / "tool_executor.py")
    exec_mod = importlib.util.module_from_spec(exec_spec)
    assert exec_spec and exec_spec.loader
    exec_spec.loader.exec_module(exec_mod)
    sys.modules["orion_agent_chain.app.tool_executor"] = exec_mod

    reg_spec = importlib.util.spec_from_file_location("orion_agent_chain.app.tool_registry", app_dir / "tool_registry.py")
    reg_mod = importlib.util.module_from_spec(reg_spec)
    assert reg_spec and reg_spec.loader
    reg_spec.loader.exec_module(reg_mod)
    sys.modules["orion_agent_chain.app.tool_registry"] = reg_mod

    api_spec = importlib.util.spec_from_file_location("orion_agent_chain.app.api", app_dir / "api.py")
    api_mod = importlib.util.module_from_spec(api_spec)
    assert api_spec and api_spec.loader
    api_spec.loader.exec_module(api_mod)
    return api_mod


api = _load_agent_api_module()


class _FakeToolExecutor:
    def __init__(self, *_args, **_kwargs):
        self.calls: list[str] = []

    async def execute_llm_verb(self, tool_id, tool_input, *, parent_correlation_id=None):
        self.calls.append(tool_id)
        return {"llm_output": f"obs for {tool_id}"}


def test_agent_chain_triage_runs_once_then_overrides_repeat(monkeypatch):
    calls = {"planner": 0}
    fake_exec = _FakeToolExecutor()

    async def _fake_planner(payload, *, parent_correlation_id=None, rpc_bus=None):
        calls["planner"] += 1
        if calls["planner"] == 1:
            return {
                "status": "ok",
                "trace": [{"step_index": 0, "thought": "triage", "action": {"tool_id": "triage", "input": {"text": "x"}}}],
            }
        if calls["planner"] == 2:
            tool_ids = [t.get("tool_id") for t in payload.get("toolset", [])]
            assert "triage" not in tool_ids
            return {
                "status": "ok",
                "trace": [{"step_index": 1, "thought": "triage again", "action": {"tool_id": "triage", "input": {"text": "x"}}}],
            }
        return {"status": "ok", "final_answer": {"content": "done", "structured": {}}}

    monkeypatch.setattr(api, "call_planner_react", _fake_planner)
    monkeypatch.setattr(api, "ToolExecutor", lambda *_a, **_k: fake_exec)
    monkeypatch.setattr(
        api,
        "_resolve_tools",
        lambda _body, output_mode=None: (
            [api.ToolDef(tool_id="triage", description="t", input_schema={}), api.ToolDef(tool_id="analyze_text", description="a", input_schema={})],
            [],
        ),
    )

    req = AgentChainRequest(text="hello", mode="agent", messages=[{"role": "user", "content": "hello"}])
    out = asyncio.run(api.execute_agent_chain(req, correlation_id=str(uuid4()), rpc_bus=object()))

    assert out.text == "done"
    assert fake_exec.calls == ["triage", "analyze_text"]


def test_agent_chain_returns_best_effort_at_step_cap(monkeypatch):
    fake_exec = _FakeToolExecutor()

    async def _fake_planner(payload, *, parent_correlation_id=None, rpc_bus=None):
        return {
            "status": "ok",
            "trace": [{"step_index": 0, "thought": "still working", "action": {"tool_id": "analyze_text", "input": {"text": "x"}}}],
        }

    monkeypatch.setattr(api, "call_planner_react", _fake_planner)
    monkeypatch.setattr(api, "ToolExecutor", lambda *_a, **_k: fake_exec)
    monkeypatch.setattr(api.settings, "default_max_steps", 2)
    monkeypatch.setattr(
        api,
        "_resolve_tools",
        lambda _body, output_mode=None: ([api.ToolDef(tool_id="analyze_text", description="a", input_schema={})], []),
    )

    req = AgentChainRequest(text="hello", mode="agent", messages=[{"role": "user", "content": "hello"}])
    out = asyncio.run(api.execute_agent_chain(req, correlation_id=str(uuid4()), rpc_bus=object()))

    assert "obs for finalize_response" in out.text or "obs for analyze_text" in out.text
