from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from app import api as agent_api
from app.tool_registry import ToolRegistry
from orion.schemas.agents.schemas import AgentChainRequest


def test_execute_agent_chain_sends_capability_backed_semantic_verbs_to_planner(monkeypatch):
    captured = {"toolsets": []}
    repo_root = Path(__file__).resolve().parents[3]

    async def _fake_planner(payload, *, parent_correlation_id=None, rpc_bus=None):
        captured["toolsets"].append([t.get("tool_id") for t in (payload.get("toolset") or [])])
        return {"status": "ok", "final_answer": {"content": "done", "structured": {}}}

    monkeypatch.setattr(agent_api, "call_planner_react", _fake_planner)
    monkeypatch.setattr(agent_api, "TOOL_REGISTRY", ToolRegistry(base_dir=repo_root / "orion" / "cognition"))

    class _NoopToolExecutor:
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(agent_api, "ToolExecutor", _NoopToolExecutor)

    req = AgentChainRequest(text="check runtime", mode="agent", messages=[{"role": "user", "content": "check runtime"}])
    out = asyncio.run(agent_api.execute_agent_chain(req, correlation_id=str(uuid4()), rpc_bus=object()))

    assert out.text == "done"
    assert captured["toolsets"], "planner should receive at least one toolset payload"
    first_toolset = set(captured["toolsets"][0])
    assert "assess_mesh_presence" in first_toolset
    assert "assess_storage_health" in first_toolset
    assert "summarize_recent_changes" in first_toolset
    assert "housekeep_runtime" in first_toolset
    assert "skills.mesh.tailscale_mesh_status.v1" not in first_toolset
    assert all(not tool_id.startswith("skills.") for tool_id in first_toolset)
