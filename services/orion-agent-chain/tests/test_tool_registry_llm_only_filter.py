from __future__ import annotations

from pathlib import Path

from app.tool_registry import ToolRegistry


def test_tool_registry_exposes_reasoning_and_capability_verbs_but_not_raw_skills():
    repo_root = Path(__file__).resolve().parents[3]
    registry = ToolRegistry(base_dir=repo_root / "orion" / "cognition")

    tools = registry.tools_for_packs(["executive_pack", "memory_pack"])
    tool_ids = {tool.tool_id for tool in tools}

    assert "triage" in tool_ids
    assert "plan_action" in tool_ids
    assert "assess_runtime_state" in tool_ids
    assert "assess_mesh_presence" in tool_ids
    assert "assess_storage_health" in tool_ids
    assert "summarize_recent_changes" in tool_ids
    assert "housekeep_runtime" in tool_ids
    assert "recall" not in tool_ids
    assert "story_weave" not in tool_ids
    assert "skills.docker.ps_status.v1" not in tool_ids
    assert "skills.mesh.tailscale_mesh_status.v1" not in tool_ids
