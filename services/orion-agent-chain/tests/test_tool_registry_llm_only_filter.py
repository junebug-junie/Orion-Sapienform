from __future__ import annotations

from pathlib import Path

from app.tool_registry import ToolRegistry


def test_tool_registry_filters_to_llm_only_verbs():
    repo_root = Path(__file__).resolve().parents[3]
    registry = ToolRegistry(base_dir=repo_root / "orion" / "cognition")

    tools = registry.tools_for_packs(["executive_pack", "memory_pack"])
    tool_ids = {tool.tool_id for tool in tools}

    assert "triage" in tool_ids
    assert "plan_action" in tool_ids
    assert "recall" not in tool_ids
    assert "story_weave" not in tool_ids
