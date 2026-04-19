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
    assert all(not tid.startswith("skills.") for tid in tool_ids)


def test_capability_backed_semantic_verbs_publish_bridge_metadata():
    repo_root = Path(__file__).resolve().parents[3]
    registry = ToolRegistry(base_dir=repo_root / "orion" / "cognition")

    tools = {tool.tool_id: tool for tool in registry.tools_for_packs(["executive_pack"])}
    expected = {
        "assess_mesh_presence": ("mesh_presence", "none"),
        "assess_storage_health": ("storage_health", "none"),
        "summarize_recent_changes": ("repo_change_intel", "none"),
        "housekeep_runtime": ("runtime_housekeeping", "low"),
    }
    for verb, (family, side_effect_level) in expected.items():
        assert verb in tools
        tool = tools[verb]
        assert tool.execution_mode == "capability_backed"
        assert tool.requires_capability_selector is True
        assert family in (tool.preferred_skill_families or [])
        assert tool.side_effect_level == side_effect_level


def test_requires_capability_selector_alone_keeps_semantic_verb_planner_visible(tmp_path):
    base = tmp_path / "cognition"
    packs_dir = base / "packs"
    verbs_dir = base / "verbs"
    packs_dir.mkdir(parents=True)
    verbs_dir.mkdir(parents=True)

    (packs_dir / "executive_pack.yaml").write_text(
        "\n".join(
            [
                "name: executive_pack",
                "label: Exec",
                "description: test",
                "verbs:",
                "  - selector_only_capability",
            ]
        ),
        encoding="utf-8",
    )
    (verbs_dir / "selector_only_capability.yaml").write_text(
        "\n".join(
            [
                "name: selector_only_capability",
                "description: test capability semantic verb",
                "services: []",
                "requires_capability_selector: true",
                "preferred_skill_families:",
                "  - runtime_housekeeping",
            ]
        ),
        encoding="utf-8",
    )

    registry = ToolRegistry(base_dir=base)
    tools = registry.tools_for_packs(["executive_pack"])
    assert [tool.tool_id for tool in tools] == ["selector_only_capability"]
    assert tools[0].execution_mode == "capability_backed"


def test_registry_validation_blocks_capability_backed_llm_service_binding(tmp_path):
    base = tmp_path / "cognition"
    packs_dir = base / "packs"
    verbs_dir = base / "verbs"
    packs_dir.mkdir(parents=True)
    verbs_dir.mkdir(parents=True)

    (packs_dir / "executive_pack.yaml").write_text(
        "\n".join(
            [
                "name: executive_pack",
                "label: Exec",
                "description: test",
                "verbs:",
                "  - bad_capability_verb",
            ]
        ),
        encoding="utf-8",
    )
    (verbs_dir / "bad_capability_verb.yaml").write_text(
        "\n".join(
            [
                "name: bad_capability_verb",
                "description: invalid capability verb",
                "services:",
                "  - LLMGatewayService",
                "execution_mode: capability_backed",
                "requires_capability_selector: true",
            ]
        ),
        encoding="utf-8",
    )

    registry = ToolRegistry(base_dir=base)
    try:
        registry.tools_for_packs(["executive_pack"])
        assert False, "expected capability-backed registry invariant validation to fail"
    except ValueError as exc:
        assert "cannot bind LLMGatewayService" in str(exc)
