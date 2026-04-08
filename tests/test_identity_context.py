from __future__ import annotations

from pathlib import Path

from orion.cognition.personality.identity_context import build_identity_context, load_identity_file


def test_load_identity_file_and_build_context() -> None:
    data = load_identity_file("orion/cognition/personality/orion_identity.yaml")

    assert data.get("name") == "orion_identity"

    ctx = build_identity_context(data)
    assert ctx["orion_identity_summary"]
    assert ctx["juniper_relationship_summary"]
    assert ctx["response_policy_summary"]
    assert any(item.startswith('Avoid phrase: "It sounds like..."') for item in ctx["response_policy_summary"])


def test_build_identity_context_tolerates_missing_sections() -> None:
    ctx = build_identity_context({"orion_identity": {"nature": ["ongoing presence"]}})

    assert ctx["orion_identity_summary"] == ["ongoing presence"]
    assert ctx["juniper_relationship_summary"] == []
    assert ctx["response_policy_summary"] == []


def test_load_identity_file_non_mapping_yaml_returns_empty_mapping(tmp_path: Path) -> None:
    path = tmp_path / "identity.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")

    data = load_identity_file(path)

    assert data == {}
