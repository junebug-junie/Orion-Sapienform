from __future__ import annotations

from app.executor import _inject_identity_context


def test_inject_identity_context_from_personality_file() -> None:
    ctx = {
        "personality_file": "orion/cognition/personality/orion_identity.yaml",
        "personality_summary": "existing summary",
    }

    _inject_identity_context(ctx)

    assert ctx["personality_summary"] == "existing summary"
    assert ctx["orion_identity_summary"]
    assert ctx["juniper_relationship_summary"]
    assert ctx["response_policy_summary"]


def test_inject_identity_context_fails_soft_on_missing_file() -> None:
    ctx = {"personality_file": "orion/cognition/personality/does_not_exist.yaml"}

    _inject_identity_context(ctx)

    assert "orion_identity_summary" not in ctx
    assert "juniper_relationship_summary" not in ctx
    assert "response_policy_summary" not in ctx
