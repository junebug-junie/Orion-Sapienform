from __future__ import annotations

import logging

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
    assert ctx["identity_kernel_source"] == "configured_yaml"


def test_inject_identity_context_fails_soft_on_missing_file() -> None:
    ctx = {"personality_file": "orion/cognition/personality/does_not_exist.yaml"}

    _inject_identity_context(ctx)

    assert "orion_identity_summary" in ctx
    assert "juniper_relationship_summary" in ctx
    assert "response_policy_summary" in ctx
    assert any("not a generic assistant" in item.lower() for item in ctx["orion_identity_summary"])
    assert any("not a generic user" in item.lower() for item in ctx["juniper_relationship_summary"])
    assert ctx["identity_kernel_source"] == "fallback_load_error"


def test_inject_identity_context_backfills_when_existing_lists_empty() -> None:
    ctx = {
        "orion_identity_summary": [],
        "juniper_relationship_summary": [],
        "response_policy_summary": [],
    }
    _inject_identity_context(ctx)
    assert ctx["orion_identity_summary"]
    assert ctx["juniper_relationship_summary"]
    assert ctx["response_policy_summary"]
    assert ctx["identity_kernel_source"] == "fallback_missing_metadata"


def test_inject_identity_context_does_not_report_fallback_when_yaml_loads(caplog) -> None:
    ctx = {"personality_file": "orion/cognition/personality/orion_identity.yaml"}
    with caplog.at_level(logging.INFO):
        _inject_identity_context(ctx)
    assert ctx["identity_kernel_source"] == "configured_yaml"
    assert "identity_kernel_source=fallback_" not in caplog.text
