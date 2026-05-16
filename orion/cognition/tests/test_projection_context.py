from __future__ import annotations

from orion.cognition.projection_context import (
    enrich_projection_context,
    inject_identity_context_for_projection,
    summarize_projection_inputs,
)


def test_inject_identity_context_populates_yaml_producer_inputs() -> None:
    ctx: dict = {}
    source = inject_identity_context_for_projection(
        ctx,
        plan_metadata={"personality_file": "orion/cognition/personality/orion_identity.yaml"},
    )
    assert source == "configured_yaml"
    assert ctx["orion_identity_summary"]
    assert ctx["juniper_relationship_summary"]
    assert ctx["response_policy_summary"]


def test_summarize_projection_inputs_reports_recall_and_identity() -> None:
    ctx = {
        "recall_bundle": {"fragments": [{"snippet": "memory"}]},
        "orion_identity_summary": ["a"],
        "juniper_relationship_summary": ["b"],
        "response_policy_summary": ["c"],
        "messages": [{"role": "user", "content": "hi"}],
    }
    summary = summarize_projection_inputs(ctx, phase="orch_mind_preflight")
    assert summary["recall_bundle_present"] is True
    assert summary["recall_fragment_count"] == 1
    assert summary["identity_yaml_inputs"]["orion_identity_summary"] == 1
