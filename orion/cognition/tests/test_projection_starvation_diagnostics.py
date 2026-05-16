from __future__ import annotations

from orion.cognition.projection import project_unified_beliefs_for_mind
from orion.cognition.projection_builder import (
    build_cognitive_projection_for_mind_with_diagnostics,
    summarize_projection_build,
)
from orion.substrate.relational.beliefs import AnchorBeliefSliceV1, UnifiedRelationalBeliefSetV1


def test_summarize_projection_build_reports_starvation_reasons() -> None:
    beliefs = UnifiedRelationalBeliefSetV1(
        anchors={anchor: AnchorBeliefSliceV1(anchor=anchor) for anchor in ("orion", "relationship", "juniper")},
        lineage=["test:empty"],
    )
    projection = project_unified_beliefs_for_mind(beliefs)
    assert projection is not None
    assert projection.item_count == 0

    diagnostics = summarize_projection_build({"verb": "chat_general"}, beliefs=beliefs, projection=projection)
    assert diagnostics["item_count"] == 0
    assert "identity_yaml" in diagnostics["projection_sources_requested"]
    assert diagnostics["dropped_counts_by_reason"].get("no_active_projection_items", 0) >= 1
    assert diagnostics["short_circuit_policy_active"] is False


def test_build_cognitive_projection_for_mind_with_diagnostics_attaches_counts() -> None:
    beliefs = UnifiedRelationalBeliefSetV1(
        anchors={
            "orion": AnchorBeliefSliceV1(
                anchor="orion",
                concepts=[],
            )
        },
        lineage=["test:rich"],
    )
    projection = project_unified_beliefs_for_mind(beliefs)
    assert projection is not None

    _, diagnostics = build_cognitive_projection_for_mind_with_diagnostics(
        {"verb": "chat_general", "correlation_id": "corr-rich"},
    )
    assert isinstance(diagnostics["projection_sources_requested"], list)
    assert isinstance(diagnostics["projection_sources_returned"], list)
    assert isinstance(diagnostics["source_counts"], dict)
    assert isinstance(diagnostics["dropped_counts_by_reason"], dict)
    assert isinstance(diagnostics["producer_errors"], list)
    assert "short_circuit_policy_active" in diagnostics
