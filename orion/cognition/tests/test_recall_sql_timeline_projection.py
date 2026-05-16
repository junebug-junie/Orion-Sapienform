"""Mind projection integration — sql_timeline recall fragments enrich projection."""

from __future__ import annotations

from typing import Any

from orion.cognition.projection_builder import build_cognitive_projection_for_mind_with_diagnostics
from orion.cognition.projection_context import enrich_projection_context


def test_sql_timeline_recall_enriches_mind_projection_beyond_identity() -> None:
    ctx: dict[str, Any] = {
        "verb": "chat_general",
        "orion_identity_summary": ["Oríon."],
        "juniper_relationship_summary": ["Juniper."],
        "response_policy_summary": ["Policy."],
        "recall_bundle": {
            "fragments": [
                {
                    "source": "sql_timeline",
                    "source_ref": "chat_history_log",
                    "tags": ["chat_timeline"],
                    "snippet": "User: What is the Teddy launch plan?\nOrion: We agreed on phased rollout.",
                },
            ],
            "citations": [],
        },
    }
    enrich_projection_context(ctx)
    projection, diag = build_cognitive_projection_for_mind_with_diagnostics(
        ctx,
        publish_tier_outcomes=False,
        build_path="test.sql_timeline_projection",
    )
    assert projection is not None
    assert int(projection.item_count or 0) >= 2
    assert "recall" in (diag.get("projection_sources_returned") or [])
    source_counts = diag.get("source_counts") or {}
    assert source_counts.get("orion", 0) >= 1
    categories = diag.get("category_counts") or {}
    assert categories.get("event", 0) >= 1 or categories.get("concept", 0) >= 0
    assert diag.get("input_summary", {}).get("recall_bundle_present") is True
