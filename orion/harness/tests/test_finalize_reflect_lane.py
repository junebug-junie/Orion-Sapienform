from __future__ import annotations

from orion.harness.finalize import build_finalize_reflect_context
from orion.harness.tests.fixtures import (
    make_appraisal,
    make_repair_overlay,
    make_thought,
)


def test_finalize_reflect_context_routes_to_background_lane() -> None:
    """5b reflection must leave the saturated `chat` worker for the metacog/background lane."""
    ctx = build_finalize_reflect_context(
        correlation_id="c-1",
        draft_text="draft",
        thought=make_thought(),
        substrate_appraisal=make_appraisal(),
        repair_overlay=make_repair_overlay(),
        user_message="How are you?",
    )
    assert ctx["llm_lane"] == "background"
    # Keep resilience: fall back to chat only if the metacog route is ever absent.
    assert ctx["allow_chat_fallback"] is True


def test_finalize_reflect_context_lane_is_top_level_for_cortex_ctx_merge() -> None:
    """cortex-exec spreads request.context into ctx at top level (main.py), and
    resolve_llm_lane_for_step reads ctx.get("llm_lane"). Guard the key placement."""
    ctx = build_finalize_reflect_context(
        correlation_id="c-1",
        draft_text="draft",
        thought=make_thought(),
        substrate_appraisal=make_appraisal(),
        repair_overlay=make_repair_overlay(),
        user_message="",
    )
    assert "llm_lane" in ctx
    assert "options" not in ctx or "llm_lane" not in ctx.get("options", {})
