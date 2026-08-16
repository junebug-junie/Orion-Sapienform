from __future__ import annotations

from orion.harness.finalize import build_finalize_reflect_context
from orion.harness.tests.fixtures import (
    make_appraisal,
    make_repair_overlay,
    make_thought,
)


def test_finalize_reflect_context_routes_to_agent_lane() -> None:
    """5b reflection routes to the `agent` lane (circe-worker-agent-1), not `chat`
    or `background`.

    Was `background` until confirmed wrong live 2026-08-16
    (corr=d9c3a9fc-0bc3-4e42-86cc-622613dfedbd): 5c's own orion_voice_finalize call
    also runs on `background`/atlas-worker-2 and can occupy it for 90s+, which
    starved this call's LLMGatewayService RPC entirely (cortex-exec's internal 300s
    timeout fired with no reply at all). `chat` was considered and rejected: it
    maps to circe-worker-1, the same worker chat_general's own live draft
    generation uses, with no admission/concurrency throttling on that route --
    would trade the 5b-vs-5c collision for 5b-vs-live-user-chat contention. `agent`
    (verified live) is currently unused by any other verb, isolating this call
    from both.
    """
    ctx = build_finalize_reflect_context(
        correlation_id="c-1",
        draft_text="draft",
        thought=make_thought(),
        substrate_appraisal=make_appraisal(),
        repair_overlay=make_repair_overlay(),
        user_message="How are you?",
    )
    assert ctx["llm_lane"] == "agent"
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
