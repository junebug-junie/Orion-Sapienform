from __future__ import annotations

from orion.harness.finalize import (
    build_finalize_reflect_context,
    build_response_repair_context,
)
from orion.harness.tests.fixtures import (
    make_appraisal,
    make_reflection,
    make_repair_overlay,
    make_thought,
)
from orion.llm.routes import AGENT_ROUTE_FCC_MODEL_LABEL


def test_finalize_reflect_context_chat_owned_no_lease_routes_chat() -> None:
    """Unleashed Hub chat (MODEL_SONNET / non-agent) finalizes on gateway chat."""
    ctx = build_finalize_reflect_context(
        correlation_id="c-1",
        draft_text="draft",
        thought=make_thought(),
        substrate_appraisal=make_appraisal(),
        repair_overlay=make_repair_overlay(),
        user_message="night night",
        fcc_model_label="MODEL_SONNET",
    )
    assert ctx["llm_lane"] == "chat"
    assert ctx["llm_route"] == "chat"
    assert ctx["allow_chat_fallback"] is False


def test_finalize_reflect_context_agent_owned_no_lease_routes_agent() -> None:
    ctx = build_finalize_reflect_context(
        correlation_id="c-1",
        draft_text="draft",
        thought=make_thought(),
        substrate_appraisal=make_appraisal(),
        repair_overlay=make_repair_overlay(),
        user_message="investigate",
        fcc_model_label=AGENT_ROUTE_FCC_MODEL_LABEL,
    )
    assert ctx["llm_lane"] == "agent"
    assert ctx["llm_route"] == "agent"
    assert ctx["allow_chat_fallback"] is False


def test_finalize_reflect_context_missing_label_defaults_chat() -> None:
    """No lease + no label → chat (unified chat default), not agent."""
    ctx = build_finalize_reflect_context(
        correlation_id="c-1",
        draft_text="draft",
        thought=make_thought(),
        substrate_appraisal=make_appraisal(),
        repair_overlay=make_repair_overlay(),
        user_message="How are you?",
    )
    assert ctx["llm_lane"] == "chat"
    assert ctx["llm_route"] == "chat"
    assert ctx["allow_chat_fallback"] is False


def test_finalize_reflect_context_lane_is_top_level_for_cortex_ctx_merge() -> None:
    """cortex-exec spreads request.context into ctx; resolve_llm_lane_for_step
    reads ctx.get("llm_lane"). Guard key placement."""
    ctx = build_finalize_reflect_context(
        correlation_id="c-1",
        draft_text="draft",
        thought=make_thought(),
        substrate_appraisal=make_appraisal(),
        repair_overlay=make_repair_overlay(),
        user_message="",
        fcc_model_label="MODEL_SONNET",
    )
    assert "llm_lane" in ctx
    assert "options" not in ctx or "llm_lane" not in ctx.get("options", {})


def test_response_repair_context_chat_owned_no_lease_routes_chat() -> None:
    ctx = build_response_repair_context(
        correlation_id="c-repair",
        draft_text="draft",
        reflection=make_reflection(),
        user_message="night night",
        fcc_model_label="MODEL_SONNET",
    )
    assert ctx["llm_route"] == "chat"
    assert ctx["llm_lane"] == "chat"
    assert ctx["allow_chat_fallback"] is False


def test_response_repair_context_agent_owned_no_lease_routes_agent() -> None:
    ctx = build_response_repair_context(
        correlation_id="c-repair",
        draft_text="draft",
        reflection=make_reflection(),
        user_message="investigate",
        fcc_model_label=AGENT_ROUTE_FCC_MODEL_LABEL,
    )
    assert ctx["llm_route"] == "agent"
    assert ctx["llm_lane"] == "agent"
    assert ctx["allow_chat_fallback"] is False
