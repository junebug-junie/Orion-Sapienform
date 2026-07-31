"""
Regression guardrails for brain-lane chat prompts + recall gating.

Rationale (2026-05): chat_quick.j2 expects `message_history` in the LIGHTWEIGHT
IDENTITY CONTEXT block. If exec never populates it, the model sees an empty
dialogue tail while `memory_digest` may still contain prior assistant text —
encouraging verbatim repetition when recall is on. Router recall gating used
`raw_user_text` only; empty raw text skipped user_message-derived guards.
"""

from __future__ import annotations

import re
from pathlib import Path

from jinja2 import Environment

from app.executor import _format_message_history_for_chat_prompt
from app.recall_utils import delivery_safe_recall_decision, plan_ctx_latest_user_text
from orion.schemas.cortex.schemas import ExecutionStep


REPO_ROOT = Path(__file__).resolve().parents[3]
CHAT_QUICK_TEMPLATE = REPO_ROOT / "orion" / "cognition" / "prompts" / "chat_quick.j2"
CHAT_GENERAL_TEMPLATE = REPO_ROOT / "orion" / "cognition" / "prompts" / "chat_general.j2"

# Keep in sync with LIGHTWEIGHT IDENTITY CONTEXT in chat_quick.j2 (non-optional lines only).
_CHAT_QUICK_REQUIRED_PLACEHOLDERS = (
    "user_message",
    "message_history",
    "memory_digest",
    "orion_identity_summary",
    "juniper_relationship_summary",
    "response_policy_summary",
)


def test_chat_quick_template_lists_message_history_placeholder() -> None:
    text = CHAT_QUICK_TEMPLATE.read_text(encoding="utf-8")
    for name in _CHAT_QUICK_REQUIRED_PLACEHOLDERS:
        needle = "{{ " + name + " }}"
        assert needle in text, f"missing {needle!r} in chat_quick.j2"
    assert "message_history" in text


def test_format_message_history_includes_latest_user_and_assistant() -> None:
    msgs = [
        {"role": "user", "content": "first ask"},
        {"role": "assistant", "content": "first reply about loops"},
        {"role": "user", "content": "recall seems broken — new ask"},
    ]
    out = _format_message_history_for_chat_prompt(msgs)
    assert "USER: recall seems broken" in out
    assert "ASSISTANT: first reply about loops" in out
    assert out.index("USER: recall") > out.index("ASSISTANT:")


def test_chat_quick_render_surfaces_transcript_not_only_user_line() -> None:
    """Empty message_history must never be the only dialogue anchor when turns exist."""
    tpl = Environment().from_string(CHAT_QUICK_TEMPLATE.read_text(encoding="utf-8"))
    msgs = [
        {"role": "user", "content": "bro you are stuck in a loop"},
        {"role": "assistant", "content": "You got me—loop's a thing."},
        {"role": "user", "content": "something is wrong with your recall."},
    ]
    hist = _format_message_history_for_chat_prompt(msgs)
    assert hist.strip()
    rendered = tpl.render(
        user_message="something is wrong with your recall.",
        message_history=hist,
        memory_digest="",
        orion_identity_summary=["stub"],
        juniper_relationship_summary=["stub"],
        response_policy_summary=["stub"],
        metadata={},
    )
    assert "something is wrong with your recall" in rendered
    assert "bro you are stuck in a loop" in rendered
    assert "You got me—loop" in rendered


_CHAT_QUICK_BASE_RENDER_ARGS = dict(
    user_message="hey",
    message_history="",
    memory_digest="",
    orion_identity_summary=["stub"],
    juniper_relationship_summary=["stub"],
    response_policy_summary=["stub"],
)


def test_chat_quick_frames_aitown_surface_as_currently_embodied_there() -> None:
    tpl = Environment().from_string(CHAT_QUICK_TEMPLATE.read_text(encoding="utf-8"))
    rendered = tpl.render(**_CHAT_QUICK_BASE_RENDER_ARGS, metadata={"surface": "aitown"})
    assert "You are embodied in ai-town right now" in rendered
    assert "You are not currently in ai-town" not in rendered


def test_chat_quick_frames_non_aitown_surface_as_outside_the_game() -> None:
    """Regression coverage (found live 2026-07-30): with no surface framing at
    all, ai-town dialogue in memory_digest bled into a hub turn verbatim --
    Orion answered Juniper by continuing an in-progress ai-town NPC exchange
    ("light folding") as if still mid-scene. Confirms hub-mode (surface unset
    or anything other than "aitown") now gets explicit instruction not to."""
    tpl = Environment().from_string(CHAT_QUICK_TEMPLATE.read_text(encoding="utf-8"))
    rendered = tpl.render(**_CHAT_QUICK_BASE_RENDER_ARGS, metadata={})
    assert "You are not currently in ai-town" in rendered
    assert "You are embodied in ai-town right now" not in rendered


def test_chat_quick_tells_model_to_look_for_the_aitown_marker() -> None:
    """Regression coverage (found live 2026-07-31): the CURRENT CONTEXT
    instruction alone told the model to treat "ai-town dialogue" as separate,
    but gave it no way to reliably tell which lines those were -- it had to
    guess from content. services/orion-recall's sql_adapter.py now prefixes
    ai-town-sourced memory_digest lines with a literal "[ai-town]" marker
    (see docs/superpowers/specs/2026-07-31-recall-aitown-source-tagging-design.md);
    this confirms the prompt actually tells the model to look for it."""
    tpl = Environment().from_string(CHAT_QUICK_TEMPLATE.read_text(encoding="utf-8"))
    rendered = tpl.render(**_CHAT_QUICK_BASE_RENDER_ARGS, metadata={})
    assert '"[ai-town]"' in rendered


def test_chat_quick_has_anti_repetition_instruction() -> None:
    tpl = Environment().from_string(CHAT_QUICK_TEMPLATE.read_text(encoding="utf-8"))
    rendered = tpl.render(**_CHAT_QUICK_BASE_RENDER_ARGS, metadata={})
    assert "Do not repeat a phrase, metaphor, or sentence structure" in rendered


def test_chat_quick_has_concrete_grounding_instruction() -> None:
    """Regression coverage (found live 2026-07-30/31): the anti-repetition fix
    alone stopped verbatim phrase reuse but not thematic drift into vague
    poetic abstraction (native ai-town NPCs spiraling into "light vs shadows"
    exchanges with no concrete referent). This instructs the model to name
    the actual thing it means instead of extending a metaphor."""
    tpl = Environment().from_string(CHAT_QUICK_TEMPLATE.read_text(encoding="utf-8"))
    rendered = tpl.render(**_CHAT_QUICK_BASE_RENDER_ARGS, metadata={})
    assert "Ground replies in something specific and concrete" in rendered


def test_chat_quick_anti_repetition_and_concrete_grounding_both_survive_together() -> None:
    """Both instructions target different failure modes (verbatim reuse vs.
    thematic drift into abstraction) and must both survive a single render --
    a future edit to one could accidentally clobber the other; separate tests
    for each wouldn't catch an adjacency/ordering regression between them."""
    tpl = Environment().from_string(CHAT_QUICK_TEMPLATE.read_text(encoding="utf-8"))
    rendered = tpl.render(**_CHAT_QUICK_BASE_RENDER_ARGS, metadata={})
    repetition_idx = rendered.index("Do not repeat a phrase, metaphor, or sentence structure")
    grounding_idx = rendered.index("Ground replies in something specific and concrete")
    assert repetition_idx < grounding_idx


def test_plan_ctx_latest_user_text_feeds_recall_gating_when_raw_missing() -> None:
    """Concrete-ops guard must see the real utterance when only user_message is set."""
    ctx = {
        "raw_user_text": "",
        "user_message": "Need runtime estimate for V100 on APC UPS battery backup and power draw.",
        "messages": [],
    }
    ut = plan_ctx_latest_user_text(ctx)
    assert "V100" in ut
    step = ExecutionStep(
        verb_name="chat_quick",
        step_name="llm_chat_quick",
        description="chat",
        order=0,
        services=["LLMGatewayService"],
        requires_memory=False,
    )
    decision = delivery_safe_recall_decision(
        {"enabled": True},
        [step],
        output_mode="direct_answer",
        verb_profile=None,
        user_text=ut,
    )
    assert decision["run_recall"] is False
    assert decision["reason"] == "concrete_ops_default_disabled"


def test_router_still_wires_plan_ctx_latest_user_text_for_recall_decision() -> None:
    """Brittle but cheap: if someone reverts router recall wiring, CI fails."""
    router_src = Path(__file__).resolve().parents[1] / "app" / "router.py"
    src = router_src.read_text(encoding="utf-8")
    assert "plan_ctx_latest_user_text" in src
    assert re.search(
        r"user_text\s*=\s*plan_ctx_latest_user_text\s*\(\s*ctx\s*\)",
        src,
    ), "router recall decision must pass plan_ctx_latest_user_text(ctx), not raw_user_text alone"


def test_chat_general_has_non_aitown_surface_framing_and_anti_repetition() -> None:
    """chat_general.j2 (the optional grounded/unified path) mirrors chat_quick.j2's
    surface-awareness and anti-repetition fixes for consistency, even though it's
    not the live path today -- so the same ai-town-content-bleed bug can't
    resurface if EMBODIMENT_SPEECH_UNIFIED_ENABLED is ever turned on."""
    text = CHAT_GENERAL_TEMPLATE.read_text(encoding="utf-8")
    assert 'metadata.get("surface") != "aitown"' in text
    assert "Do not repeat a phrase, metaphor, or sentence structure" in text
    assert "Ground replies in something specific and concrete" in text
    assert '"[ai-town]"' in text
