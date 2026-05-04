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
    )
    assert "something is wrong with your recall" in rendered
    assert "bro you are stuck in a loop" in rendered
    assert "You got me—loop" in rendered


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
