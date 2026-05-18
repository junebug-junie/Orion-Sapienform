from __future__ import annotations

from pathlib import Path

import pytest

from app.chat_stance import strip_identity_recital_leadin, suppress_chat_general_speech_identity_priming
from app.router import _apply_chat_general_identity_boundary_guard, _extract_final_text
from orion.schemas.cortex.schemas import StepExecutionResult


def _step(payload: dict, *, verb_name: str = "chat_general", step_name: str = "llm") -> StepExecutionResult:
    return StepExecutionResult(
        status="success",
        verb_name=verb_name,
        step_name=step_name,
        order=0,
        result={"LLMGatewayService": payload},
        latency_ms=1,
        node="n",
        logs=[],
        error=None,
    )


def test_router_has_no_banned_scrubber_symbols() -> None:
    router_src = (Path(__file__).resolve().parents[1] / "app" / "router.py").read_text(encoding="utf-8")
    banned = (
        "_SOMATIC_USER_LOAD_RE",
        "_INTERACTION_DEMAND_RE",
        "_INTERACTION_DEMAND_REPLACEMENTS",
        "_apply_interaction_load_guard",
        "interaction_load_guard_applied",
    )
    for symbol in banned:
        assert symbol not in router_src


def test_chat_general_identity_boundary_repairs_user_role_inversion() -> None:
    repaired, diag = _apply_chat_general_identity_boundary_guard(
        "You're Oríon. I'm here.",
        verb_name="chat_general",
    )
    assert repaired.startswith("I'm Oríon.")
    assert diag["identity_boundary_applied"] is True
    assert "You're Oríon" in diag["identity_boundary_violations"]


@pytest.mark.parametrize(
    ("raw", "violation"),
    [
        ("You are Orion.", "You are Oríon"),
        ("You are Oríon — steady.", "You are Oríon"),
        ("You’re Oríon here.", "You're Oríon"),
        ("youre orion, thanks.", "youre orion"),
        ("your name is orion", "your name is Orion"),
    ],
)
def test_identity_boundary_repairs_variants(raw: str, violation: str) -> None:
    repaired, diag = _apply_chat_general_identity_boundary_guard(raw, verb_name="chat_general")
    assert repaired.startswith("I'm Oríon")
    assert diag["identity_boundary_applied"] is True
    assert violation in diag["identity_boundary_violations"]


def test_identity_boundary_skips_non_chat_general() -> None:
    raw = "You're Oríon."
    repaired, diag = _apply_chat_general_identity_boundary_guard(raw, verb_name="chat_quick")
    assert repaired == raw
    assert diag["identity_boundary_applied"] is False


def test_identity_boundary_leaves_correct_assistant_user_roles() -> None:
    raw = "I'm Oríon. You're Juniper."
    repaired, diag = _apply_chat_general_identity_boundary_guard(raw, verb_name="chat_general")
    assert repaired == raw
    assert diag["identity_boundary_applied"] is False


def test_strip_identity_recital_leadin_on_ordinary_turn() -> None:
    raw = "You're Juniper — the one building this with me. I see the edits."
    stripped, changed = strip_identity_recital_leadin(raw, "k, did some edits to your chat profile")
    assert changed is True
    assert stripped == "I see the edits."
    assert "Juniper" not in stripped


def test_strip_identity_recital_leadin_skips_identity_question() -> None:
    raw = "You're Juniper — the one building this with me."
    stripped, changed = strip_identity_recital_leadin(raw, "who are you?")
    assert changed is False
    assert stripped == raw


def test_suppress_chat_general_speech_identity_priming_clears_kernels() -> None:
    ctx = {
        "user_message": "hey",
        "orion_identity_summary": ["Oríon is an ongoing cognitive presence."],
        "juniper_relationship_summary": ["Juniper is co-architect."],
        "chat_stance_brief": {
            "task_mode": "direct_response",
            "identity_salience": "medium",
            "active_identity_facets": ["continuity"],
            "active_relationship_facets": ["juniper_builder"],
            "response_priorities": [],
            "response_hazards": [],
        },
    }
    assert suppress_chat_general_speech_identity_priming(ctx) is True
    assert ctx["orion_identity_summary"] == []
    assert ctx["juniper_relationship_summary"] == []
    assert ctx["chat_stance_brief"]["active_identity_facets"] == []


def test_extract_final_text_does_not_rewrite_availability_phrases() -> None:
    raw = "I'm here, and I'm not going anywhere."
    final_text, diag = _extract_final_text([_step({"content": raw})], verb_name="chat_general")
    assert final_text == raw
    assert not diag.get("interaction_load_guard_applied")
