from __future__ import annotations

from pathlib import Path

from app.chat_stance import (
    enforce_chat_stance_quality,
    strip_identity_recital_leadin,
    strip_transactional_closers,
)
from orion.schemas.chat_stance import ChatStanceBrief


def _relational_brief(**overrides) -> ChatStanceBrief:
    base = {
        "conversation_frame": "reflective",
        "task_mode": "reflective_dialogue",
        "identity_salience": "low",
        "user_intent": "Companion presence; mind off recovery.",
        "self_relevance": "Hold space; be curious.",
        "juniper_relevance": "Relational continuity matters this turn.",
        "active_relationship_facets": ["shared_history", "companionship"],
        "active_identity_facets": [],
        "active_growth_axes": [],
        "social_posture": ["presence"],
        "reflective_themes": ["recovery"],
        "active_tensions": ["existential"],
        "dream_motifs": [],
        "response_priorities": [
            "companion_presence",
            "situated_curiosity",
            "hold_space",
            "no_solutioning",
        ],
        "response_hazards": [
            "avoid_task_tracking",
            "avoid_next_steps",
            "avoid_transactional_closers",
        ],
        "answer_strategy": "RelationalHoldSpace",
        "stance_summary": "Be present; ask one situated question; do not solution.",
    }
    base.update(overrides)
    return ChatStanceBrief.model_validate(base)


def test_strip_identity_recital_leadin_skips_relational_turn() -> None:
    raw = "You're Juniper — the one building this with me. What's been the weirdest part?"
    brief = {
        "conversation_frame": "reflective",
        "task_mode": "reflective_dialogue",
    }
    stripped, changed = strip_identity_recital_leadin(
        raw,
        "someone to talk. im lonely",
        chat_stance_brief=brief,
    )
    assert changed is False
    assert stripped == raw


def test_enforce_preserves_relational_frame_only_brief() -> None:
    brief = ChatStanceBrief.model_validate(
        {
            "conversation_frame": "reflective",
            "task_mode": "direct_response",
            "identity_salience": "low",
            "user_intent": "Hold space.",
            "self_relevance": "x",
            "juniper_relevance": "Relational continuity matters.",
            "active_relationship_facets": ["companionship"],
            "response_priorities": ["companion_presence"],
            "response_hazards": [],
            "answer_strategy": "DirectAnswer",
            "stance_summary": "short",
        }
    )
    enriched, _ = enforce_chat_stance_quality(brief, {"user_message": "just talk to me"})
    assert enriched.active_relationship_facets == ["companionship"]
    assert enriched.juniper_relevance == "Relational continuity matters."


def test_enforce_preserves_relational_brief_on_non_identity_turn() -> None:
    brief = _relational_brief()
    ctx = {"user_message": "just be curious about it and take my mind off recovery"}

    enriched, _ = enforce_chat_stance_quality(brief, ctx)

    assert enriched.task_mode == "reflective_dialogue"
    assert enriched.conversation_frame == "reflective"
    assert "shared_history" in enriched.active_relationship_facets
    assert enriched.juniper_relevance == "Relational continuity matters this turn."
    assert "companion_presence" in enriched.response_priorities
    assert "avoid_identity_recital" not in enriched.response_priorities
    assert "identity_recital_on_ordinary_turn" not in enriched.response_hazards


def test_enforce_still_compresses_instrumental_direct_response() -> None:
    brief = ChatStanceBrief.model_validate(
        {
            "conversation_frame": "mixed",
            "task_mode": "direct_response",
            "identity_salience": "low",
            "user_intent": "Quick ack.",
            "self_relevance": "x",
            "juniper_relevance": "y",
            "active_relationship_facets": ["juniper_builder"],
            "response_priorities": ["answer_directly_first"],
            "response_hazards": [],
            "answer_strategy": "DirectAnswer",
            "stance_summary": "short",
        }
    )
    ctx = {"user_message": "hey"}

    enriched, _ = enforce_chat_stance_quality(brief, ctx)

    assert enriched.juniper_relevance == "Prioritize practical usefulness over relationship labels."
    assert enriched.active_relationship_facets == []
    assert "avoid_identity_recital" in enriched.response_priorities


def test_enforce_preserves_playful_exchange_frame() -> None:
    brief = _relational_brief(
        conversation_frame="playful_relational",
        task_mode="playful_exchange",
    )
    ctx = {"user_message": "someone to talk. im lonely"}

    enriched, _ = enforce_chat_stance_quality(brief, ctx)

    assert enriched.task_mode == "playful_exchange"
    assert enriched.active_relationship_facets


def test_stance_brief_prompt_has_connection_seek_vocabulary() -> None:
    prompt = Path("orion/cognition/prompts/chat_stance_brief.j2").read_text(encoding="utf-8")
    assert "interface_cost" in prompt
    assert "connection_seek" in prompt
    assert "Do not use keyword matching" in prompt
    assert "companion_presence" in prompt
    assert "LOW-BANDWIDTH / EMBODIED INTERACTION ASSESSMENT" not in prompt


def test_speech_prompt_relational_curiosity_overrides_attention_frame() -> None:
    prompt = Path("orion/cognition/prompts/chat_general.j2").read_text(encoding="utf-8")
    assert "reflective_dialogue" in prompt
    assert "playful_exchange" in prompt
    assert "advisory" in prompt.lower()
    assert "avoid_transactional_closers" in prompt or "avoid_next_steps" in prompt


def test_strip_transactional_closers_on_relational_turn() -> None:
    raw = (
        "I hear the weight of that, Juniper. 5:30am feels like a long time away, but it's coming. "
        "Let me know if you need anything to make the time pass more comfortably."
    )
    stripped, changed = strip_transactional_closers(raw, chat_stance_brief=_relational_brief())
    assert changed is True
    assert "Let me know" not in stripped
    assert stripped.endswith("it's coming.")


def test_strip_transactional_closers_skips_instrumental_turn() -> None:
    raw = "Patch is ready. Let me know when you're ready to deploy."
    brief = {
        "task_mode": "direct_response",
        "conversation_frame": "mixed",
        "response_hazards": [],
    }
    stripped, changed = strip_transactional_closers(raw, chat_stance_brief=brief)
    assert changed is False
    assert stripped == raw


def test_enforce_upgrades_companion_thread_continuation() -> None:
    brief = ChatStanceBrief(
        conversation_frame="mixed",
        task_mode="direct_response",
        identity_salience="low",
        user_intent="Vent continuation.",
        self_relevance="Answer directly.",
        juniper_relevance="Prioritize practical usefulness over relationship labels.",
        active_identity_facets=[],
        active_growth_axes=[],
        active_relationship_facets=[],
        social_posture=[],
        reflective_themes=[],
        active_tensions=[],
        dream_motifs=[],
        response_priorities=["direct_answer"],
        response_hazards=[],
        situation_relevance="background",
        temporal_context="now",
        audience_context="private",
        environmental_context="hospital",
        operational_context="none",
        situation_response_guidance=[],
        answer_strategy="Acknowledge and stay present.",
        stance_summary="Hold space.",
    )
    ctx = {
        "user_message": (
            "thanks, it's just hard... We have to be out of here by 5:30am and it will be a "
            "terrible night's sleep with nurses in and out."
        ),
        "message_history": [
            {"role": "user", "content": "just looking for a shoulder to talk. Can you help me keep my mind off?"},
            {
                "role": "assistant",
                "content": "I'm here, Juniper. Let's just sit with it — whatever it is.",
            },
        ],
    }
    enriched, _ = enforce_chat_stance_quality(brief, ctx)
    assert enriched.task_mode == "reflective_dialogue"
    assert "avoid_transactional_closers" in enriched.response_hazards
    assert "companion_presence" in enriched.response_priorities

