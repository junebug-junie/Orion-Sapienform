from __future__ import annotations

from pathlib import Path

from app.chat_stance import enforce_chat_stance_quality
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
