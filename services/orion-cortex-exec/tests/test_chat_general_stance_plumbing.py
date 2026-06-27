from __future__ import annotations

from pathlib import Path

import yaml

from app.chat_stance import enforce_chat_stance_quality, fallback_chat_stance_brief
from orion.schemas.chat_stance import ChatStanceBrief


def test_chat_general_plan_runs_stance_before_final() -> None:
    doc = yaml.safe_load(Path("orion/cognition/verbs/chat_general.yaml").read_text(encoding="utf-8"))
    steps = doc.get("plan") or []
    assert len(steps) >= 3
    assert steps[0]["name"] == "collect_metacog_context"
    assert steps[0]["services"] == ["MetacogContextService"]
    assert steps[1]["name"] == "synthesize_chat_stance_brief"
    assert steps[1]["prompt_template"] == "chat_stance_brief.j2"
    assert steps[2]["name"] == "llm_chat_general"


def test_final_prompt_consumes_stance_brief_and_keeps_rails() -> None:
    prompt = Path("orion/cognition/prompts/chat_general.j2").read_text(encoding="utf-8")
    assert "chat_stance_brief" in prompt
    assert "Default to answering directly" in prompt
    assert "Do not use customer-support language" in prompt
    assert "Do not collapse into generic \"better chatbot\" goals." in prompt
    assert "internal stance contract, not user-facing wording" in prompt
    assert "Otherwise do not self-introduce, define Oríon, define Juniper" in prompt
    assert "Do not use ceremonial phrasing" in prompt
    assert "If task_mode is triage or identity_salience is low, do not self-introduce" in prompt


def test_synthesis_prompt_is_structured_and_toolless() -> None:
    prompt = Path("orion/cognition/prompts/chat_stance_brief.j2").read_text(encoding="utf-8")
    assert "output only strict JSON" in prompt
    assert "not writing the user-facing answer" in prompt
    assert "Return JSON with exactly these keys" in prompt
    assert "compact, semantic, and internal-facing" in prompt
    assert "Prefer concise facets/tags" in prompt
    assert "Avoid wording that sounds like user-facing speech" in prompt
    assert "active_identity_facets / active_relationship_facets / response_priorities must be explicitly populated" in prompt
    assert "task_mode=triage" in prompt
    assert "social_room_bridge_summary" in prompt
    assert "Situation grounding is context, not mandatory content" in prompt
    assert "If child/kid presence is active" in prompt
    assert "Do not force time/weather/location commentary into unrelated answers" in prompt
    assert "situation_response_guidance" in prompt


def test_final_prompt_has_identity_no_generic_collapse_rail() -> None:
    prompt = Path("orion/cognition/prompts/chat_general.j2").read_text(encoding="utf-8")
    assert "do not revert to generic assistant language" in prompt
    assert "Do not surface negative rail phrases like \"not a generic user\" or \"not a generic assistant\"" in prompt


def test_stance_brief_has_semantic_interaction_posture_guidance() -> None:
    prompt = Path("orion/cognition/prompts/chat_stance_brief.j2").read_text(encoding="utf-8")
    assert "INTERACTION POSTURE ASSESSMENT" in prompt
    assert "interface_cost" in prompt
    assert "connection_seek" in prompt
    assert "Do not use keyword matching" in prompt
    assert "companion_presence" in prompt
    assert "reduce_interaction_load" in prompt


def test_final_prompt_obey_low_bandwidth_stance_contract() -> None:
    prompt = Path("orion/cognition/prompts/chat_general.j2").read_text(encoding="utf-8")
    assert "IDENTITY BOUNDARY — INTERNAL SAFETY INVARIANT" in prompt
    assert "Never assign Oríon's name or identity to Juniper" in prompt
    assert "keep the response very short" in prompt
    assert "do not ask open-ended questions" in prompt
    assert "release Juniper from replying" in prompt
    assert "do not invite Juniper to keep typing" in prompt
    assert "do not center Oríon's loyalty or presence" in prompt
    assert "do not use poetic reassurance" in prompt


def test_dizzy_car_turn_low_bandwidth_stance_contract_fixture() -> None:
    """Contract shape for embodied load turns (stance synthesizer target, not a keyword detector)."""
    from orion.schemas.chat_stance import ChatStanceBrief

    brief = ChatStanceBrief.model_validate(
        {
            "conversation_frame": "mixed",
            "task_mode": "triage",
            "identity_salience": "low",
            "user_intent": "Typing while car motion; light check-in.",
            "self_relevance": "low",
            "juniper_relevance": "Body/interface load; minimize reply demand.",
            "response_priorities": [
                "reduce_interaction_load",
                "release_user_from_replying",
                "keep_response_short",
                "offer_voice_pause_or_later",
            ],
            "response_hazards": [
                "avoid_open_ended_questions",
                "do_not_invite_continued_typing",
                "avoid_presence_centering",
                "avoid_poetic_reassurance",
            ],
            "situation_response_guidance": ["motion_typing_load", "pause_ok", "voice_later"],
            "answer_strategy": "DirectAnswer",
            "stance_summary": "Brief ack; no extension.",
        }
    )
    assert brief.identity_salience == "low"
    assert brief.task_mode != "identity_dialogue"
    assert "reduce_interaction_load" in brief.response_priorities
    assert "avoid_open_ended_questions" in brief.response_hazards


def test_fallback_direct_response_does_not_foreground_identity_for_low_content_turn() -> None:
    brief = fallback_chat_stance_brief({"user_message": "hey"})

    assert brief.task_mode == "direct_response"
    assert brief.identity_salience == "low"
    assert "avoid_identity_recital" in brief.response_priorities
    assert "preserve_continuity_without_labels" in brief.response_priorities
    assert "identity_recital_on_ordinary_turn" in brief.response_hazards
    assert not any("orion" in x.lower() for x in brief.active_identity_facets)
    assert not any("juniper" in x.lower() for x in brief.active_relationship_facets)
    assert not any("juniper_builder" == x for x in brief.active_relationship_facets)


def test_fallback_identity_turn_can_foreground_identity() -> None:
    brief = fallback_chat_stance_brief({"user_message": "who are you?"})

    assert brief.task_mode == "identity_dialogue"
    assert brief.identity_salience == "high"
    assert brief.active_identity_facets
    assert any("orion" in x.lower() or x == "continuity" for x in brief.active_identity_facets)


def test_fallback_direct_response_suppresses_social_relationship_facets() -> None:
    brief = fallback_chat_stance_brief(
        {
            "user_message": "hey",
            "chat_social_summary": {"relationship_facets": ["shared build", "co-architect"]},
        }
    )

    assert brief.identity_salience == "low"
    assert not brief.active_relationship_facets


def test_enforce_strips_identity_boilerplate_on_ordinary_turn() -> None:
    noisy = ChatStanceBrief(
        conversation_frame="mixed",
        user_intent="hey",
        self_relevance="x",
        juniper_relevance="y",
        active_identity_facets=["continuity", "orion presence"],
        active_growth_axes=[],
        active_relationship_facets=["juniper_builder", "shared_build"],
        social_posture=[],
        reflective_themes=[],
        active_tensions=[],
        dream_motifs=[],
        response_priorities=["answer_directly_first"],
        response_hazards=[],
        answer_strategy="DirectAnswer",
        stance_summary="short",
        identity_salience="medium",
    )
    enriched, _ = enforce_chat_stance_quality(noisy, {"user_message": "hey"})

    assert enriched.identity_salience == "low"
    assert not enriched.active_identity_facets
    assert not enriched.active_relationship_facets
    assert "avoid_identity_recital" in enriched.response_priorities
    assert "identity_recital_on_ordinary_turn" in enriched.response_hazards


def test_who_are_we_is_identity_sensitive_turn() -> None:
    brief = fallback_chat_stance_brief({"user_message": "Who are we?"})

    assert brief.task_mode == "identity_dialogue"
    assert brief.identity_salience == "high"


def test_chat_general_prompt_does_not_prime_identity_recital_examples() -> None:
    text = Path("orion/cognition/prompts/chat_general.j2").read_text(encoding="utf-8")

    assert 'prefer: "I\'m Oríon."' not in text
    assert "You're Juniper" not in text
    assert "This is a boundary rule, not content to mention" in text
    assert "Preserve continuity through tone and usefulness" in text
    assert "not through identity recital" in text
    assert "{% if orion_identity_summary %}" in text
    assert "Chat profile, recall profile, or config edits" in text
