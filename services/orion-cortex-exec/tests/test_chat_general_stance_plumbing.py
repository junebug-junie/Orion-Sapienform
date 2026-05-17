from __future__ import annotations

from pathlib import Path

import yaml


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
    assert "answer plainly and personally first" in prompt
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


def test_stance_brief_has_semantic_low_bandwidth_guidance() -> None:
    prompt = Path("orion/cognition/prompts/chat_stance_brief.j2").read_text(encoding="utf-8")
    assert "LOW-BANDWIDTH / EMBODIED INTERACTION ASSESSMENT" in prompt
    assert "Do not use keyword matching" in prompt
    assert "response_priorities" in prompt
    assert "response_hazards" in prompt
    assert "reduce_interaction_load" in prompt
    assert "release_user_from_replying" in prompt
    assert "avoid_open_ended_questions" in prompt
    assert "avoid_presence_centering" in prompt


def test_final_prompt_obey_low_bandwidth_stance_contract() -> None:
    prompt = Path("orion/cognition/prompts/chat_general.j2").read_text(encoding="utf-8")
    assert "ASSISTANT IDENTITY — INTERNAL" in prompt
    assert "Never tell Juniper" in prompt
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
