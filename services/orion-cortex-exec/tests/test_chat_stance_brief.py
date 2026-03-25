from __future__ import annotations

from app.chat_stance import build_chat_stance_inputs, fallback_chat_stance_brief, parse_chat_stance_brief
from orion.schemas.chat_stance import ChatStanceBrief


def test_chat_stance_brief_schema_roundtrip() -> None:
    brief = ChatStanceBrief(
        conversation_frame="technical",
        user_intent="Patch chat stack.",
        self_relevance="Identity continuity matters.",
        juniper_relevance="Juniper requested bounded synthesis.",
        answer_strategy="DirectAnswer",
        stance_summary="Use brief then answer directly.",
    )
    dumped = brief.model_dump(mode="json")
    reparsed = ChatStanceBrief.model_validate(dumped)
    assert reparsed.conversation_frame == "technical"
    assert reparsed.answer_strategy == "DirectAnswer"


def test_build_chat_stance_inputs_maps_social_reflective_and_dream() -> None:
    ctx = {
        "orion_identity_summary": ["Distributed self-model"],
        "juniper_relationship_summary": ["Trusted collaborator"],
        "response_policy_summary": ["Avoid assistant sludge"],
        "social_posture": ["warm", "direct"],
        "active_relationship_facets": ["collaborative build"],
        "social_hazards": ["over-softening"],
        "recall_bundle": {
            "fragments": [
                {"source": "orion_journal", "snippet": "Theme: continuity through memory integration."},
                {"source": "metacog_tick", "snippet": "Tension: speed vs coherence."},
                {"source": "dream_cycle", "snippet": "Motif: bridges and roots."},
            ]
        },
    }

    built = build_chat_stance_inputs(ctx)

    assert built["identity"]["orion"] == ["Distributed self-model"]
    assert "collaborative build" in built["social"]["relationship_facets"]
    assert "warm" in built["social"]["social_posture"]
    assert built["reflective"]["themes"]
    assert built["reflective"]["dream_motifs"]


def test_parse_chat_stance_brief_and_fallback() -> None:
    parsed = parse_chat_stance_brief(
        '{"conversation_frame":"mixed","user_intent":"help","self_relevance":"x","juniper_relevance":"y","active_identity_facets":[],"active_growth_axes":[],"active_relationship_facets":[],"social_posture":[],"reflective_themes":[],"active_tensions":[],"dream_motifs":[],"response_priorities":["direct"],"response_hazards":["sludge"],"answer_strategy":"DirectAnswer","stance_summary":"short"}'
    )
    assert parsed is not None
    assert parsed.user_intent == "help"

    fb = fallback_chat_stance_brief({"user_message": "can you help"})
    assert fb.answer_strategy == "DirectAnswer"
    assert "generic assistant sludge" in fb.response_hazards
