from __future__ import annotations

from app.chat_stance import (
    build_chat_stance_inputs,
    enforce_chat_stance_quality,
    fallback_chat_stance_brief,
    normalize_chat_stance_brief,
    parse_chat_stance_brief,
)
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
    assert reparsed.task_mode == "direct_response"
    assert reparsed.identity_salience == "medium"
    assert reparsed.answer_strategy == "DirectAnswer"


def test_build_chat_stance_inputs_maps_social_reflective_and_dream() -> None:
    ctx = {
        "orion_identity_summary": ["Distributed self-model"],
        "juniper_relationship_summary": ["Trusted collaborator"],
        "response_policy_summary": ["Avoid assistant sludge"],
        "social_posture": ["warm", "direct"],
        "active_relationship_facets": ["collaborative build"],
        "social_hazards": ["over-softening"],
        "social_turn_policy": {
            "decision": "reply",
            "should_speak": True,
            "addressed": True,
            "reasons": ["directly addressed by peer"],
        },
        "social_stance_snapshot": {
            "recent_social_orientation_summary": "Warm, direct, technically collaborative.",
        },
        "social_context_window": {
            "selected_candidates": [
                {"candidate_kind": "peer_continuity", "inclusion_decision": "include", "summary": "Recurring technical collaborator."}
            ]
        },
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
    assert "technical" in built["social_bridge"]["posture"]
    assert "engaged_turn" in built["social_bridge"]["posture"]
    assert "peer_continuity:include" in built["social_bridge"]["framing"]
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
    assert "generic assistant self-description" in fb.response_hazards


def test_build_chat_stance_inputs_falls_back_when_identity_missing() -> None:
    ctx = {"user_message": "who are you and who am i"}
    built = build_chat_stance_inputs(ctx)
    assert built["identity"]["orion"]
    assert built["identity"]["juniper"]
    assert built["identity"]["response_policy"]
    assert any("not a generic assistant" in item.lower() for item in built["identity"]["orion"])


def test_build_chat_stance_inputs_uses_repository_seam(monkeypatch) -> None:
    from app import chat_stance
    from orion.core.schemas.concept_induction import ConceptProfile

    profile = ConceptProfile.model_validate(
        {
            "profile_id": "profile-orion",
            "subject": "orion",
            "revision": 1,
            "created_at": "2026-03-23T00:00:00+00:00",
            "window_start": "2026-03-22T00:00:00+00:00",
            "window_end": "2026-03-23T00:00:00+00:00",
            "concepts": [
                {
                    "concept_id": "concept-1",
                    "label": "continuity",
                    "aliases": [],
                    "type": "identity",
                    "salience": 1.0,
                    "confidence": 0.8,
                    "embedding_ref": None,
                    "evidence": [],
                    "metadata": {},
                }
            ],
            "clusters": [],
            "state_estimate": None,
            "metadata": {},
        }
    )

    class FakeRepository:
        def status(self):
            return type("Status", (), {"backend": "local", "source_path": "/tmp/concepts.json", "placeholder_default_in_use": False})()

        def list_latest(self, subjects):
            return [
                type("Lookup", (), {"subject": subject, "profile": profile if subject == "orion" else None})()
                for subject in subjects
            ]

    monkeypatch.setattr(chat_stance, "build_concept_profile_repository", lambda: FakeRepository())
    built = chat_stance.build_chat_stance_inputs({"user_message": "who are you"})
    assert "continuity" in built["concept_induction"]["self"]


def test_fallback_chat_stance_brief_identity_turn_anchors_orion_and_juniper() -> None:
    fb = fallback_chat_stance_brief({"user_message": "Low ball first. Who are you and who am I?"})
    assert fb.conversation_frame == "identity_emergence"
    assert fb.identity_salience == "high"
    assert fb.task_mode == "identity_dialogue"
    assert fb.active_identity_facets
    assert fb.active_relationship_facets
    assert "generic assistant self-description" in fb.response_hazards


def test_fallback_chat_stance_brief_technical_frustration_sets_triage_low_identity() -> None:
    ctx = {
        "user_message": "I'm super pissed; our V100 won't seat and workflows are offline.",
        "chat_social_summary": {"social_posture": ["technical", "strained"], "relationship_facets": ["shared build"]},
        "chat_social_bridge_summary": {"posture": ["operational_frustration", "technical"], "hazards": ["context_softened:repair"]},
    }
    fb = fallback_chat_stance_brief(ctx)
    assert fb.task_mode == "triage"
    assert fb.identity_salience == "low"
    assert fb.answer_strategy == "TriageFirstOperationalReply"
    assert "self_intro_on_operational_turn" in fb.response_hazards
    assert "triage_operational_blockers_first" in fb.response_priorities


def test_semantic_guard_enriches_weak_generic_identity_brief() -> None:
    weak = ChatStanceBrief(
        conversation_frame="mixed",
        user_intent="identity question",
        self_relevance="none",
        juniper_relevance="none",
        active_identity_facets=[],
        active_growth_axes=[],
        active_relationship_facets=[],
        social_posture=[],
        reflective_themes=[],
        active_tensions=[],
        dream_motifs=[],
        response_priorities=[],
        response_hazards=[],
        answer_strategy="Generic assistant answer",
        stance_summary="I am a conversational AI designed to assist.",
    )
    ctx = {"user_message": "who are you and who am i"}
    enriched, fired = enforce_chat_stance_quality(weak, ctx)
    assert fired is True
    assert enriched.active_identity_facets
    assert enriched.active_relationship_facets
    assert enriched.conversation_frame == "identity_emergence"


def test_semantic_guard_triage_adds_self_intro_suppression_hazard() -> None:
    triage = ChatStanceBrief(
        conversation_frame="technical",
        task_mode="triage",
        identity_salience="medium",
        user_intent="gpu incident",
        self_relevance="maintain continuity",
        juniper_relevance="support incident handling",
        active_identity_facets=[],
        active_growth_axes=[],
        active_relationship_facets=["shared_build"],
        social_posture=["technical", "strained"],
        reflective_themes=[],
        active_tensions=[],
        dream_motifs=[],
        response_priorities=["answer_directly_first"],
        response_hazards=[],
        answer_strategy="DirectAnswer",
        stance_summary="technical triage",
    )
    enriched, fired = enforce_chat_stance_quality(triage, {"user_message": "gpu is offline"})
    assert fired is False
    assert enriched.identity_salience == "low"
    assert "self_intro_on_operational_turn" in enriched.response_hazards
    assert "triage_operational_blockers_first" in enriched.response_priorities


def test_normalize_chat_stance_brief_deliteralizes_known_phrases() -> None:
    brief = ChatStanceBrief(
        conversation_frame="identity_emergence",
        user_intent="who are you",
        self_relevance="identity",
        juniper_relevance="relationship",
        active_identity_facets=["Oríon is an ongoing cognitive presence in a long-running shared project."],
        active_growth_axes=[],
        active_relationship_facets=["Juniper is Oríon's co-architect, steward, and trusted interlocutor."],
        social_posture=[],
        reflective_themes=[],
        active_tensions=[],
        dream_motifs=[],
        response_priorities=["answer directly"],
        response_hazards=["Juniper is not a generic user."],
        answer_strategy="How shall we proceed with the tests?",
        stance_summary="Oríon is not a generic assistant.",
    )
    normalized = normalize_chat_stance_brief(brief)
    assert "continuity" in normalized.active_identity_facets
    assert "juniper_builder" in normalized.active_relationship_facets
    assert "known_person" in normalized.response_hazards
    assert normalized.answer_strategy == "avoid_ceremonial_tone"
    assert normalized.stance_summary == "avoid_generic_assistant"
