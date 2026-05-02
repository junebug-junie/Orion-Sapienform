from __future__ import annotations

from app.chat_stance import (
    build_chat_stance_debug_payload,
    build_chat_stance_inputs,
    enforce_chat_stance_quality,
    fallback_chat_stance_brief,
    normalize_chat_stance_brief,
    parse_chat_stance_brief,
    parse_chat_stance_brief_with_debug,
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


def test_build_chat_stance_inputs_includes_situation_category() -> None:
    ctx = {
        "user_message": "My kid asked why the sky is blue.",
        "situation_prompt_fragment": {
            "compact_text": "Situation compact",
            "should_mention": True,
            "mention_policy": "only_if_relevant",
            "summary_lines": ["line1"],
            "relevance_notes": ["kid_friendly_explanation"],
            "caution_lines": ["do not force mention"],
        },
        "situation_brief": {
            "presence": {
                "audience_mode": "kid_present",
                "requestor": {"display_name": "Juniper"},
                "privacy_mode": "session_only",
                "persist_to_memory": False,
                "companions": [{"display_name": "Kid", "age_band": "child", "role_hint": "listener"}],
            },
            "conversation_phase": {
                "phase_change": "long_gap",
                "continuity_mode": "reorient",
                "topic_staleness_risk": "medium",
                "time_since_last_user_turn_seconds": 7200,
            },
            "time": {"local_datetime": "2026-04-27T20:00:00-06:00", "time_of_day_label": "evening", "day_phase": "night", "weekday": "Sunday"},
            "place": {"coarse_location": "Utah", "locality": "Utah", "region": "UT"},
            "environment": {
                "current_weather": {"condition": "clear"},
                "forecast_next_2h": {"summary": "clear"},
                "forecast_next_6h": {"summary": "dry"},
                "forecast_next_24h": {"summary": "mild"},
                "practical_flags": {"take_jacket": False},
            },
            "lab": {"available": False, "thermal_risk": "unknown", "power_risk": "unknown"},
            "surface": {"surface": "hub_desktop", "input_modality": "typed"},
            "affordances": [{"kind": "kid_friendly_explanation", "trigger_relevance": "active", "suggestion": "Use age-appropriate explanation", "confidence": "medium"}],
        },
    }
    built = build_chat_stance_inputs(ctx)
    assert "situation" in built
    assert built["situation"]["presence"]["audience_mode"] == "kid_present"
    assert built["situation"]["conversation_phase"]["phase_change"] == "long_gap"
    assert built["situation"]["situation_relevance"] == "active"


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

        def list_latest(self, subjects, *, observer=None):
            return [
                type("Lookup", (), {"subject": subject, "profile": profile if subject == "orion" else None})()
                for subject in subjects
            ]

    monkeypatch.setattr(chat_stance, "build_concept_profile_repository", lambda: FakeRepository())
    built = chat_stance.build_chat_stance_inputs({"user_message": "who are you"})
    assert "continuity" in built["concept_induction"]["self"]


def test_chat_stance_shadow_mode_uses_local_result_and_passes_observer(monkeypatch) -> None:
    from app import chat_stance

    class FakeRepository:
        def __init__(self) -> None:
            self.last_observer = None

        def status(self):
            return type(
                "Status",
                (),
                {
                    "backend": "shadow",
                    "source_path": "/tmp/concepts.json",
                    "placeholder_default_in_use": False,
                },
            )()

        def list_latest(self, subjects, *, observer=None):
            self.last_observer = observer
            return [
                type(
                    "Lookup",
                    (),
                    {
                        "subject": "orion",
                        "profile": type(
                            "Profile",
                            (),
                            {
                                "concepts": [
                                    type(
                                        "Concept",
                                        (),
                                        {
                                            "label": "continuity",
                                            "type": "identity",
                                            "salience": 1.0,
                                            "confidence": 0.9,
                                        },
                                    )()
                                ],
                            },
                        )(),
                    },
                )()
            ]

    fake_repository = FakeRepository()
    monkeypatch.setattr(chat_stance, "build_concept_profile_repository", lambda: fake_repository)
    built = chat_stance.build_chat_stance_inputs(
        {
            "user_message": "who are you",
            "session_id": "sid-1",
            "correlation_id": "corr-1",
        }
    )
    assert "continuity" in built["concept_induction"]["self"]
    assert fake_repository.last_observer["consumer"] == "chat_stance"
    assert fake_repository.last_observer["session_id"] == "sid-1"
    assert fake_repository.last_observer["correlation_id"] == "corr-1"


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


def test_fallback_chat_stance_brief_preserves_situation_guidance() -> None:
    ctx = {
        "user_message": "My kid asked why the sky is blue.",
        "situation_prompt_fragment": {
            "compact_text": "Situation compact",
            "should_mention": True,
            "mention_policy": "only_if_relevant",
            "summary_lines": ["line1"],
            "relevance_notes": ["kid_friendly_explanation", "lightly re-anchor after long gap"],
            "caution_lines": ["do not force mention"],
        },
        "situation_brief": {
            "presence": {"audience_mode": "kid_present", "requestor": {"display_name": "Juniper"}},
            "conversation_phase": {"phase_change": "long_gap"},
            "lab": {"available": False, "thermal_risk": "unknown", "power_risk": "unknown"},
            "affordances": [{"kind": "kid_friendly_explanation", "trigger_relevance": "active", "suggestion": "Use age-appropriate explanation"}],
        },
    }
    fb = fallback_chat_stance_brief(ctx)
    assert fb.situation_relevance == "active"
    assert fb.audience_context == "kid_present"
    assert fb.temporal_context == "long_gap"
    assert fb.situation_response_guidance
    assert "do not force irrelevant time/weather commentary" in fb.response_hazards


def test_fallback_chat_stance_brief_suppresses_irrelevant_weather_priority() -> None:
    ctx = {
        "user_message": "what's been on your mind lately?",
        "situation_prompt_fragment": {
            "compact_text": "Situation compact",
            "should_mention": False,
            "mention_policy": "only_if_relevant",
            "summary_lines": [],
            "relevance_notes": [],
            "caution_lines": ["do not force mention"],
        },
        "situation_brief": {
            "presence": {"audience_mode": "solo", "requestor": {"display_name": "Juniper"}},
            "conversation_phase": {"phase_change": "short_pause"},
            "affordances": [{"kind": "fatigue_or_sleep_boundary", "trigger_relevance": "background", "suggestion": "Use only when relevant"}],
        },
    }
    fb = fallback_chat_stance_brief(ctx)
    assert fb.situation_relevance == "background"
    assert not any("weather" in item.lower() for item in fb.response_priorities)
    assert "do not force irrelevant time/weather commentary" in fb.response_hazards


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


def test_parse_chat_stance_brief_with_debug_reports_normalization_flag() -> None:
    brief, debug = parse_chat_stance_brief_with_debug(
        '{"conversation_frame":"mixed","user_intent":"help","self_relevance":"x","juniper_relevance":"y","active_identity_facets":["ongoing cognitive presence in a long-running shared project"],"active_growth_axes":[],"active_relationship_facets":[],"social_posture":[],"reflective_themes":[],"active_tensions":[],"dream_motifs":[],"response_priorities":["direct"],"response_hazards":["sludge"],"answer_strategy":"DirectAnswer","stance_summary":"short"}'
    )
    assert brief is not None
    assert debug["normalized_applied"] is True


def test_build_chat_stance_debug_payload_includes_grouped_contract_and_lineage() -> None:
    ctx = {
        "user_message": "what changed?",
        "memory_digest": "m1",
        "orion_identity_summary": ["orion-id"],
        "juniper_relationship_summary": ["juniper-rel"],
        "response_policy_summary": ["policy"],
        "chat_reasoning_summary_used": True,
        "chat_autonomy_selected_subject": "orion",
        "chat_autonomy_backend": "graph",
        "chat_stance_inputs": {
            "identity": {"orion": ["orion-id"], "juniper": ["juniper-rel"], "response_policy": ["policy"]},
            "concept_induction": {"self": ["continuity"], "relationship": [], "growth": [], "tension": []},
            "social": {"social_posture": ["direct"], "relationship_facets": ["shared_build"], "hazards": ["h1"]},
            "social_bridge": {"posture": ["direct"], "hazards": [], "framing": [], "turn_decision": "reply", "summary": ["s"]},
            "reflective": {"themes": [], "tensions": [], "dream_motifs": []},
            "autonomy": {"summary": {"stance_hint": "focus"}, "debug": {"_runtime": {"backend": "graph"}}},
            "reasoning_summary": {"summary_text": "r", "hazards": ["h"], "tensions": [], "fallback_recommended": False},
            "situation": {
                "situation_relevance": "active",
                "presence": {"audience_mode": "kid_present"},
                "conversation_phase": {"phase_change": "long_gap"},
                "situation_prompt_fragment": {"compact_text": "compact"},
                "affordances": [{"kind": "kid_friendly_explanation"}],
            },
        },
        "situation_prompt_fragment": {"compact_text": "compact"},
        "presence_context": {"audience_mode": "kid_present"},
    }
    payload = build_chat_stance_debug_payload(
        ctx=ctx,
        synthesized_brief={"task_mode": "direct_response"},
        final_brief={"task_mode": "direct_response"},
        fallback_invoked=False,
        normalized_applied=True,
        semantic_fallback=False,
        quality_modified=False,
    )
    assert payload["source_inputs"]["concept_induction"]["self"] == ["continuity"]
    assert "situation" in payload["overview"]["categories_present"]
    assert payload["source_inputs"]["situation"]["presence"]["audience_mode"] == "kid_present"
    assert payload["enforcement"]["normalized_applied"] is True
    assert payload["final_prompt_contract"]["chat_stance_brief"]["task_mode"] == "direct_response"
    assert payload["final_prompt_contract"]["situation_prompt_fragment"]["compact_text"] == "compact"
    assert payload["lineage_summary"][0].startswith("concept/self injected")


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


def test_ensure_chat_stance_pipeline_ctx_populates_stance_inputs_for_agent_mode(monkeypatch) -> None:
    monkeypatch.setenv("AUTONOMY_REPOSITORY_BACKEND", "local")
    from app.executor import ensure_chat_stance_pipeline_ctx

    ctx: dict = {"mode": "agent", "user_message": "hello", "verb": "chat_general", "correlation_id": "pytest-stance-1"}
    ensure_chat_stance_pipeline_ctx(ctx)
    inputs = ctx.get("chat_stance_inputs")
    assert isinstance(inputs, dict)
    assert inputs.get("identity", {}).get("orion")
    assert "autonomy" in inputs
    first = inputs
    ensure_chat_stance_pipeline_ctx(ctx)
    assert ctx.get("chat_stance_inputs") is first


def test_parse_chat_stance_brief_coerces_playful_invitation_and_scalar_social_posture() -> None:
    raw = (
        '{"conversation_frame":"playful_invitation","task_mode":"playful_exchange","identity_salience":"medium",'
        '"user_intent":"greet","self_relevance":"x","juniper_relevance":"y",'
        '"active_identity_facets":[],"active_growth_axes":[],"active_relationship_facets":[],'
        '"social_posture":"friendly and playful",'
        '"reflective_themes":[],"active_tensions":[],"dream_motifs":[],'
        '"response_priorities":["direct"],"response_hazards":["sludge"],'
        '"answer_strategy":"DirectAnswer","stance_summary":"short"}'
    )
    brief, debug = parse_chat_stance_brief_with_debug(raw)
    assert brief is not None
    assert debug.get("parse_error") is None
    assert debug.get("coercion_applied") is True
    assert brief.conversation_frame == "playful_relational"
    assert brief.social_posture == ["friendly and playful"]


def test_parse_chat_stance_brief_unknown_frame_defaults_to_mixed() -> None:
    raw = (
        '{"conversation_frame":"ceremonial_launch","task_mode":"direct_response","identity_salience":"medium",'
        '"user_intent":"u","self_relevance":"s","juniper_relevance":"j",'
        '"active_identity_facets":[],"active_growth_axes":[],"active_relationship_facets":[],"social_posture":[],'
        '"reflective_themes":[],"active_tensions":[],"dream_motifs":[],'
        '"response_priorities":[],"response_hazards":[],"answer_strategy":"A","stance_summary":"z"}'
    )
    brief, debug = parse_chat_stance_brief_with_debug(raw)
    assert brief is not None
    assert brief.conversation_frame == "mixed"


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
