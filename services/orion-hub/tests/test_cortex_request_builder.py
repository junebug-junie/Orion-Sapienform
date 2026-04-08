from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cortex_request_builder.py"
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SPEC = importlib.util.spec_from_file_location("hub_cortex_request_builder", MODULE_PATH)
assert SPEC and SPEC.loader
hub_builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hub_builder)


def test_agent_mode_emits_supervised_delivery_ready_request() -> None:
    req, debug, _ = hub_builder.build_chat_request(
        payload={
            "mode": "agent",
            "packs": ["executive_pack"],
        },
        session_id="sid-agent",
        user_id="user-1",
        trace_id="trace-agent",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_http",
        prompt="Write me a deployment guide for this service.",
    )

    assert req.mode == "agent"
    assert req.route_intent == "none"
    assert req.verb is None
    assert req.options["supervised"] is True
    assert req.recall == {
        "enabled": True,
        "required": False,
        "mode": "hybrid",
        "profile": None,
    }
    assert debug["selected_ui_route"] == "agent"
    assert debug["supervised"] is True
    assert debug["force_agent_chain"] is False
    assert req.packs == ["executive_pack"]
    assert req.metadata["hub_route"]["selected_ui_route"] == "agent"


def test_agent_mode_with_recall_disabled_keeps_supervised_route_and_explicit_recall_shape() -> None:
    req, debug, _ = hub_builder.build_chat_request(
        payload={
            "mode": "agent",
            "packs": ["executive_pack"],
            "use_recall": False,
        },
        session_id="sid-agent-no-recall",
        user_id="user-1",
        trace_id="trace-agent-no-recall",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_http",
        prompt="Write me a deployment guide for this service.",
    )

    assert req.mode == "agent"
    assert req.route_intent == "none"
    assert req.verb is None
    assert req.options["supervised"] is True
    assert req.recall == {
        "enabled": False,
        "required": False,
        "mode": "hybrid",
        "profile": None,
    }
    assert debug["selected_ui_route"] == "agent"
    assert debug["supervised"] is True
    assert debug["recall_enabled"] is False
    assert debug["recall_required"] is False
    assert debug["recall_profile"] is None


def test_agent_mode_recall_toggle_preserves_supervised_routing_intent() -> None:
    enabled_req, enabled_debug, _ = hub_builder.build_chat_request(
        payload={"mode": "agent", "packs": ["executive_pack"], "use_recall": True},
        session_id="sid-agent-enabled",
        user_id="user-1",
        trace_id="trace-agent-enabled",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_http",
        prompt="Write me a deployment guide for this service.",
    )
    disabled_req, disabled_debug, _ = hub_builder.build_chat_request(
        payload={"mode": "agent", "packs": ["executive_pack"], "use_recall": False},
        session_id="sid-agent-disabled",
        user_id="user-1",
        trace_id="trace-agent-disabled",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_http",
        prompt="Write me a deployment guide for this service.",
    )

    assert enabled_req.mode == disabled_req.mode == "agent"
    assert enabled_req.route_intent == disabled_req.route_intent == "none"
    assert enabled_req.options["supervised"] is True
    assert disabled_req.options["supervised"] is True
    assert enabled_debug["supervised"] is True
    assert disabled_debug["supervised"] is True


def test_agent_mode_preserves_explicit_recall_profile_override() -> None:
    req, debug, _ = hub_builder.build_chat_request(
        payload={"mode": "agent", "use_recall": True, "recall_profile": "reflect.v1"},
        session_id="sid-agent-explicit-profile",
        user_id="user-1",
        trace_id="trace-agent-explicit-profile",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_http",
        prompt="Continue from my previous answer.",
    )

    assert req.mode == "agent"
    assert req.recall["enabled"] is True
    assert req.recall["profile"] == "reflect.v1"
    assert debug["recall_profile"] == "reflect.v1"


def test_agent_mode_preserves_bounded_prior_turns_in_messages() -> None:
    history = [
        {"role": "user", "content": "Let's define metacognition lanes for Orion."},
        {"role": "assistant", "content": "Great. We can define perception, reflection, and action lanes."},
        {"role": "user", "content": "well, lanes would you build?"},
    ]
    req, _, _ = hub_builder.build_chat_request(
        payload={"mode": "agent"},
        session_id="sid-agent-continuity",
        user_id="user-1",
        trace_id="trace-agent-continuity",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_http",
        prompt=history[-1]["content"],
        messages=history,
    )

    assert [m.role for m in req.messages] == ["user", "assistant", "user"]
    assert req.messages[0].content == "Let's define metacognition lanes for Orion."
    assert req.messages[-1].content == "well, lanes would you build?"


def test_agent_mode_messages_are_turn_bounded() -> None:
    history = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
    ]
    bounded = hub_builder.build_continuity_messages(
        history=history,
        latest_user_prompt="u3",
        turns=2,
    )
    assert [m["content"] for m in bounded] == ["a1", "u2", "a2", "u3"]


def test_auto_mode_emits_auto_route_intent_without_forcing_supervisor() -> None:
    req, debug, _ = hub_builder.build_chat_request(
        payload={
            "mode": "auto",
            "packs": ["executive_pack", "memory_pack"],
        },
        session_id="sid-auto",
        user_id="user-2",
        trace_id="trace-auto",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_ws",
        prompt="Compare Docker Compose and Kubernetes for this deployment.",
    )

    assert req.mode == "auto"
    assert req.route_intent == "auto"
    assert req.options["route_intent"] == "auto"
    assert req.options.get("supervised") is None
    assert debug["selected_ui_route"] == "auto"
    assert debug["supervised"] is False
    assert debug["packs"] == ["executive_pack", "memory_pack"]
    assert req.metadata["hub_route"]["selected_ui_route"] == "auto"


def test_quick_chat_selection_uses_chat_quick_verb_without_changing_brain_mode_contract() -> None:
    req, debug, _ = hub_builder.build_chat_request(
        payload={
            "mode": "brain",
            "verbs": ["chat_quick"],
        },
        session_id="sid-quick",
        user_id="user-quick",
        trace_id="trace-quick",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_http",
        prompt="Quick check: does this compile?",
    )

    assert req.mode == "brain"
    assert req.verb == "chat_quick"
    assert req.route_intent == "none"
    assert req.options.get("route_intent") is None
    assert debug["verb"] == "chat_quick"


def test_social_room_profile_forces_brain_chat_verb_and_safe_recall() -> None:
    req, debug, _ = hub_builder.build_chat_request(
        payload={
            "mode": "auto",
            "chat_profile": "social_room",
            "concept_evidence": [
                {"ref_id": "identity-1", "source_kind": "memory.identity.snapshot.v1", "summary": "Oríon stayed peer-grounded.", "confidence": 0.8}
            ],
            "social_peer_continuity": {
                "peer_key": "callsyne:room-alpha:peer-1",
                "platform": "callsyne",
                "room_id": "room-alpha",
                "participant_id": "peer-1",
                "participant_name": "CallSyne Peer",
                "aliases": ["Peer"],
                "participant_kind": "peer_ai",
                "recent_shared_topics": ["continuity", "collaboration"],
                "interaction_tone_summary": "warm, direct, grounded",
                "safe_continuity_summary": "Recurring peer who likes synthesis and grounded follow-up.",
                "evidence_refs": ["social-turn-1"],
                "evidence_count": 2,
                "last_seen_at": "2026-03-22T12:00:00+00:00",
                "confidence": 0.6,
                "trust_tier": "known",
            },
            "social_room_continuity": {
                "room_key": "callsyne:room-alpha",
                "platform": "callsyne",
                "room_id": "room-alpha",
                "recurring_topics": ["continuity", "memory"],
                "active_participants": ["CallSyne Peer"],
                "recent_thread_summary": "The room keeps circling grounded collaboration.",
                "room_tone_summary": "Calm, warm, curious.",
                "open_threads": ["How to stay grounded together?"],
                "evidence_refs": ["social-turn-1"],
                "evidence_count": 2,
                "last_updated_at": "2026-03-22T12:00:00+00:00",
            },
            "social_stance_snapshot": {
                "stance_id": "orion-social-room",
                "curiosity": 0.7,
                "warmth": 0.8,
                "directness": 0.65,
                "playfulness": 0.35,
                "caution": 0.4,
                "depth_preference": 0.6,
                "recent_social_orientation_summary": "Recent social stance leans warm, curious, direct.",
                "evidence_refs": ["social-turn-1"],
                "evidence_count": 2,
                "last_updated_at": "2026-03-22T12:00:00+00:00",
            },
            "social_peer_style_hint": {
                "peer_style_key": "callsyne:room-alpha:peer-1",
                "platform": "callsyne",
                "room_id": "room-alpha",
                "participant_id": "peer-1",
                "participant_name": "CallSyne Peer",
                "style_hints_summary": "Prefers direct but grounded replies.",
                "preferred_directness": 0.7,
                "preferred_depth": 0.55,
                "question_appetite": 0.6,
                "playfulness_tendency": 0.25,
                "formality_tendency": 0.45,
                "summarization_preference": 0.4,
                "evidence_count": 3,
                "confidence": 0.6,
                "last_updated_at": "2026-03-22T12:00:00+00:00",
            },
                "social_room_ritual_summary": {
                    "ritual_key": "callsyne:room-alpha",
                    "platform": "callsyne",
                "room_id": "room-alpha",
                "greeting_style": "warm",
                "reentry_style": "grounded",
                "thread_revival_style": "direct",
                "pause_handoff_style": "brief",
                "summary_cadence_preference": 0.4,
                "room_tone_summary": "Calm, warm, curious.",
                "culture_summary": "The room leans warm on greeting and brief on pause.",
                "evidence_count": 3,
                    "confidence": 0.6,
                    "last_updated_at": "2026-03-22T12:00:00+00:00",
                },
                "social_context_window": {
                    "window_id": "context-window-1",
                    "platform": "callsyne",
                    "room_id": "room-alpha",
                    "thread_key": "callsyne:room-alpha:thread:thread-1",
                    "participant_id": "peer-1",
                    "selected_candidates": [
                        {
                            "candidate_id": "candidate-1",
                            "platform": "callsyne",
                            "room_id": "room-alpha",
                            "thread_key": "callsyne:room-alpha:thread:thread-1",
                            "participant_id": "peer-1",
                            "candidate_kind": "peer_continuity",
                            "reference_key": "callsyne:room-alpha:peer-1",
                            "summary": "Recurring peer who likes synthesis and grounded follow-up.",
                            "relevance_score": 0.9,
                            "priority_band": "high",
                            "freshness_band": "fresh",
                            "inclusion_decision": "include",
                            "rationale": "Addressed-peer context should lead.",
                            "reasons": ["addressed_peer_context"],
                            "max_window_budget": 4,
                            "metadata": {"source": "social-memory"},
                        }
                    ],
                    "budget_max": 4,
                    "total_candidates_considered": 1,
                    "rationale": "Compact local-first selection.",
                    "reasons": ["addressed_peer_context_preferred"],
                    "metadata": {"source": "social-memory"},
                },
                "social_context_selection_decision": {
                    "decision_id": "context-decision-1",
                    "platform": "callsyne",
                    "room_id": "room-alpha",
                    "thread_key": "callsyne:room-alpha:thread:thread-1",
                    "selected_candidate_ids": ["candidate-1"],
                    "total_candidates_considered": 1,
                    "included_count": 1,
                    "softened_count": 0,
                    "excluded_count": 0,
                    "budget_max": 4,
                    "rationale": "Compact local-first selection.",
                    "reasons": ["addressed_peer_context_preferred"],
                    "metadata": {"source": "social-memory"},
                },
            },
        session_id="sid-social",
        user_id="user-social",
        trace_id="trace-social",
        default_mode="brain",
        auto_default_enabled=True,
        source_label="hub_ws",
        prompt="Stay with me in a more conversational room.",
    )

    assert req.mode == "brain"
    assert req.route_intent == "none"
    assert req.verb == "chat_social_room"
    assert req.recall == {
        "enabled": True,
        "required": False,
        "mode": "hybrid",
        "profile": "social.room.v1",
    }
    assert req.options["tool_execution_policy"] == "none"
    assert req.options["action_execution_policy"] == "none"
    assert req.metadata["chat_profile"] == "social_room"
    assert req.metadata["social_grounding_state"]["profile"] == "social_room"
    assert req.metadata["social_concept_evidence"][0]["ref_id"] == "identity-1"
    assert req.metadata["social_peer_continuity"]["participant_id"] == "peer-1"
    assert req.metadata["social_room_continuity"]["room_id"] == "room-alpha"
    assert req.metadata["social_stance_snapshot"]["warmth"] == 0.8
    assert req.metadata["social_peer_style_hint"]["participant_id"] == "peer-1"
    assert req.metadata["social_room_ritual_summary"]["room_id"] == "room-alpha"
    assert "Oríon stays" in req.metadata["social_style_adaptation"]["core_identity_anchor"]
    assert req.metadata["social_skill_selection"]["used"] is False
    assert req.metadata["social_skill_selection"]["suppressed_reason"] == "no_skill_needed"
    assert req.metadata["social_skill_result"] is None
    assert debug["chat_profile"] == "social_room"
    assert debug["social_skill_selection"]["suppressed_reason"] == "no_skill_needed"
    assert debug["social_style_adaptation"]["guardrail"].startswith("Adapt lightly")
    assert debug["social_inspection"]["metadata"]["tool_execution_available"] == "false"
    assert any(section["section_kind"] == "context_window" for section in debug["social_inspection"]["sections"])


def test_social_room_epistemic_phrase_hint_is_injected_for_interpretive_turns() -> None:
    req, _, _ = hub_builder.build_chat_request(
        payload={
            "chat_profile": "social_room",
            "social_epistemic_signal": {
                "claim_kind": "inference",
                "confidence_level": "medium",
                "ambiguity_level": "low",
                "source_basis": "social_memory",
            },
            "social_epistemic_decision": {
                "decision": "answer_inference",
                "rationale": "the safest answer is an inference grounded in the visible room context",
            },
            "social_room_continuity": {
                "room_key": "callsyne:room-alpha",
                "platform": "callsyne",
                "room_id": "room-alpha",
                "recurring_topics": ["grounding"],
                "active_participants": ["CallSyne Peer"],
                "recent_thread_summary": "The room is discussing pacing.",
                "room_tone_summary": "Calm and grounded.",
                "open_threads": ["How fast should the room move?"],
                "evidence_refs": ["social-turn-1"],
                "evidence_count": 2,
                "last_updated_at": "2026-03-22T12:00:00+00:00",
                "active_claims": [
                    {
                        "claim_id": "claim-1",
                        "platform": "callsyne",
                        "room_id": "room-alpha",
                        "claim_kind": "peer_claim",
                        "normalized_summary": "The room is moving too fast.",
                        "current_stance": "provisional",
                        "confidence": 0.5,
                        "source_basis": "recent_turns",
                    }
                ],
                "claim_consensus_states": [
                    {
                        "claim_id": "claim-1",
                        "platform": "callsyne",
                        "room_id": "room-alpha",
                        "normalized_claim_key": "The room is moving too fast.",
                        "consensus_state": "partial",
                        "supporting_evidence_count": 2,
                        "orion_stance": "unknown",
                    }
                ],
            },
        },
        session_id="sid-epistemic-hint",
        user_id="user-social",
        trace_id="trace-epistemic-hint",
        default_mode="brain",
        auto_default_enabled=True,
        source_label="hub_ws",
        prompt="What's your read on why the room keeps circling this?",
    )

    assert req.metadata["social_epistemic_phrase_hint"]["lead_in"].startswith("Lead naturally with an interpretive frame")
    assert "read" in req.metadata["social_epistemic_phrase_hint"]["caution"].lower()


def test_social_room_clarifying_epistemic_hint_stays_compact() -> None:
    req, _, _ = hub_builder.build_chat_request(
        payload={
            "chat_profile": "social_room",
            "social_epistemic_signal": {
                "claim_kind": "clarification_needed",
                "confidence_level": "medium",
                "ambiguity_level": "high",
            },
            "social_epistemic_decision": {
                "decision": "ask_clarifying_question",
                "rationale": "clarify target before asserting",
            },
            "social_room_continuity": {
                "room_key": "callsyne:room-alpha",
                "platform": "callsyne",
                "room_id": "room-alpha",
                "recurring_topics": ["grounding"],
                "active_participants": ["CallSyne Peer"],
                "recent_thread_summary": "The room is discussing pacing.",
                "room_tone_summary": "Calm and grounded.",
                "open_threads": ["How fast should the room move?"],
                "evidence_refs": ["social-turn-1"],
                "evidence_count": 2,
                "last_updated_at": "2026-03-22T12:00:00+00:00",
            },
        },
        session_id="sid-epistemic-clarify",
        user_id="user-social",
        trace_id="trace-epistemic-clarify",
        default_mode="brain",
        auto_default_enabled=True,
        source_label="hub_ws",
        prompt="Wait, which thread do you mean?",
    )

    assert req.metadata["social_epistemic_phrase_hint"]["lead_in"] == "Ask one short clarifying question first."
    assert "without preamble" not in req.metadata["social_epistemic_phrase_hint"]["lead_in"].lower()
    assert req.metadata["social_epistemic_phrase_hint"]["caution"] == "Clarify scope, thread, or target before making a claim."


def test_social_room_floor_metadata_is_injected_without_enabling_tools() -> None:
    req, _, _ = hub_builder.build_chat_request(
        payload={
            "chat_profile": "social_room",
            "social_room_continuity": {
                "room_key": "callsyne:room-alpha",
                "platform": "callsyne",
                "room_id": "room-alpha",
                "recurring_topics": ["pacing"],
                "active_participants": ["Cadence"],
                "recent_thread_summary": "The room is comparing pacing reads.",
                "room_tone_summary": "Calm and direct.",
                "open_threads": ["Does Cadence agree with the bridge summary?"],
                "evidence_refs": ["social-turn-1"],
                "evidence_count": 1,
                "last_updated_at": "2026-03-22T12:00:00+00:00",
                "turn_handoff": {
                    "platform": "callsyne",
                    "room_id": "room-alpha",
                    "thread_key": "callsyne:room-alpha:thread:1",
                    "audience_scope": "peer",
                    "target_participant_name": "Cadence",
                    "decision_kind": "yield_to_peer",
                    "handoff_text": "Cadence, does that match your read?",
                    "rationale": "bridge summary should yield back to the relevant peer",
                    "confidence": 0.74,
                },
                "closure_signal": {
                    "platform": "callsyne",
                    "room_id": "room-alpha",
                    "thread_key": "callsyne:room-alpha:thread:1",
                    "audience_scope": "thread",
                    "closure_kind": "left_open",
                    "resolved": False,
                    "closure_text": "I’ll leave that open for an answer.",
                    "rationale": "the floor should stay open for the peer response",
                    "confidence": 0.58,
                },
                "floor_decision": {
                    "platform": "callsyne",
                    "room_id": "room-alpha",
                    "thread_key": "callsyne:room-alpha:thread:1",
                    "audience_scope": "peer",
                    "target_participant_name": "Cadence",
                    "decision_kind": "yield_to_peer",
                    "rationale": "bridge summary should yield back to the relevant peer",
                    "confidence": 0.74,
                },
            },
        },
        session_id="sid-floor",
        user_id="user-floor",
        trace_id="trace-floor",
        default_mode="brain",
        auto_default_enabled=True,
        source_label="hub_ws",
        prompt="Where are we landing on pacing?",
    )

    assert req.options["tool_execution_policy"] == "none"
    assert req.options["action_execution_policy"] == "none"
    assert req.metadata["social_turn_handoff"]["target_participant_name"] == "Cadence"
    assert req.metadata["social_closure_signal"]["closure_kind"] == "left_open"
    assert req.metadata["social_floor_decision"]["decision_kind"] == "yield_to_peer"


def test_social_room_skill_metadata_is_injected_for_summary_recall_and_identity_prompts() -> None:
    summarize_req, summarize_debug, _ = hub_builder.build_chat_request(
        payload={
            "chat_profile": "social_room",
            "social_room_continuity": {
                "room_key": "callsyne:room-alpha",
                "platform": "callsyne",
                "room_id": "room-alpha",
                "recurring_topics": ["continuity"],
                "active_participants": ["CallSyne Peer"],
                "recent_thread_summary": "The room is threading grounded collaboration.",
                "room_tone_summary": "Warm and curious.",
                "open_threads": ["How do we stay grounded together?"],
                "evidence_refs": ["social-turn-1"],
                "evidence_count": 2,
                "last_updated_at": "2026-03-22T12:00:00+00:00",
            },
        },
        session_id="sid-skill-sum",
        user_id="user-social",
        trace_id="trace-skill-sum",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_http",
        prompt="Can you summarize what we were just talking about?",
    )
    recall_req, _, _ = hub_builder.build_chat_request(
        payload={
            "chat_profile": "social_room",
            "social_peer_continuity": {
                "peer_key": "callsyne:room-alpha:peer-1",
                "platform": "callsyne",
                "room_id": "room-alpha",
                "participant_id": "peer-1",
                "participant_name": "CallSyne Peer",
                "aliases": [],
                "participant_kind": "peer_ai",
                "recent_shared_topics": ["continuity"],
                "interaction_tone_summary": "warm",
                "safe_continuity_summary": "You like grounded synthesis.",
                "evidence_refs": ["social-turn-1"],
                "evidence_count": 1,
                "last_seen_at": "2026-03-22T12:00:00+00:00",
                "confidence": 0.4,
                "trust_tier": "new",
            },
            "social_room_continuity": {
                "room_key": "callsyne:room-alpha",
                "platform": "callsyne",
                "room_id": "room-alpha",
                "recurring_topics": ["continuity"],
                "active_participants": ["CallSyne Peer"],
                "recent_thread_summary": "private mirror detail should never surface",
                "room_tone_summary": "Warm and curious.",
                "open_threads": [],
                "evidence_refs": ["social-turn-1"],
                "evidence_count": 1,
                "last_updated_at": "2026-03-22T12:00:00+00:00",
            },
        },
        session_id="sid-skill-recall",
        user_id="user-social",
        trace_id="trace-skill-recall",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_http",
        prompt="What do you remember about me here?",
    )
    self_req, _, _ = hub_builder.build_chat_request(
        payload={"chat_profile": "social_room"},
        session_id="sid-skill-self",
        user_id="user-social",
        trace_id="trace-skill-self",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_http",
        prompt="Who are you in this room?",
    )

    assert summarize_req.metadata["social_skill_selection"]["selected_skill"] == "social_summarize_thread"
    assert "grounded collaboration" in summarize_req.metadata["social_skill_result"]["summary"].lower()
    assert summarize_debug["social_skill_result"]["skill_name"] == "social_summarize_thread"
    assert recall_req.metadata["social_skill_selection"]["selected_skill"] == "social_safe_recall"
    assert "private" not in " ".join(recall_req.metadata["social_skill_result"]["snippets"]).lower()
    assert self_req.metadata["social_skill_selection"]["selected_skill"] == "social_self_ground"
    assert "Oríon" in self_req.metadata["social_skill_result"]["summary"]


def test_social_room_skill_metadata_supports_followup_reflection_pause_and_disable() -> None:
    follow_req, _, _ = hub_builder.build_chat_request(
        payload={
            "chat_profile": "social_room",
            "social_turn_policy": {"decision": "ask_follow_up", "novelty_score": 0.1},
            "social_room_continuity": {
                "room_key": "callsyne:room-alpha",
                "platform": "callsyne",
                "room_id": "room-alpha",
                "recurring_topics": ["continuity"],
                "active_participants": ["CallSyne Peer"],
                "recent_thread_summary": "The room is staying with one live thread.",
                "room_tone_summary": "Warm and searching.",
                "open_threads": ["how to stay grounded together"],
                "evidence_refs": ["social-turn-1"],
                "evidence_count": 2,
                "last_updated_at": "2026-03-22T12:00:00+00:00",
            },
        },
        session_id="sid-skill-follow",
        user_id="user-social",
        trace_id="trace-skill-follow",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_http",
        prompt="I'm not sure where to go next.",
    )
    reflect_req, _, _ = hub_builder.build_chat_request(
        payload={
            "chat_profile": "social_room",
            "social_room_continuity": {
                "room_key": "callsyne:room-alpha",
                "platform": "callsyne",
                "room_id": "room-alpha",
                "recurring_topics": ["continuity"],
                "active_participants": ["CallSyne Peer"],
                "recent_thread_summary": "The room is staying with one live thread.",
                "room_tone_summary": "Warm and searching.",
                "open_threads": [],
                "evidence_refs": ["social-turn-1"],
                "evidence_count": 2,
                "last_updated_at": "2026-03-22T12:00:00+00:00",
            },
        },
        session_id="sid-skill-reflect",
        user_id="user-social",
        trace_id="trace-skill-reflect",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_http",
        prompt="What do you notice about this room?",
    )
    pause_req, _, _ = hub_builder.build_chat_request(
        payload={"chat_profile": "social_room"},
        session_id="sid-skill-pause",
        user_id="user-social",
        trace_id="trace-skill-pause",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_http",
        prompt="Let's pause here for now.",
    )
    disabled_req, disabled_debug, _ = hub_builder.build_chat_request(
        payload={
            "chat_profile": "social_room",
            "social_skill_selection_config": {"enabled": False},
        },
        session_id="sid-skill-disabled",
        user_id="user-social",
        trace_id="trace-skill-disabled",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_http",
        prompt="Can you summarize this room?",
    )

    assert follow_req.metadata["social_skill_selection"]["selected_skill"] == "social_followup_question"
    assert "?" in follow_req.metadata["social_skill_result"]["summary"]
    assert reflect_req.metadata["social_skill_selection"]["selected_skill"] == "social_room_reflection"
    assert pause_req.metadata["social_skill_selection"]["selected_skill"] == "social_exit_or_pause"
    assert disabled_req.metadata["social_skill_selection"]["suppressed_reason"] == "skills_disabled"
    assert disabled_req.metadata["social_skill_result"] is None
    assert disabled_debug["social_skill_selection"]["suppressed_reason"] == "skills_disabled"


def test_social_room_style_adaptation_can_be_disabled_or_confidence_gated() -> None:
    disabled_req, _, _ = hub_builder.build_chat_request(
        payload={
            "chat_profile": "social_room",
            "social_style_config": {"enabled": False},
            "social_peer_style_hint": {
                "peer_style_key": "callsyne:room-alpha:peer-1",
                "platform": "callsyne",
                "room_id": "room-alpha",
                "participant_id": "peer-1",
                "participant_name": "CallSyne Peer",
                "style_hints_summary": "Prefers direct but grounded replies.",
                "preferred_directness": 0.8,
                "preferred_depth": 0.6,
                "question_appetite": 0.6,
                "playfulness_tendency": 0.3,
                "formality_tendency": 0.5,
                "summarization_preference": 0.45,
                "evidence_count": 3,
                "confidence": 0.7,
                "last_updated_at": "2026-03-22T12:00:00+00:00",
            },
        },
        session_id="sid-style-off",
        user_id="user-social",
        trace_id="trace-style-off",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_http",
        prompt="Hello there.",
    )
    gated_req, _, _ = hub_builder.build_chat_request(
        payload={
            "chat_profile": "social_room",
            "social_style_config": {"confidence_floor": 0.9},
            "social_peer_style_hint": {
                "peer_style_key": "callsyne:room-alpha:peer-1",
                "platform": "callsyne",
                "room_id": "room-alpha",
                "participant_id": "peer-1",
                "participant_name": "CallSyne Peer",
                "style_hints_summary": "Prefers direct but grounded replies.",
                "preferred_directness": 0.8,
                "preferred_depth": 0.6,
                "question_appetite": 0.6,
                "playfulness_tendency": 0.3,
                "formality_tendency": 0.5,
                "summarization_preference": 0.45,
                "evidence_count": 3,
                "confidence": 0.7,
                "last_updated_at": "2026-03-22T12:00:00+00:00",
            },
        },
        session_id="sid-style-gated",
        user_id="user-social",
        trace_id="trace-style-gated",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_http",
        prompt="Hello there.",
    )

    assert "disabled" in disabled_req.metadata["social_style_adaptation"]["guardrail"].lower()
    assert gated_req.metadata["social_style_adaptation"]["peer_adaptation_hint"] == ""


def test_social_room_artifact_dialogue_metadata_defaults_to_narrow_scope() -> None:
    req, debug, _ = hub_builder.build_chat_request(
        payload={"chat_profile": "social_room"},
        session_id="sid-artifact",
        user_id="user-social",
        trace_id="trace-artifact",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_http",
        prompt="Can you keep a short takeaway from this?",
    )

    assert req.options["tool_execution_policy"] == "none"
    assert req.options["action_execution_policy"] == "none"
    assert req.metadata["social_artifact_proposal"]["proposed_scope"] == "session_only"
    assert req.metadata["social_artifact_proposal"]["confirmation_needed"] is True
    assert req.metadata["social_skill_selection"]["selected_skill"] == "social_artifact_dialogue"
    assert "session-only" in req.metadata["social_skill_result"]["summary"]
    assert debug["social_artifact_proposal"]["decision_state"] == "clarify_scope"


def test_social_room_artifact_dialogue_can_confirm_pending_proposal() -> None:
    req, debug, _ = hub_builder.build_chat_request(
        payload={
            "chat_profile": "social_room",
            "social_artifact_proposal": {
                "proposal_id": "proposal-1",
                "artifact_type": "room_norm",
                "proposed_summary_text": "grounded collaboration cue",
                "proposed_scope": "room_local",
                "decision_state": "proposed",
                "confirmation_needed": True,
                "rationale": "narrow room-local carry-forward request",
            },
        },
        session_id="sid-artifact-confirm",
        user_id="user-social",
        trace_id="trace-artifact-confirm",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_http",
        prompt="Yes, that works room-local.",
    )

    assert req.metadata["social_artifact_confirmation"]["decision_state"] == "accepted"
    assert req.metadata["social_artifact_confirmation"]["confirmed_scope"] == "room_local"
    assert req.metadata["social_skill_selection"]["selected_skill"] == "social_artifact_dialogue"
    assert "carry forward" in req.metadata["social_skill_result"]["summary"].lower()
    assert debug["social_artifact_confirmation"]["decision_state"] == "accepted"
