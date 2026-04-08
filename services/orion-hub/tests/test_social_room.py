from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "social_room.py"
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SPEC = importlib.util.spec_from_file_location("hub_social_room", MODULE_PATH)
assert SPEC and SPEC.loader
hub_social_room = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hub_social_room)


def test_social_room_client_meta_preserves_external_room_context() -> None:
    meta = hub_social_room.social_room_client_meta(
        payload={
            "chat_profile": "social_room",
            "external_room": {
                "platform": "callsyne",
                "room_id": "room-alpha",
                "transport_message_id": "msg-1",
            },
            "external_participant": {
                "participant_id": "peer-1",
                "participant_name": "CallSyne Peer",
                "participant_kind": "peer_ai",
            },
            "social_peer_continuity": {
                "peer_key": "callsyne:room-alpha:peer-1",
                "platform": "callsyne",
                "room_id": "room-alpha",
                "participant_id": "peer-1",
                "participant_name": "CallSyne Peer",
                "aliases": [],
                "participant_kind": "peer_ai",
                "recent_shared_topics": ["continuity"],
                "interaction_tone_summary": "warm, direct, grounded",
                "safe_continuity_summary": "Recurring peer.",
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
                "recent_thread_summary": "Grounded collaboration.",
                "room_tone_summary": "Warm and curious.",
                "open_threads": ["grounding"],
                "evidence_refs": ["social-turn-1"],
                "evidence_count": 1,
                "last_updated_at": "2026-03-22T12:00:00+00:00",
            },
            "social_stance_snapshot": {
                "stance_id": "orion-social-room",
                "curiosity": 0.7,
                "warmth": 0.8,
                "directness": 0.65,
                "playfulness": 0.3,
                "caution": 0.4,
                "depth_preference": 0.6,
                "recent_social_orientation_summary": "Recent social stance leans warm, curious, direct.",
                "evidence_refs": ["social-turn-1"],
                "evidence_count": 1,
                "last_updated_at": "2026-03-22T12:00:00+00:00",
            },
            "social_skill_selection": {
                "considered_skills": ["social_summarize_thread"],
                "selected_skill": "social_summarize_thread",
                "used": True,
                "selection_reason": "explicit request to summarize the room/thread",
            },
            "social_skill_result": {
                "skill_name": "social_summarize_thread",
                "used": True,
                "summary": "Grounded collaboration is the current thread.",
                "snippets": ["Grounded collaboration is the current thread."],
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
                "room_tone_summary": "Warm and curious.",
                "culture_summary": "The room leans warm on greeting and brief on pause.",
                "evidence_count": 3,
                "confidence": 0.6,
                "last_updated_at": "2026-03-22T12:00:00+00:00",
            },
            "social_style_adaptation": {
                "snapshot_id": "social-style:room-alpha:peer-1",
                "platform": "callsyne",
                "room_id": "room-alpha",
                "participant_id": "peer-1",
                "core_identity_anchor": "Oríon stays warm, direct, grounded as a peer.",
                "peer_adaptation_hint": "Prefers direct but grounded replies.",
                "room_ritual_hint": "Use warm greeting cues and brief pause/handoff.",
                "directness_delta": 0.07,
                "depth_delta": 0.02,
                "question_frequency_delta": 0.03,
                "playfulness_delta": -0.02,
                "summarization_tendency_delta": 0.02,
                "guardrail": "Adapt lightly to the peer and room while remaining Orion.",
                "confidence": 0.6,
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
                        "summary": "Recurring peer.",
                        "relevance_score": 0.9,
                        "priority_band": "high",
                        "freshness_band": "fresh",
                        "inclusion_decision": "include",
                        "rationale": "Addressed-peer context should lead.",
                        "reasons": ["addressed_peer_context"],
                        "max_window_budget": 4,
                        "metadata": {"source": "social-memory"}
                    }
                ],
                "budget_max": 4,
                "total_candidates_considered": 1,
                "rationale": "Compact local-first selection.",
                "reasons": ["addressed_peer_context_preferred"],
                "metadata": {"source": "social-memory"}
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
                "metadata": {"source": "social-memory"}
            },
            "social_episode_snapshot": {
                "snapshot_id": "callsyne:room-alpha:thread-1:episode",
                "platform": "callsyne",
                "room_id": "room-alpha",
                "thread_key": "callsyne:room-alpha:thread:thread-1",
                "participant_id": "peer-1",
                "summary": "The last coherent exchange was about grounded collaboration.",
                "resumptive_hint": "Resume from grounded collaboration if the room is still on that thread.",
                "focus_topics": ["continuity", "grounding"],
                "last_active_at": "2026-03-22T12:00:00+00:00",
                "freshness_band": "fresh",
                "superseded_by_live_state": False,
                "rationale": "Compact resumptive snapshot.",
                "metadata": {"source": "social-memory"},
            },
            "social_reentry_anchor": {
                "anchor_id": "callsyne:room-alpha:thread-1:reentry",
                "platform": "callsyne",
                "room_id": "room-alpha",
                "thread_key": "callsyne:room-alpha:thread:thread-1",
                "participant_id": "peer-1",
                "source_snapshot_id": "callsyne:room-alpha:thread-1:episode",
                "anchor_text": "Use a grounded re-entry: briefly name grounded collaboration and check whether that is still where the room is.",
                "freshness_band": "fresh",
                "reentry_style": "grounded",
                "rationale": "Re-entry anchors remain subordinate to live state.",
                "created_at": "2026-03-22T12:00:00+00:00",
                "metadata": {"source": "social-memory"},
            },
        },
        route_debug={"recall_profile": "social.room.v1"},
        trace_verb="chat_social_room",
        memory_digest="memory",
    )

    assert meta["external_room"]["platform"] == "callsyne"
    assert meta["external_participant"]["participant_id"] == "peer-1"
    assert meta["social_peer_continuity"]["participant_id"] == "peer-1"
    assert meta["social_room_continuity"]["room_id"] == "room-alpha"
    assert meta["social_stance_snapshot"]["warmth"] == 0.8
    assert meta["social_skill_selection"]["selected_skill"] == "social_summarize_thread"
    assert meta["social_skill_result"]["skill_name"] == "social_summarize_thread"
    assert meta["social_peer_style_hint"]["participant_id"] == "peer-1"
    assert meta["social_room_ritual_summary"]["room_id"] == "room-alpha"
    assert meta["social_style_adaptation"]["participant_id"] == "peer-1"
    assert meta["social_context_window"]["selected_candidates"][0]["candidate_kind"] == "peer_continuity"
    assert meta["social_context_selection_decision"]["budget_max"] == 4
    assert meta["social_episode_snapshot"]["summary"].startswith("The last coherent exchange")
    assert meta["social_reentry_anchor"]["reentry_style"] == "grounded"


def test_social_room_skill_selector_defaults_to_no_skill_when_not_needed() -> None:
    selection, result, request = hub_social_room.select_social_room_skill(
        payload={"messages": [{"role": "user", "content": "I’m with you."}]},
        prompt="I’m with you.",
        skills_enabled=True,
        allowlist=hub_social_room.resolve_social_skill_allowlist(),
    )

    assert request.profile == "social_room"
    assert selection.used is False
    assert selection.suppressed_reason == "no_skill_needed"
    assert result is None


def test_shared_artifact_dialogue_defaults_to_session_only_when_scope_is_ambiguous() -> None:
    proposal, revision, confirmation, result, reason = hub_social_room.build_social_artifact_dialogue(
        payload={},
        prompt="Can you keep a short takeaway from this?",
    )

    assert proposal is not None
    assert revision is None
    assert confirmation is None
    assert proposal.proposed_scope == "session_only"
    assert proposal.confirmation_needed is True
    assert proposal.decision_state == "clarify_scope"
    assert result is not None and "session-only" in result.summary
    assert "clarification" in reason or "proposal" in reason


def test_shared_artifact_dialogue_can_revise_and_narrow_a_pending_proposal() -> None:
    proposal, revision, confirmation, result, _ = hub_social_room.build_social_artifact_dialogue(
        payload={
            "social_artifact_proposal": {
                "proposal_id": "proposal-1",
                "artifact_type": "room_norm",
                "proposed_summary_text": "grounded collaboration with warm pacing in this room",
                "proposed_scope": "room_local",
                "decision_state": "proposed",
                "confirmation_needed": True,
                "rationale": "starting room-local draft",
            }
        },
        prompt="Can you make that shorter and safer?",
    )

    assert proposal is not None
    assert revision is not None
    assert confirmation is None
    assert revision.revised_scope == "session_only"
    assert len(revision.revised_summary_text.split()) <= len(proposal.proposed_summary_text.split())
    assert result is not None and "Shorter version" in result.summary


def test_shared_artifact_dialogue_can_confirm_a_pending_proposal() -> None:
    proposal, revision, confirmation, result, _ = hub_social_room.build_social_artifact_dialogue(
        payload={
            "social_artifact_proposal": {
                "proposal_id": "proposal-1",
                "artifact_type": "room_norm",
                "proposed_summary_text": "grounded collaboration cue",
                "proposed_scope": "room_local",
                "decision_state": "proposed",
                "confirmation_needed": True,
                "rationale": "starting room-local draft",
            }
        },
        prompt="Yes, that works room-local.",
    )

    assert proposal is not None
    assert revision is None
    assert confirmation is not None
    assert confirmation.decision_state == "accepted"
    assert confirmation.confirmed_scope == "room_local"
    assert result is not None and "room-local" in result.summary


def test_social_room_skill_selector_uses_thread_summary_skill() -> None:
    selection, result, _ = hub_social_room.select_social_room_skill(
        payload={
            "messages": [{"role": "user", "content": "Can you summarize where we are?"}],
            "social_room_continuity": {
                "recent_thread_summary": "The room is comparing grounded collaboration and pacing.",
                "open_threads": ["How do we pace this well?"],
            },
        },
        prompt="Can you summarize where we are?",
        skills_enabled=True,
        allowlist=hub_social_room.resolve_social_skill_allowlist(),
    )

    assert selection.selected_skill == "social_summarize_thread"
    assert selection.used is True
    assert result is not None
    assert "grounded collaboration" in result.summary.lower()


def test_social_room_skill_selector_blocks_private_material_in_safe_recall() -> None:
    selection, result, _ = hub_social_room.select_social_room_skill(
        payload={
            "messages": [{"role": "user", "content": "What do you remember about me?"}],
            "social_peer_continuity": {
                "safe_continuity_summary": "You like grounded synthesis.",
            },
            "social_room_continuity": {
                "recent_thread_summary": "private mirror detail should never surface",
            },
        },
        prompt="What do you remember about me?",
        skills_enabled=True,
        allowlist=hub_social_room.resolve_social_skill_allowlist(),
    )

    assert selection.selected_skill == "social_safe_recall"
    assert result is not None
    assert "private" not in " ".join(result.snippets).lower()
    assert any("suppressed" in note for note in result.safety_notes)


def test_social_room_skill_selector_handles_shared_artifact_dialogue() -> None:
    selection, result, _ = hub_social_room.select_social_room_skill(
        payload={},
        prompt="Can you keep a short takeaway from this?",
        skills_enabled=True,
        allowlist=hub_social_room.resolve_social_skill_allowlist(),
    )

    assert selection.selected_skill == "social_artifact_dialogue"
    assert result is not None
    assert "session-only" in result.summary


def test_shared_artifact_dialogue_blocks_private_material() -> None:
    proposal, revision, confirmation, result, _ = hub_social_room.build_social_artifact_dialogue(
        payload={},
        prompt="Can you keep this private journal detail?",
    )

    assert proposal is None
    assert revision is None
    assert confirmation is not None and confirmation.decision_state == "declined"
    assert result is not None
    assert "turn that into a shared artifact" in result.summary.lower()


def test_social_room_skill_selector_handles_self_ground_followup_reflection_and_pause() -> None:
    allowlist = hub_social_room.resolve_social_skill_allowlist()

    self_selection, self_result, _ = hub_social_room.select_social_room_skill(
        payload={},
        prompt="Who are you, exactly?",
        skills_enabled=True,
        allowlist=allowlist,
    )
    follow_selection, follow_result, _ = hub_social_room.select_social_room_skill(
        payload={
            "social_turn_policy": {"decision": "ask_follow_up", "novelty_score": 0.1},
            "social_room_continuity": {"open_threads": ["how to stay grounded together"]},
        },
        prompt="I’m not sure where to go next.",
        skills_enabled=True,
        allowlist=allowlist,
    )
    reflection_selection, reflection_result, _ = hub_social_room.select_social_room_skill(
        payload={"social_room_continuity": {"room_tone_summary": "Warm and searching."}},
        prompt="What do you notice about this room?",
        skills_enabled=True,
        allowlist=allowlist,
    )
    pause_selection, pause_result, _ = hub_social_room.select_social_room_skill(
        payload={},
        prompt="Let’s pause here for now.",
        skills_enabled=True,
        allowlist=allowlist,
    )

    assert self_selection.selected_skill == "social_self_ground"
    assert self_result is not None and "Oríon" in self_result.summary
    assert follow_selection.selected_skill == "social_followup_question"
    assert follow_result is not None and "?" in follow_result.summary
    assert reflection_selection.selected_skill == "social_room_reflection"
    assert reflection_result is not None and "Warm and searching" in reflection_result.summary
    assert pause_selection.selected_skill == "social_exit_or_pause"
    assert pause_result is not None and "pause" in pause_result.summary.lower()


def test_style_adaptation_snapshot_is_bounded_and_identity_preserving() -> None:
    snapshot = hub_social_room.build_style_adaptation_snapshot(
        payload={
            "external_room": {"platform": "callsyne", "room_id": "room-alpha"},
            "external_participant": {"participant_id": "peer-1"},
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
                "confidence": 0.65,
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
                "summary_cadence_preference": 0.45,
                "room_tone_summary": "Warm and curious.",
                "culture_summary": "The room leans warm on greeting and brief on pause.",
                "evidence_count": 3,
                "confidence": 0.7,
                "last_updated_at": "2026-03-22T12:00:00+00:00",
            },
        },
        confidence_floor=0.35,
        adaptation_enabled=True,
        rituals_enabled=True,
    )

    assert "remaining Orion" in snapshot.guardrail
    assert abs(snapshot.directness_delta) <= 0.35
    assert abs(snapshot.depth_delta) <= 0.35
    assert abs(snapshot.question_frequency_delta) <= 0.35
    assert abs(snapshot.playfulness_delta) <= 0.35
    assert abs(snapshot.summarization_tendency_delta) <= 0.35
    assert "Oríon" in snapshot.core_identity_anchor
