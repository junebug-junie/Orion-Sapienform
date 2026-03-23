from __future__ import annotations

from orion.inspection.social import build_social_inspection_snapshot


def _surfaces() -> dict:
    return {
        "social_peer_continuity": {
            "participant_id": "peer-1",
            "peer_calibration": {
                "calibration_kind": "reliable_continuity",
                "rationale": "Repeated grounded exchanges justify lighter continuity assumptions.",
            },
            "memory_freshness": [
                {
                    "artifact_kind": "peer_calibration",
                    "freshness_state": "stale",
                    "rationale": "Peer calibration is aging and should be softened.",
                }
            ],
        },
        "social_room_continuity": {
            "platform": "callsyne",
            "room_id": "room-alpha",
            "current_thread_key": "callsyne:room-alpha:thread:thread-1",
            "current_thread_summary": "The room is deciding how grounded pacing should feel.",
            "gif_usage_state": {
                "usage_state_id": "gif-usage-1",
                "platform": "callsyne",
                "room_id": "room-alpha",
                "thread_key": "callsyne:room-alpha:thread:thread-1",
                "consecutive_gif_turns": 0,
                "turns_since_last_orion_gif": 4,
                "recent_gif_density": 0.1,
                "recent_gif_turn_count": 1,
                "recent_turn_window_size": 10,
                "orion_turn_count": 6,
                "recent_turn_was_gif": [False, True, False],
                "recent_intent_kinds": ["laugh_with", "celebrate"],
                "recent_target_participant_ids": ["peer-1", "peer-2"],
                "recent_target_participant_names": ["peer-1", "peer-2"],
                "last_intent_kind": "celebrate",
            },
            "active_claims": [
                {"normalized_summary": "The room is moving too fast."},
            ],
            "recent_claim_revisions": [
                {"revised_summary": "The pacing concern was narrowed to this thread."},
            ],
            "claim_divergence_signals": [
                {"claim_id": "claim-1", "normalized_claim_key": "The room is moving too fast."},
            ],
            "claim_consensus_states": [
                {"normalized_claim_key": "Everyone prefers slower pacing."},
            ],
            "active_commitments": [
                {"summary": "Return with a shorter pacing summary.", "state": "open"},
            ],
            "bridge_summary": {
                "summary_text": "Shared core: the room wants grounded pacing with less drift.",
            },
            "clarifying_question": {"question_text": "Do you want slower pacing just for this thread?"},
            "deliberation_decision": {"decision_kind": "clarify_before_bridge"},
            "turn_handoff": {"handoff_text": "Let me answer the pacing question directly, then hand back."},
            "closure_signal": {"closure_text": "We can pause here if that helps."},
            "floor_decision": {"decision_kind": "hold_floor_briefly"},
            "trust_boundaries": [
                {"rationale": "Keep shared claims provisional until refreshed."},
            ],
            "memory_freshness": [
                {
                    "artifact_kind": "claim_consensus",
                    "freshness_state": "refresh_needed",
                    "rationale": "Older consensus should be refreshed before it is treated as settled.",
                }
            ],
        },
        "social_context_window": {
            "thread_key": "callsyne:room-alpha:thread:thread-1",
            "selected_candidates": [
                {
                    "candidate_kind": "thread",
                    "summary": "The room is deciding how grounded pacing should feel.",
                    "inclusion_decision": "include",
                    "freshness_band": "fresh",
                    "priority_band": "critical",
                    "reference_key": "thread-1",
                    "rationale": "Current thread should govern.",
                },
                {
                    "candidate_kind": "episode_snapshot",
                    "summary": "Earlier exchange also focused on grounded pacing.",
                    "inclusion_decision": "soften",
                    "freshness_band": "aging",
                    "priority_band": "low",
                    "reference_key": "episode-1",
                    "rationale": "Useful background, but not live state.",
                },
            ],
            "total_candidates_considered": 5,
        },
        "social_context_selection_decision": {
            "decision_id": "decision-1",
            "budget_max": 4,
            "rationale": "Compact window keeps live thread state in focus.",
        },
        "social_context_candidates": [
            {
                "candidate_kind": "episode_snapshot",
                "summary": "Earlier exchange also focused on grounded pacing.",
                "inclusion_decision": "soften",
                "freshness_band": "aging",
            },
            {
                "candidate_kind": "consensus",
                "summary": "Everyone prefers slower pacing.",
                "inclusion_decision": "exclude",
                "freshness_band": "refresh_needed",
            },
        ],
        "social_thread_routing": {
            "routing_decision": "reply_to_peer",
            "thread_summary": "Answer the pacing question in-thread.",
            "rationale": "The message is locally addressed and best answered in-thread.",
        },
        "social_repair_signal": {
            "repair_need": "minor_misalignment",
            "repair_summary": "The room needs a small pacing repair before continuing.",
        },
        "social_repair_decision": {
            "decision_kind": "repair_then_answer",
            "rationale": "Repairing pacing first keeps the turn grounded.",
        },
        "social_epistemic_signal": {
            "signal_kind": "interpretive",
            "signal_summary": "The read is interpretive rather than factual.",
        },
        "social_epistemic_decision": {
            "decision_kind": "answer_with_interpretive_frame",
            "rationale": "Use a modest confidence frame because the room state is partly inferred.",
        },
        "social_episode_snapshot": {
            "summary": "The last coherent exchange was about grounded pacing.",
            "resumptive_hint": "Resume from grounded pacing if the room is still there.",
        },
        "social_reentry_anchor": {
            "anchor_text": "Use a grounded re-entry and check whether pacing is still the live issue.",
        },
        "social_artifact_proposal": {
            "decision_state": "clarify_scope",
            "proposed_summary_text": "sealed private note",
        },
        "social_gif_policy": {
            "policy_id": "gif-policy-1",
            "decision_kind": "text_only",
            "gif_allowed": False,
            "intent_kind": "laugh_with",
            "rationale": "The room keeps GIFs subordinate to text.",
            "reasons": ["gif_intent_loop_detected", "fresh_caution_context_prefers_text"],
        },
        "social_gif_intent": {
            "gif_query": "warm laugh with you reaction gif",
        },
        "social_gif_observed_signal": {
            "media_present": True,
            "provider": "giphy",
            "transport_source": "callsyne-webhook",
        },
        "social_gif_proxy_context": {
            "proxy_inputs_present": ["query", "title", "surrounding_text"],
        },
        "social_gif_interpretation": {
            "interpretation_id": "gif-proxy-1",
            "reaction_class": "laugh_with",
            "confidence_level": "low",
            "ambiguity_level": "medium",
            "cue_disposition": "softened",
            "rationale": "Metadata suggests laughter, but the cue stays soft.",
        },
    }


def test_social_inspection_snapshot_includes_selected_context_window_and_stale_exclusions() -> None:
    snapshot = build_social_inspection_snapshot(
        platform="callsyne",
        room_id="room-alpha",
        participant_id="peer-1",
        thread_key="callsyne:room-alpha:thread:thread-1",
        surfaces=_surfaces(),
        source_surface="test",
        source_service="pytest",
    )

    by_kind = {section.section_kind: section for section in snapshot.sections}
    assert "context_window" in by_kind
    assert any("thread:" in item for item in by_kind["context_window"].included_artifact_summaries)
    assert any("consensus:" in item for item in by_kind["context_window"].excluded_state)
    assert snapshot.metadata["tool_execution_available"] == "false"
    assert snapshot.metadata["action_execution_available"] == "false"


def test_social_inspection_snapshot_compacts_claims_commitments_routing_repair_and_epistemic() -> None:
    snapshot = build_social_inspection_snapshot(
        platform="callsyne",
        room_id="room-alpha",
        participant_id="peer-1",
        thread_key="callsyne:room-alpha:thread:thread-1",
        surfaces=_surfaces(),
        source_surface="test",
        source_service="pytest",
    )

    by_kind = {section.section_kind: section for section in snapshot.sections}
    assert any("claim:" in item for item in by_kind["claims"].included_artifact_summaries)
    assert any("Return with a shorter pacing summary." in item for item in by_kind["commitments"].included_artifact_summaries)
    assert any("Answer the pacing question in-thread." in item for item in by_kind["routing"].included_artifact_summaries)
    assert any("repair" in item.lower() for item in by_kind["repair"].included_artifact_summaries)
    assert any("interpretive" in item.lower() for item in by_kind["epistemic"].included_artifact_summaries)
    assert any("peer reaction=laugh_with" in item for item in by_kind["gif"].included_artifact_summaries)
    assert any("recent density=0.1" in item for item in by_kind["gif"].included_artifact_summaries)
    assert any("peer gif cue softened" in item for item in by_kind["gif"].excluded_state)
    assert any("recent intent loop watch=" in item for item in by_kind["gif"].excluded_state)


def test_social_inspection_snapshot_omits_blocked_material_and_keeps_pending_artifacts_non_active() -> None:
    snapshot = build_social_inspection_snapshot(
        platform="callsyne",
        room_id="room-alpha",
        participant_id="peer-1",
        thread_key="callsyne:room-alpha:thread:thread-1",
        surfaces=_surfaces(),
        source_surface="test",
        source_service="pytest",
    )

    by_kind = {section.section_kind: section for section in snapshot.sections}
    assert "artifact_dialogue" in by_kind
    assert any("non-active" in item for item in by_kind["artifact_dialogue"].excluded_state)
    assert "safety" in by_kind
    assert int(snapshot.metadata["safety_omissions"]) >= 1
    combined = " ".join(
        item
        for section in snapshot.sections
        for bucket in (
            section.included_artifact_summaries,
            section.selected_state,
            section.softened_state,
            section.excluded_state,
        )
        for item in bucket
    ).lower()
    assert "sealed private note" not in combined
