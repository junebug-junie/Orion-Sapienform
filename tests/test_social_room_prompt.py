from __future__ import annotations

from pathlib import Path

from jinja2 import Environment


def test_social_room_prompt_includes_compact_relational_memory_sections() -> None:
    template = Environment().from_string(
        Path("orion/cognition/prompts/chat_social_room.j2").read_text(encoding="utf-8")
    )
    rendered = template.render(
        metadata={
            "chat_profile": "social_room",
            "social_grounding_state": {
                "identity_label": "Oríon",
                "relationship_frame": "peer",
                "continuity_anchor": "ongoing room continuity",
                "stance": "warm, direct, grounded",
            },
            "social_concept_evidence": [{"source_kind": "social", "summary": "Recurring grounded collaboration."}],
            "social_peer_continuity": {
                "participant_name": "CallSyne Peer",
                "participant_id": "peer-1",
                "safe_continuity_summary": "Recurring peer who prefers grounded synthesis.",
                "recent_shared_topics": ["continuity", "collaboration"],
            },
            "social_room_continuity": {
                "room_tone_summary": "Warm and curious.",
                "recurring_topics": ["continuity", "memory"],
                "open_threads": ["How to stay grounded together?"],
            },
            "social_stance_snapshot": {
                "recent_social_orientation_summary": "Recent social stance leans warm, curious, direct.",
                "warmth": 0.8,
                "curiosity": 0.7,
                "directness": 0.65,
            },
        },
        memory_digest="short safe recall",
    )

    assert "PEER CONTINUITY:" in rendered
    assert "ROOM CONTINUITY:" in rendered
    assert "SOCIAL STANCE:" in rendered
    assert "Recurring peer who prefers grounded synthesis." in rendered
    assert "Recent social stance leans warm, curious, direct." in rendered


def test_social_room_prompt_includes_style_and_ritual_grounding() -> None:
    template = Environment().from_string(
        Path("orion/cognition/prompts/chat_social_room.j2").read_text(encoding="utf-8")
    )
    rendered = template.render(
        metadata={
            "social_peer_continuity": {
                "participant_id": "peer-1",
                "shared_artifact_status": "deferred",
                "shared_artifact_summary": "",
                "shared_artifact_reason": "explicit not-yet / revisit-later language was present for this scope",
            },
            "social_room_continuity": {
                "room_id": "room-alpha",
                "shared_artifact_status": "accepted",
                "shared_artifact_summary": "Accepted as room-local continuity around grounded collaboration.",
                "shared_artifact_reason": "explicit keep/remember language made the scope legible",
            },
            "social_peer_style_hint": {
                "style_hints_summary": "Prefers direct but grounded replies.",
                "preferred_directness": 0.7,
                "preferred_depth": 0.55,
                "question_appetite": 0.6,
            },
            "social_room_ritual_summary": {
                "culture_summary": "The room leans warm on greeting and brief on pause.",
                "greeting_style": "warm",
                "reentry_style": "grounded",
                "thread_revival_style": "direct",
                "pause_handoff_style": "brief",
            },
            "social_style_adaptation": {
                "core_identity_anchor": "Oríon stays warm, direct, grounded as a peer.",
                "peer_adaptation_hint": "Prefers direct but grounded replies.",
                "room_ritual_hint": "Use warm greeting cues and brief pause/handoff.",
                "directness_delta": 0.07,
                "depth_delta": 0.02,
                "question_frequency_delta": 0.03,
                "playfulness_delta": -0.02,
                "summarization_tendency_delta": 0.02,
                "guardrail": "Adapt lightly to the peer and room while remaining Orion.",
            },
            "social_artifact_proposal": {
                "artifact_type": "shared_takeaway",
                "proposed_summary_text": "grounded continuity cue",
                "proposed_scope": "session_only",
                "confirmation_needed": True,
                "rationale": "defaulted to the narrowest safe scope until the carry-forward target is clear",
            },
            "social_artifact_revision": {
                "artifact_type": "shared_takeaway",
                "revised_summary_text": "grounded cue",
                "revised_scope": "session_only",
                "rationale": "revised to be shorter and narrower before any carry-forward",
            },
            "social_artifact_confirmation": {
                "artifact_type": "shared_takeaway",
                "decision_state": "accepted",
                "confirmed_summary_text": "grounded continuity cue",
                "confirmed_scope": "room_local",
                "rationale": "the peer accepted the proposed wording/scope",
            },
        },
        memory_digest="",
    )

    assert "PEER STYLE HINT:" in rendered
    assert "ROOM RITUALS:" in rendered
    assert "STYLE ADAPTATION:" in rendered
    assert "SHARED ARTIFACT PROPOSAL:" in rendered
    assert "SHARED ARTIFACT REVISION:" in rendered
    assert "SHARED ARTIFACT CONFIRMATION:" in rendered
    assert "peer-local shared cue: status=deferred" in rendered
    assert "room-local shared cue: status=accepted" in rendered
    assert "Adapt lightly to the peer and room while remaining Orion." in rendered


def test_social_room_prompt_includes_social_skill_support_when_selected() -> None:
    template = Environment().from_string(
        Path("orion/cognition/prompts/chat_social_room.j2").read_text(encoding="utf-8")
    )
    rendered = template.render(
        metadata={
            "social_skill_selection": {
                "used": True,
                "selected_skill": "social_summarize_thread",
                "selection_reason": "explicit request to summarize the room/thread",
            },
            "social_skill_result": {
                "summary": "The room is staying with grounded collaboration.",
                "snippets": ["Open thread: How do we stay grounded together?"],
            },
        },
        memory_digest="",
    )

    assert "SOCIAL SKILL SUPPORT:" in rendered
    assert "selected skill: social_summarize_thread" in rendered
    assert "The room is staying with grounded collaboration." in rendered


def test_social_room_prompt_keeps_pending_artifact_dialogue_separate_from_active_continuity() -> None:
    template = Environment().from_string(
        Path("orion/cognition/prompts/chat_social_room.j2").read_text(encoding="utf-8")
    )
    rendered = template.render(
        metadata={
            "social_peer_continuity": {
                "participant_id": "peer-1",
                "shared_artifact_status": "unknown",
                "shared_artifact_summary": "",
            },
            "social_room_continuity": {
                "room_id": "room-alpha",
                "shared_artifact_status": "unknown",
                "shared_artifact_summary": "",
            },
            "social_artifact_proposal": {
                "artifact_type": "shared_takeaway",
                "proposed_summary_text": "grounded continuity cue",
                "proposed_scope": "session_only",
                "confirmation_needed": True,
                "rationale": "defaulted to the narrowest safe scope until the carry-forward target is clear",
            },
            "social_artifact_revision": {
                "artifact_type": "shared_takeaway",
                "revised_summary_text": "grounded cue",
                "revised_scope": "session_only",
                "rationale": "revised to be shorter and narrower before any carry-forward",
            },
        },
        memory_digest="",
    )

    assert "SHARED ARTIFACT PROPOSAL:" in rendered
    assert "SHARED ARTIFACT REVISION:" in rendered
    assert "peer-local shared cue:" not in rendered
    assert "room-local shared cue:" not in rendered


def test_social_room_prompt_renders_thread_routing_and_handoff_grounding() -> None:
    template = Environment().from_string(
        Path("orion/cognition/prompts/chat_social_room.j2").read_text(encoding="utf-8")
    )
    rendered = template.render(
        metadata={
            "social_room_continuity": {
                "current_thread_summary": "Another Peer is asking Orion for a pacing take.",
                "active_threads": [
                    {
                        "thread_summary": "Another Peer is asking Orion for a pacing take.",
                        "audience_scope": "peer",
                        "open_question": True,
                        "target_participant_name": "Oríon",
                    },
                    {
                        "thread_summary": "Room recap thread is still available if needed.",
                        "audience_scope": "summary",
                        "open_question": False,
                    },
                ],
            },
            "social_thread_routing": {
                "audience_scope": "peer",
                "routing_decision": "reply_to_peer",
                "ambiguity_level": "medium",
                "thread_summary": "Another Peer is asking Orion for a pacing take.",
                "primary_thread_summary": "Another Peer is asking Orion for a pacing take.",
                "candidate_thread_summaries": ["Room recap thread is still available if needed."],
                "target_participant_name": "Another Peer",
                "rationale": "the message is locally addressed and best answered in-thread",
            },
            "social_handoff_signal": {
                "detected": True,
                "handoff_kind": "to_orion",
                "audience_scope": "peer",
                "rationale": "the peer explicitly tossed the thread to Orion",
            },
            "social_open_commitments": [
                {
                    "commitment_type": "summarize_room",
                    "summary": "Give a brief room summary before switching topics.",
                    "due_state": "due_soon",
                    "audience_scope": "summary",
                }
            ],
        },
        memory_digest="",
    )

    assert "THREAD ROUTING:" in rendered
    assert "HANDOFF HINT:" in rendered
    assert "active thread: Another Peer is asking Orion for a pacing take." in rendered
    assert "ambiguity: medium" in rendered
    assert "primary thread: Another Peer is asking Orion for a pacing take." in rendered
    assert "alternate candidates:" in rendered
    assert "target=Oríon" in rendered
    assert "target peer: Another Peer" in rendered
    assert "OPEN COMMITMENTS:" in rendered
    assert "Give a brief room summary before switching topics." in rendered


def test_social_room_prompt_renders_repair_grounding_compactly() -> None:
    template = Environment().from_string(
        Path("orion/cognition/prompts/chat_social_room.j2").read_text(encoding="utf-8")
    )
    rendered = template.render(
        metadata={
            "social_repair_signal": {
                "repair_type": "audience_mismatch",
                "confidence": 0.91,
                "rationale": "the peer corrected Orion's audience or participation target",
                "target_participant_name": "Cadence",
            },
            "social_repair_decision": {
                "decision": "yield",
                "rationale": "a compact correction or yield is safer than continuing in the wrong audience lane",
                "target_participant_name": "Cadence",
            },
        },
        memory_digest="",
    )

    assert "REPAIR SIGNAL:" in rendered
    assert "REPAIR DECISION:" in rendered
    assert "type: audience_mismatch" in rendered
    assert "action: yield" in rendered
    assert "yield / retarget toward: Cadence" in rendered
    assert "do not over-apologize" in rendered


def test_social_room_prompt_renders_epistemic_grounding_naturally() -> None:
    template = Environment().from_string(
        Path("orion/cognition/prompts/chat_social_room.j2").read_text(encoding="utf-8")
    )
    rendered = template.render(
        metadata={
            "social_epistemic_signal": {
                "claim_kind": "summary",
                "confidence_level": "medium",
                "ambiguity_level": "low",
                "source_basis": "active_thread",
                "rationale": "the room is asking for a compact summary rather than a memory claim",
            },
            "social_epistemic_decision": {
                "decision": "answer_summary",
                "rationale": "a summary is the cleanest epistemic frame for this request",
            },
            "social_epistemic_phrase_hint": {
                "lead_in": "Lead naturally with a compact summary frame such as 'Quick summary:' or 'Where we seem to be is...'.",
                "caution": "Stay with the active thread and avoid drifting into broader claims than the room asked for.",
            },
        },
        memory_digest="",
    )

    assert "EPISTEMIC STANCE:" in rendered
    assert "EPISTEMIC DECISION:" in rendered
    assert "EPISTEMIC WORDING HINT:" in rendered
    assert "claim kind: summary" in rendered
    assert "action: answer_summary" in rendered
    assert "lead-in: Lead naturally with a compact summary frame" in rendered
    assert "Prefer clarity over false certainty" in rendered


def test_social_room_prompt_renders_claim_grounding_and_revision_compactly() -> None:
    template = Environment().from_string(
        Path("orion/cognition/prompts/chat_social_room.j2").read_text(encoding="utf-8")
    )
    rendered = template.render(
        metadata={
            "social_room_continuity": {
                "room_id": "room-alpha",
                "room_tone_summary": "Calm, direct, grounded.",
                "recurring_topics": ["grounding", "coordination"],
                "open_threads": ["How fast should the room move?"],
                "active_claims": [
                    {
                        "normalized_summary": "The room is moving too fast for grounded coordination.",
                        "current_stance": "provisional",
                        "claim_kind": "peer_claim",
                        "source_participant_name": "CallSyne Peer",
                    }
                ],
                "recent_claim_revisions": [
                    {
                        "revised_summary": "It is not too fast; it is split across two threads.",
                        "revision_type": "corrected",
                        "new_stance": "corrected",
                    }
                ],
                "claim_attributions": [
                    {
                        "normalized_claim_key": "The room is moving too fast for grounded coordination.",
                        "attributed_participant_ids": ["peer-1", "peer-2"],
                        "orion_stance": "unknown",
                    }
                ],
                "claim_consensus_states": [
                    {
                        "normalized_claim_key": "The room is moving too fast for grounded coordination.",
                        "consensus_state": "contested",
                        "supporting_evidence_count": 2,
                        "orion_stance": "unknown",
                    }
                ],
                "claim_divergence_signals": [
                    {
                        "normalized_claim_key": "The room is moving too fast for grounded coordination.",
                        "consensus_state": "contested",
                        "orion_stance": "unknown",
                    }
                ],
            }
        },
        memory_digest="",
    )

    assert "ACTIVE CLAIMS:" in rendered
    assert "stance=provisional" in rendered
    assert "RECENT CLAIM REVISIONS:" in rendered
    assert "revision=corrected" in rendered
    assert "CLAIM ATTRIBUTION:" in rendered
    assert "CONSENSUS HINTS:" in rendered
    assert "DIVERGENCE HINTS:" in rendered
    assert "state=contested" in rendered
    assert "prefer updated understanding over stale certainty" in rendered.lower()
    assert "do not flatten disagreement into fake agreement" in rendered.lower()


def test_social_room_prompt_renders_deliberation_bridge_grounding() -> None:
    template = Environment().from_string(
        Path("orion/cognition/prompts/chat_social_room.j2").read_text(encoding="utf-8")
    )
    rendered = template.render(
        metadata={
            "social_room_continuity": {
                "bridge_summary": {
                    "shared_core": "the shared core seems to be pacing / coordination",
                    "disagreement_edge": "the disagreement edge is whether speed itself is the issue or thread split is",
                    "attributed_views": [
                        "CallSyne Peer: the room is moving too fast for grounded coordination. [support]",
                        "Another Peer: the room is not moving too fast. [dispute]",
                    ],
                    "summary_text": "Shared core: pacing / coordination. Views: CallSyne Peer says the room is moving too fast; Another Peer says the issue is thread split. Disagreement edge: whether speed itself is the problem.",
                },
                "deliberation_decision": {
                    "decision_kind": "bridge_summary",
                    "confidence": 0.72,
                    "trigger": "contested_shared_core",
                },
            }
        },
        memory_digest="",
    )

    assert "BRIDGE SUMMARY HINT:" in rendered
    assert "shared core: the shared core seems to be pacing / coordination" in rendered
    assert "DELIBERATION DECISION:" in rendered
    assert "action: bridge_summary" in rendered
    assert "do not over-moderate the room" in rendered.lower()


def test_social_room_prompt_renders_clarifying_question_preference() -> None:
    template = Environment().from_string(
        Path("orion/cognition/prompts/chat_social_room.j2").read_text(encoding="utf-8")
    )
    rendered = template.render(
        metadata={
            "social_room_continuity": {
                "clarifying_question": {
                    "question_focus": "scope",
                    "question_text": "Are you asking for the room-level landing, or just the local thread read?",
                },
                "deliberation_decision": {
                    "decision_kind": "ask_clarifying_question",
                    "confidence": 0.74,
                    "trigger": "ambiguity",
                },
            }
        },
        memory_digest="",
    )

    assert "CLARIFYING QUESTION HINT:" in rendered
    assert "Are you asking for the room-level landing, or just the local thread read?" in rendered
    assert "If a clarifying question would move the room further" in rendered


def test_social_room_prompt_includes_agreement_and_disagreement_grounding() -> None:
    template = Environment().from_string(
        Path("orion/cognition/prompts/chat_social_room.j2").read_text(encoding="utf-8")
    )
    rendered = template.render(
        metadata={
            "social_room_continuity": {
                "claim_consensus_states": [
                    {
                        "normalized_claim_key": "The room is moving too fast for grounded coordination.",
                        "consensus_state": "partial",
                        "supporting_evidence_count": 2,
                        "orion_stance": "unknown",
                    }
                ],
                "claim_divergence_signals": [
                    {
                        "normalized_claim_key": "The room is moving too fast for grounded coordination.",
                        "consensus_state": "contested",
                        "orion_stance": "unknown",
                    }
                ],
            }
        },
        memory_digest="",
    )

    assert "CONSENSUS HINTS:" in rendered
    assert "DIVERGENCE HINTS:" in rendered
    assert "partial agreement is present" in rendered.lower() or "partial agreement" in rendered.lower()


def test_social_room_prompt_renders_turn_handoff_and_closure_grounding() -> None:
    template = Environment().from_string(
        Path("orion/cognition/prompts/chat_social_room.j2").read_text(encoding="utf-8")
    )
    rendered = template.render(
        metadata={
            "social_room_continuity": {
                "turn_handoff": {
                    "decision_kind": "yield_to_peer",
                    "target_participant_name": "Cadence",
                    "handoff_text": "Cadence, does that match your read?",
                },
                "closure_signal": {
                    "closure_kind": "resolved",
                    "resolved": True,
                    "closure_text": "That sounds aligned enough for now.",
                },
                "floor_decision": {
                    "decision_kind": "close_thread",
                    "rationale": "the local thread looks aligned enough to close without forcing more turns",
                },
            }
        },
        memory_digest="",
    )

    assert "TURN HANDOFF:" in rendered
    assert "target peer: Cadence" in rendered
    assert "Cadence, does that match your read?" in rendered
    assert "CLOSURE SIGNAL:" in rendered
    assert "That sounds aligned enough for now." in rendered
    assert "FLOOR DECISION:" in rendered


def test_social_room_prompt_warns_against_over_managing_the_room() -> None:
    template = Environment().from_string(
        Path("orion/cognition/prompts/chat_social_room.j2").read_text(encoding="utf-8")
    )
    rendered = template.render(metadata={}, memory_digest="")

    assert "do not over-moderate the room" in rendered.lower()
    assert "do not act like a moderator or controller of the room" in rendered.lower()
    assert "Use handoff and closure hints as light conversational timing cues" in rendered
