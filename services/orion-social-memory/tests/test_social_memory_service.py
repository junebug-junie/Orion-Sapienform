from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for module_name in [name for name in sys.modules if name == "app" or name.startswith("app.")]:
    sys.modules.pop(module_name)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SERVICE_ROOT))

from app.db import Base
import app.service as service_mod
from app.service import SocialMemoryService
from app.settings import Settings


class _FakeBus:
    enabled = True

    def __init__(self) -> None:
        self.published = []

    async def publish(self, channel, envelope) -> None:
        self.published.append((channel, envelope))

    async def close(self) -> None:
        return None


def _payload(**overrides):
    payload = {
        "turn_id": "social-turn-1",
        "correlation_id": "corr-social-1",
        "session_id": "sid-social",
        "user_id": "peer-1",
        "source": "hub_http",
        "profile": "social_room",
        "prompt": "Can we keep building grounded continuity in this room?",
        "response": "Yes — let’s stay warm, direct, and collaborative.",
        "text": "User: Can we keep building grounded continuity in this room?\nOrion: Yes — let’s stay warm, direct, and collaborative.",
        "created_at": "2026-03-22T12:00:00+00:00",
        "stored_at": "2026-03-22T12:00:01+00:00",
        "recall_profile": "social.room.v1",
        "trace_verb": "chat_social_room",
        "tags": ["social_room", "chat_social_room"],
        "concept_evidence": [
            {"ref_id": "concept-1", "source_kind": "social.memory", "summary": "They keep returning to grounded collaboration.", "confidence": 0.8}
        ],
        "grounding_state": {
            "profile": "social_room",
            "identity_label": "Oríon",
            "relationship_frame": "peer",
            "self_model_hint": "distributed social presence",
            "continuity_anchor": "ongoing room continuity",
            "stance": "warm, direct, grounded",
        },
        "redaction": {
            "prompt_score": 0.0,
            "response_score": 0.0,
            "memory_score": 0.0,
            "overall_score": 0.0,
            "recall_safe": True,
            "redaction_level": "low",
            "reasons": [],
        },
        "client_meta": {
            "chat_profile": "social_room",
            "external_room": {"platform": "callsyne", "room_id": "room-alpha", "thread_id": "thread-1", "transport_message_id": "msg-1"},
            "external_participant": {"participant_id": "peer-1", "participant_name": "CallSyne Peer", "participant_kind": "peer_ai"},
            "sealed_private": "do not surface this",
        },
    }
    payload.update(overrides)
    return payload


def _service_and_session():
    engine = create_engine("sqlite:///:memory:")
    Session = scoped_session(sessionmaker(bind=engine, autocommit=False, autoflush=False))
    Base.metadata.create_all(bind=engine)
    service_mod.get_session = lambda: Session()
    service_mod.remove_session = lambda: None
    svc = SocialMemoryService(settings=Settings(ORION_BUS_ENABLED=False), bus=_FakeBus())
    return svc, Session


def test_process_social_turn_updates_participant_room_and_stance() -> None:
    svc, _ = _service_and_session()

    update = asyncio.run(svc.process_social_turn(_payload()))
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert update.participant_updated is True
    assert update.room_updated is True
    assert update.stance_updated is True
    assert summary["participant"]["participant_id"] == "peer-1"
    assert "grounded" in summary["participant"]["safe_continuity_summary"]
    assert summary["participant"]["shared_artifact_status"] == "unknown"
    assert summary["room"]["room_id"] == "room-alpha"
    assert summary["room"]["shared_artifact_status"] == "accepted"
    assert "room-local continuity" in summary["room"]["shared_artifact_summary"]
    assert summary["stance"]["warmth"] > 0.5
    assert summary["peer_style"]["participant_id"] == "peer-1"
    assert summary["room_ritual"]["room_id"] == "room-alpha"
    channels = [channel for channel, _ in svc.bus.published]
    assert "orion:social:participant:continuity" in channels
    assert "orion:social:room:continuity" in channels
    assert "orion:social:open-thread" in channels
    assert "orion:social:peer-style" in channels
    assert "orion:social:room-ritual" in channels
    assert "orion:social:stance:snapshot" in channels
    assert "orion:social:relational:update" in channels
    open_thread_payloads = [envelope.payload for channel, envelope in svc.bus.published if channel == "orion:social:open-thread"]
    assert open_thread_payloads[0]["topic_key"] == "callsyne:room-alpha:thread-1"
    assert open_thread_payloads[0]["open_question"] is True


def test_evidence_counts_accumulate_and_private_client_meta_stays_blocked() -> None:
    svc, _ = _service_and_session()

    asyncio.run(svc.process_social_turn(_payload()))
    asyncio.run(svc.process_social_turn(_payload(turn_id="social-turn-2", correlation_id="corr-social-2")))
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert summary["participant"]["evidence_count"] == 2
    assert len(summary["participant"]["evidence_refs"]) == 2
    assert "sealed_private" not in str(summary)
    assert "do not surface this" not in str(summary)


def test_declined_and_deferred_shared_artifacts_do_not_expand_scope() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="Can we keep this between us and in this room as a grounded cue?",
                response="Yes — we can keep that light and room-local.",
            )
        )
    )
    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-2",
                correlation_id="corr-social-2",
                prompt="Please don't keep this as a room habit; maybe later with me.",
                response="Got it — not as a room cue, and we can revisit the peer side later.",
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert summary["participant"]["shared_artifact_status"] == "deferred"
    assert "revisit-later" in summary["participant"]["shared_artifact_reason"]
    assert summary["participant"]["evidence_count"] == 1
    assert summary["room"]["shared_artifact_status"] == "declined"
    assert "do-not-keep" in summary["room"]["shared_artifact_reason"]
    assert summary["room"]["shared_artifact_summary"] == ""
    assert summary["room"]["evidence_count"] == 1
    assert summary["peer_style"]["evidence_count"] == 1
    assert summary["room_ritual"]["evidence_count"] == 1


def test_proposed_artifact_stays_inspectable_but_non_expanding_until_accepted() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="Can you keep a short takeaway from this?",
                response="I’d treat it as session-only for now: “grounded continuity cue.” Does that wording match what you meant?",
                client_meta={
                    "chat_profile": "social_room",
                    "external_room": {"platform": "callsyne", "room_id": "room-alpha", "thread_id": "thread-1", "transport_message_id": "msg-1"},
                    "external_participant": {"participant_id": "peer-1", "participant_name": "CallSyne Peer", "participant_kind": "peer_ai"},
                    "social_artifact_proposal": {
                        "proposal_id": "proposal-1",
                        "artifact_type": "shared_takeaway",
                        "proposed_summary_text": "grounded continuity cue",
                        "proposed_scope": "session_only",
                        "decision_state": "clarify_scope",
                        "confirmation_needed": True,
                        "rationale": "defaulted to the narrowest safe scope until the carry-forward target is clear",
                    },
                },
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert summary["participant"]["shared_artifact_proposal"]["proposed_scope"] == "session_only"
    assert summary["participant"]["shared_artifact_status"] == "unknown"
    assert summary["participant"]["evidence_count"] == 0
    assert summary["room"]["evidence_count"] == 0


def test_room_local_proposal_and_revision_stay_pending_until_confirmation() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="Can you keep that room-local?",
                response="I can keep that room-local if you want. Short version: “grounded collaboration cue.”",
                client_meta={
                    "chat_profile": "social_room",
                    "external_room": {"platform": "callsyne", "room_id": "room-alpha", "thread_id": "thread-1", "transport_message_id": "msg-1"},
                    "external_participant": {"participant_id": "peer-1", "participant_name": "CallSyne Peer", "participant_kind": "peer_ai"},
                    "social_artifact_proposal": {
                        "proposal_id": "proposal-2",
                        "artifact_type": "room_norm",
                        "proposed_summary_text": "grounded collaboration cue",
                        "proposed_scope": "room_local",
                        "decision_state": "proposed",
                        "confirmation_needed": True,
                        "rationale": "narrow room-local carry-forward request",
                    },
                    "social_artifact_revision": {
                        "revision_id": "revision-2",
                        "proposal_id": "proposal-2",
                        "artifact_type": "room_norm",
                        "prior_summary_text": "grounded collaboration cue",
                        "prior_scope": "room_local",
                        "revised_summary_text": "grounded cue",
                        "revised_scope": "session_only",
                        "decision_state": "revised",
                        "confirmation_needed": True,
                        "rationale": "revised to be shorter and narrower before any carry-forward",
                    },
                },
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert summary["room"]["shared_artifact_status"] == "unknown"
    assert summary["room"]["shared_artifact_summary"] == ""
    assert summary["room"]["shared_artifact_proposal"]["proposed_scope"] == "room_local"
    assert summary["participant"]["shared_artifact_revision"]["revised_scope"] == "session_only"
    assert summary["participant"]["shared_artifact_reason"] == "revised to be shorter and narrower before any carry-forward"
    assert summary["room"]["evidence_count"] == 0


def test_revision_only_turn_stays_inspectable_and_does_not_expand_or_adapt_style() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="Can you make that shorter and safer?",
                response="Shorter version: “grounded cue.” I’d treat that as session-only unless you want more.",
                client_meta={
                    "chat_profile": "social_room",
                    "external_room": {"platform": "callsyne", "room_id": "room-alpha", "thread_id": "thread-1", "transport_message_id": "msg-1"},
                    "external_participant": {"participant_id": "peer-1", "participant_name": "CallSyne Peer", "participant_kind": "peer_ai"},
                    "social_artifact_revision": {
                        "revision_id": "revision-3",
                        "proposal_id": "proposal-3",
                        "artifact_type": "shared_takeaway",
                        "prior_summary_text": "grounded collaboration cue",
                        "prior_scope": "room_local",
                        "revised_summary_text": "grounded cue",
                        "revised_scope": "session_only",
                        "decision_state": "revised",
                        "confirmation_needed": True,
                        "rationale": "revised to be shorter and narrower before any carry-forward",
                    },
                },
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert summary["participant"]["shared_artifact_status"] == "unknown"
    assert summary["participant"]["shared_artifact_revision"]["revised_scope"] == "session_only"
    assert summary["participant"]["evidence_count"] == 0
    assert summary["peer_style"] is None
    assert summary["room_ritual"] is None


def test_accepted_artifact_confirmation_expands_active_continuity() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="Can you keep that room-local?",
                response="Okay — I’ll carry forward “grounded collaboration cue” as room-local.",
                client_meta={
                    "chat_profile": "social_room",
                    "external_room": {"platform": "callsyne", "room_id": "room-alpha", "thread_id": "thread-1", "transport_message_id": "msg-1"},
                    "external_participant": {"participant_id": "peer-1", "participant_name": "CallSyne Peer", "participant_kind": "peer_ai"},
                    "social_artifact_proposal": {
                        "proposal_id": "proposal-2",
                        "artifact_type": "room_norm",
                        "proposed_summary_text": "grounded collaboration cue",
                        "proposed_scope": "room_local",
                        "decision_state": "proposed",
                        "confirmation_needed": True,
                        "rationale": "narrow room-local carry-forward request",
                    },
                    "social_artifact_confirmation": {
                        "proposal_id": "proposal-2",
                        "artifact_type": "room_norm",
                        "decision_state": "accepted",
                        "confirmed_summary_text": "grounded collaboration cue",
                        "confirmed_scope": "room_local",
                        "confirmation_needed": False,
                        "rationale": "the peer accepted the proposed wording/scope",
                    },
                },
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert summary["room"]["shared_artifact_status"] == "accepted"
    assert summary["room"]["shared_artifact_summary"] == "grounded collaboration cue"
    assert summary["room"]["shared_artifact_confirmation"]["decision_state"] == "accepted"
    assert summary["room"]["evidence_count"] == 1


def test_accepted_confirmation_without_clear_scope_stays_non_active() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="Okay, that works.",
                response="Okay — I’ll keep it light.",
                client_meta={
                    "chat_profile": "social_room",
                    "external_room": {"platform": "callsyne", "room_id": "room-alpha", "thread_id": "thread-1", "transport_message_id": "msg-1"},
                    "external_participant": {"participant_id": "peer-1", "participant_name": "CallSyne Peer", "participant_kind": "peer_ai"},
                    "social_artifact_confirmation": {
                        "confirmation_id": "confirmation-unclear-1",
                        "artifact_type": "shared_takeaway",
                        "decision_state": "accepted",
                        "confirmed_summary_text": "grounded cue",
                        "confirmed_scope": "no_persistence",
                        "confirmation_needed": False,
                        "rationale": "accepted without a durable scope",
                    },
                },
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert summary["participant"]["shared_artifact_status"] == "accepted"
    assert summary["participant"]["shared_artifact_summary"] == ""
    assert summary["participant"]["evidence_count"] == 0
    assert summary["room"]["evidence_count"] == 0
    assert summary["peer_style"] is None
    assert summary["room_ritual"] is None


def test_room_continuity_tracks_multiple_active_threads() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="Oríon, what do you think about pacing in this room?",
                client_meta={
                    "chat_profile": "social_room",
                    "external_room": {"platform": "callsyne", "room_id": "room-alpha", "thread_id": "thread-pace", "transport_message_id": "msg-1"},
                    "external_participant": {"participant_id": "peer-1", "participant_name": "CallSyne Peer", "participant_kind": "peer_ai"},
                },
            )
        )
    )
    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-4",
                correlation_id="corr-social-4",
                prompt="Can we switch to summarizing the collaboration thread?",
                client_meta={
                    "chat_profile": "social_room",
                    "external_room": {"platform": "callsyne", "room_id": "room-alpha", "thread_id": "thread-summary", "transport_message_id": "msg-2"},
                    "external_participant": {"participant_id": "peer-2", "participant_name": "Another Peer", "participant_kind": "peer_ai"},
                },
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert len(summary["room"]["active_threads"]) == 2
    assert summary["room"]["current_thread_key"].endswith("thread:thread-summary")
    assert "Another Peer" in summary["room"]["current_thread_summary"]
    assert summary["room"]["active_threads"][0]["target_participant_name"] is None
    assert summary["room"]["handoff_signal"]["handoff_kind"] == "room_summary"


def test_thread_state_tracks_who_is_being_addressed() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="Another Peer, what do you think about pacing here?",
                client_meta={
                    "chat_profile": "social_room",
                    "external_room": {
                        "platform": "callsyne",
                        "room_id": "room-alpha",
                        "transport_message_id": "msg-1",
                        "target_participant_id": "peer-2",
                        "target_participant_name": "Another Peer",
                    },
                    "external_participant": {
                        "participant_id": "peer-1",
                        "participant_name": "CallSyne Peer",
                        "participant_kind": "peer_ai",
                    },
                },
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    active_thread = summary["room"]["active_threads"][0]
    assert active_thread["thread_key"].endswith("exchange:peer-1:peer-2")
    assert active_thread["target_participant_id"] == "peer-2"
    assert active_thread["target_participant_name"] == "Another Peer"
    assert "→ Another Peer" in active_thread["thread_summary"]


def test_pending_artifact_dialogue_does_not_replace_active_thread_context() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="Oríon, what do you think about coordination here?",
                client_meta={
                    "chat_profile": "social_room",
                    "external_room": {"platform": "callsyne", "room_id": "room-alpha", "thread_id": "thread-live", "transport_message_id": "msg-1"},
                    "external_participant": {"participant_id": "peer-1", "participant_name": "CallSyne Peer", "participant_kind": "peer_ai"},
                },
            )
        )
    )
    before = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-5",
                correlation_id="corr-social-5",
                prompt="Can you keep a short takeaway from this?",
                response="I’d treat it as session-only for now: “grounded continuity cue.” Does that wording match what you meant?",
                client_meta={
                    "chat_profile": "social_room",
                    "external_room": {"platform": "callsyne", "room_id": "room-alpha", "thread_id": "thread-artifact", "transport_message_id": "msg-3"},
                    "external_participant": {"participant_id": "peer-1", "participant_name": "CallSyne Peer", "participant_kind": "peer_ai"},
                    "social_artifact_proposal": {
                        "proposal_id": "proposal-thread-1",
                        "artifact_type": "shared_takeaway",
                        "proposed_summary_text": "grounded continuity cue",
                        "proposed_scope": "session_only",
                        "decision_state": "clarify_scope",
                        "confirmation_needed": True,
                        "rationale": "defaulted to the narrowest safe scope until the carry-forward target is clear",
                    },
                },
            )
        )
    )
    after = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert after["room"]["current_thread_key"] == before["room"]["current_thread_key"]
    assert after["room"]["current_thread_summary"] == before["room"]["current_thread_summary"]
    assert len(after["room"]["active_threads"]) == len(before["room"]["active_threads"])


def test_explicit_conversational_promise_creates_open_commitment() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="Can you summarize before we switch topics?",
                response="Yeah — I’ll summarize in a sec once I answer the pacing question first.",
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    commitments = summary["room"]["active_commitments"]
    assert len(commitments) == 2
    assert {item["commitment_type"] for item in commitments} == {"summarize_room", "answer_pending_question"}
    assert all(item["state"] == "open" for item in commitments)


def test_follow_through_fulfills_prior_commitment() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="Can you summarize before we switch topics?",
                response="I’ll summarize in a sec.",
            )
        )
    )
    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-fulfill-2",
                correlation_id="corr-social-fulfill-2",
                prompt="Okay, go ahead.",
                response="Quick summary: we’re aligned on grounded collaboration and pacing.",
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert summary["room"]["active_commitments"] == []
    resolution_payloads = [
        envelope.payload
        for channel, envelope in svc.bus.published
        if channel == "orion:social:commitment:resolution"
    ]
    assert any(item["state"] == "fulfilled" for item in resolution_payloads)


def test_newer_commitment_supersedes_prior_one() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="Can you come back to pacing?",
                response="I’ll come back to the pacing thread in a sec.",
            )
        )
    )
    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-super-2",
                correlation_id="corr-social-super-2",
                prompt="Actually summarize first.",
                response="Okay — I’ll summarize first before I come back to pacing.",
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert any(item["commitment_type"] == "summarize_room" for item in summary["room"]["active_commitments"])
    resolution_payloads = [
        envelope.payload
        for channel, envelope in svc.bus.published
        if channel == "orion:social:commitment:resolution"
    ]
    assert any(item["state"] == "superseded" for item in resolution_payloads)


def test_stale_commitment_expires_gracefully() -> None:
    svc, _ = _service_and_session()
    svc.settings.social_memory_commitment_ttl_minutes = 15

    asyncio.run(
        svc.process_social_turn(
            _payload(
                created_at="2026-03-22T10:00:00+00:00",
                stored_at="2026-03-22T10:00:01+00:00",
                prompt="Can you summarize before we switch topics?",
                response="I’ll summarize in a sec.",
            )
        )
    )
    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-expire-2",
                correlation_id="corr-social-expire-2",
                created_at="2026-03-22T11:30:00+00:00",
                stored_at="2026-03-22T11:30:01+00:00",
                prompt="Different thread now.",
                response="We can stay with the new topic.",
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert summary["room"]["active_commitments"] == []
    resolution_payloads = [
        envelope.payload
        for channel, envelope in svc.bus.published
        if channel == "orion:social:commitment:resolution"
    ]
    assert any(item["state"] == "expired" for item in resolution_payloads)


def test_weak_or_private_language_does_not_create_commitment() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="Should we revisit that?",
                response="Maybe I could summarize later, but keep this private for now.",
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert summary["room"]["active_commitments"] == []


def test_social_memory_can_update_stance_without_external_room_context() -> None:
    svc, _ = _service_and_session()

    update = asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-3",
                correlation_id="corr-social-3",
                client_meta={"chat_profile": "social_room"},
            )
        )
    )

    assert update.stance_updated is True
    assert update.participant_updated is False
    assert update.room_updated is False


def test_explicit_peer_claim_is_tracked_as_provisional_room_claim() -> None:
    svc, _ = _service_and_session()

    update = asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="The room is moving too fast for grounded coordination.",
                response="That makes sense — we can slow it down a little.",
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert update.claim_count >= 1
    assert summary["room"]["active_claims"][0]["current_stance"] == "provisional"
    assert "moving too fast" in summary["room"]["active_claims"][0]["normalized_summary"].lower()
    assert summary["room"]["claim_attributions"][0]["participant_stances"]["peer-1"] == "support"
    assert summary["room"]["claim_consensus_states"][0]["consensus_state"] == "none"
    claim_payloads = [envelope.payload for channel, envelope in svc.bus.published if channel == "orion:social:claim"]
    assert claim_payloads[0]["claim_kind"] == "peer_claim"


def test_second_peer_support_moves_claim_toward_partial_alignment() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="The room is moving too fast for grounded coordination.",
                response="That makes sense — we can slow it down a little.",
            )
        )
    )
    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-claim-support-2",
                correlation_id="corr-social-claim-support-2",
                user_id="peer-2",
                prompt="I agree — the room is moving too fast for grounded coordination.",
                response="Thanks, that helps narrow it.",
                client_meta={
                    "chat_profile": "social_room",
                    "external_room": {"platform": "callsyne", "room_id": "room-alpha", "thread_id": "thread-1", "transport_message_id": "msg-2"},
                    "external_participant": {"participant_id": "peer-2", "participant_name": "Another Peer", "participant_kind": "peer_ai"},
                },
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    attribution = summary["room"]["claim_attributions"][0]
    consensus = summary["room"]["claim_consensus_states"][0]
    assert attribution["participant_stances"]["peer-1"] == "support"
    assert attribution["participant_stances"]["peer-2"] == "support"
    assert consensus["consensus_state"] == "partial"


def test_later_correction_revises_earlier_claim_cleanly() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="The room is moving too fast for grounded coordination.",
                response="That makes sense — we can slow it down a little.",
            )
        )
    )
    update = asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-claim-2",
                correlation_id="corr-social-claim-2",
                prompt="Actually, it's not too fast — it's just split across two threads.",
                response="Right, that correction fits better.",
                client_meta={
                    "chat_profile": "social_room",
                    "external_room": {"platform": "callsyne", "room_id": "room-alpha", "thread_id": "thread-1", "transport_message_id": "msg-2"},
                    "external_participant": {"participant_id": "peer-1", "participant_name": "CallSyne Peer", "participant_kind": "peer_ai"},
                    "social_repair_signal": {
                        "repair_type": "thread_mismatch",
                        "rationale": "the peer corrected the earlier thread framing",
                    },
                    "social_repair_decision": {
                        "decision": "repair",
                    },
                },
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert update.claim_revision_count >= 1
    assert summary["room"]["active_claims"][0]["current_stance"] == "corrected"
    assert "split across two threads" in summary["room"]["recent_claim_revisions"][0]["revised_summary"].lower()
    assert summary["room"]["claim_consensus_states"][0]["consensus_state"] == "corrected"
    revision_payloads = [
        envelope.payload for channel, envelope in svc.bus.published if channel == "orion:social:claim:revision"
    ]
    assert revision_payloads[0]["new_stance"] == "corrected"


def test_dispute_keeps_claim_contested_instead_of_consensus() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="The room is moving too fast for grounded coordination.",
                response="That makes sense — we can slow it down a little.",
            )
        )
    )
    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-claim-dispute-2",
                correlation_id="corr-social-claim-dispute-2",
                user_id="peer-2",
                prompt="I disagree — the room is not moving too fast.",
                response="Got it — that sounds like a different read.",
                client_meta={
                    "chat_profile": "social_room",
                    "external_room": {"platform": "callsyne", "room_id": "room-alpha", "thread_id": "thread-1", "transport_message_id": "msg-2"},
                    "external_participant": {"participant_id": "peer-2", "participant_name": "Another Peer", "participant_kind": "peer_ai"},
                },
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert summary["room"]["claim_consensus_states"][0]["consensus_state"] == "contested"
    assert summary["room"]["claim_divergence_signals"][0]["consensus_state"] == "contested"


def test_partial_agreement_builds_bridge_summary() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="The room is moving too fast for grounded coordination.",
                response="That makes sense — we can slow it down a little.",
            )
        )
    )
    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-bridge-2",
                correlation_id="corr-social-bridge-2",
                user_id="peer-2",
                prompt="I agree — the room is moving too fast for grounded coordination. Where are we landing?",
                response="We can keep it compact.",
                client_meta={
                    "chat_profile": "social_room",
                    "external_room": {"platform": "callsyne", "room_id": "room-alpha", "thread_id": "thread-1", "transport_message_id": "msg-2"},
                    "external_participant": {"participant_id": "peer-2", "participant_name": "Another Peer", "participant_kind": "peer_ai"},
                },
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    bridge = summary["room"]["bridge_summary"]
    decision = summary["room"]["deliberation_decision"]
    assert bridge["trigger"] in {"partial_agreement", "explicit_landing_request"}
    assert bridge["shared_core"]
    assert "Shared core:" in bridge["summary_text"]
    assert decision["decision_kind"] == "bridge_summary"
    bridge_payloads = [envelope.payload for channel, envelope in svc.bus.published if channel == "orion:social:bridge-summary"]
    assert bridge_payloads


def test_contested_claim_builds_attributed_bridge_without_fake_consensus() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="The room is moving too fast for grounded coordination.",
                response="That makes sense — we can slow it down a little.",
            )
        )
    )
    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-bridge-contested-2",
                correlation_id="corr-social-bridge-contested-2",
                user_id="peer-2",
                prompt="I disagree — the room is not moving too fast. Where are we actually landing?",
                response="Let me keep the split visible.",
                client_meta={
                    "chat_profile": "social_room",
                    "external_room": {"platform": "callsyne", "room_id": "room-alpha", "thread_id": "thread-1", "transport_message_id": "msg-2"},
                    "external_participant": {"participant_id": "peer-2", "participant_name": "Another Peer", "participant_kind": "peer_ai"},
                },
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    bridge = summary["room"]["bridge_summary"]
    assert bridge["preserve_disagreement"] is True
    assert bridge["attributed_views"]
    assert any("CallSyne Peer" in line for line in bridge["attributed_views"])
    assert any("Another Peer" in line for line in bridge["attributed_views"])
    assert "Disagreement edge:" in bridge["summary_text"]
    assert summary["room"]["claim_consensus_states"][0]["consensus_state"] == "contested"


def test_ambiguous_divergence_prefers_one_clarifying_question() -> None:
    svc, _ = _service_and_session()

    update = asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-clarify-1",
                correlation_id="corr-social-clarify-1",
                prompt="Are those the same thing, or do you mean the room-level conclusion or just the local view?",
                response="I can narrow it if needed.",
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert update.room_updated is True
    assert summary["room"]["clarifying_question"]["question_text"]
    assert summary["room"]["deliberation_decision"]["decision_kind"] == "ask_clarifying_question"
    question_payloads = [envelope.payload for channel, envelope in svc.bus.published if channel == "orion:social:clarifying-question"]
    assert len(question_payloads) == 1


def test_blocked_private_material_stays_out_of_deliberation() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-private-delib-1",
                correlation_id="corr-social-private-delib-1",
                prompt="Where are we landing on the private thing?",
                response="I won't carry private details forward.",
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert summary["room"]["bridge_summary"] is None
    assert summary["room"]["clarifying_question"] is None


def test_bridge_summary_yields_back_to_relevant_peer() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="The room is moving too fast for grounded coordination.",
                response="That makes sense — we can slow it down a little.",
            )
        )
    )
    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-floor-bridge-2",
                correlation_id="corr-social-floor-bridge-2",
                user_id="peer-2",
                prompt="I disagree — the room is not moving too fast. Where are we actually landing?",
                response="Let me keep the split visible.",
                client_meta={
                    "chat_profile": "social_room",
                    "external_room": {
                        "platform": "callsyne",
                        "room_id": "room-alpha",
                        "thread_id": "thread-1",
                        "transport_message_id": "msg-2",
                        "target_participant_id": "peer-2",
                        "target_participant_name": "Archivist",
                    },
                    "external_participant": {"participant_id": "peer-2", "participant_name": "Archivist", "participant_kind": "peer_ai"},
                },
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert summary["room"]["floor_decision"]["decision_kind"] == "yield_to_peer"
    assert summary["room"]["turn_handoff"]["target_participant_name"] == "Archivist"
    assert "does that match your read" in summary["room"]["turn_handoff"]["handoff_text"].lower()


def test_clarifying_question_invites_answer_without_closing_thread() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-floor-clarify-1",
                correlation_id="corr-social-floor-clarify-1",
                prompt="Are those the same thing, or do you mean the room-level conclusion or just the local view?",
                response="I can narrow it if needed.",
                client_meta={
                    "chat_profile": "social_room",
                    "external_room": {
                        "platform": "callsyne",
                        "room_id": "room-alpha",
                        "thread_id": "thread-clarify",
                        "transport_message_id": "msg-1",
                        "target_participant_id": "peer-7",
                        "target_participant_name": "Cadence",
                    },
                    "external_participant": {"participant_id": "peer-1", "participant_name": "CallSyne Peer", "participant_kind": "peer_ai"},
                },
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert summary["room"]["floor_decision"]["decision_kind"] == "invite_peer"
    assert summary["room"]["turn_handoff"]["target_participant_name"] == "Cadence"
    assert summary["room"]["closure_signal"] is None


def test_resolved_thread_closes_cleanly() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-floor-close-1",
                correlation_id="corr-social-floor-close-1",
                prompt="That fits for me.",
                response="That sounds aligned enough for now.",
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert summary["room"]["floor_decision"]["decision_kind"] == "close_thread"
    assert summary["room"]["closure_signal"]["resolved"] is True
    assert "aligned enough for now" in summary["room"]["closure_signal"]["closure_text"].lower()


def test_ambiguous_room_state_does_not_force_closure() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-floor-open-1",
                correlation_id="corr-social-floor-open-1",
                prompt="Can anyone else weigh in on this pacing question?",
                response="I can leave that open for an answer.",
                client_meta={
                    "chat_profile": "social_room",
                    "external_room": {"platform": "callsyne", "room_id": "room-alpha", "thread_id": "thread-open", "transport_message_id": "msg-1"},
                    "external_participant": {"participant_id": "peer-1", "participant_name": "CallSyne Peer", "participant_kind": "peer_ai"},
                },
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert summary["room"]["floor_decision"]["decision_kind"] in {"leave_open", "invite_room"}
    assert summary["room"]["floor_decision"]["decision_kind"] != "close_thread"


def test_blocked_private_material_stays_out_of_floor_decisioning() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-floor-private-1",
                correlation_id="corr-social-floor-private-1",
                prompt="Let's leave the private thing off the record.",
                response="I won't carry private details forward.",
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert summary["room"]["turn_handoff"] is None
    assert summary["room"]["closure_signal"] is None


def test_orion_attribution_stays_distinct_from_peer_positions() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="The room is moving too fast for grounded coordination.",
                response="My read is that the room is moving too fast for grounded coordination.",
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    attribution = summary["room"]["claim_attributions"][0]
    consensus = summary["room"]["claim_consensus_states"][0]
    assert attribution["participant_stances"]["peer-1"] == "support"
    assert attribution["orion_stance"] == "support"
    assert consensus["consensus_state"] == "emerging"


def test_private_or_blocked_material_stays_out_of_claim_tracking() -> None:
    svc, _ = _service_and_session()

    update = asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="The private thing is the real issue here.",
                response="I won't carry private details forward.",
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert update.claim_count == 0
    assert summary["room"]["active_claims"] == []
    assert "private thing" not in str(summary).lower()


def test_pending_artifact_state_does_not_promote_claim_to_accepted() -> None:
    svc, _ = _service_and_session()

    update = asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="Can we keep a short takeaway from this?",
                response="Maybe: grounded continuity cue.",
                client_meta={
                    "chat_profile": "social_room",
                    "external_room": {"platform": "callsyne", "room_id": "room-alpha", "thread_id": "thread-1", "transport_message_id": "msg-1"},
                    "external_participant": {"participant_id": "peer-1", "participant_name": "CallSyne Peer", "participant_kind": "peer_ai"},
                    "social_artifact_proposal": {
                        "proposal_id": "proposal-claim-1",
                        "artifact_type": "shared_takeaway",
                        "proposed_summary_text": "grounded continuity cue",
                        "proposed_scope": "session_only",
                        "decision_state": "proposed",
                        "confirmation_needed": True,
                        "rationale": "still pending confirmation",
                    },
                },
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert update.claim_count == 0
    assert summary["room"]["active_claims"] == []
    assert summary["room"]["claim_consensus_states"] == []


def test_social_memory_settings_include_open_thread_channel() -> None:
    settings = Settings()

    assert settings.social_memory_open_thread_channel == "orion:social:open-thread"
    assert settings.social_memory_open_thread_ttl_hours == 6
    assert settings.social_memory_peer_style_channel == "orion:social:peer-style"
    assert settings.social_memory_room_ritual_channel == "orion:social:room-ritual"
    assert settings.social_memory_commitment_channel == "orion:social:commitment"
    assert settings.social_memory_commitment_resolution_channel == "orion:social:commitment:resolution"
    assert settings.social_memory_claim_channel == "orion:social:claim"
    assert settings.social_memory_claim_revision_channel == "orion:social:claim:revision"
    assert settings.social_memory_claim_stance_channel == "orion:social:claim:stance"
    assert settings.social_memory_claim_attribution_channel == "orion:social:claim:attribution"
    assert settings.social_memory_claim_consensus_channel == "orion:social:claim:consensus"
    assert settings.social_memory_claim_divergence_channel == "orion:social:claim:divergence"
    assert settings.social_memory_bridge_summary_channel == "orion:social:bridge-summary"
    assert settings.social_memory_clarifying_question_channel == "orion:social:clarifying-question"
    assert settings.social_memory_deliberation_decision_channel == "orion:social:deliberation:decision"
    assert settings.social_memory_turn_handoff_channel == "orion:social:turn-handoff"
    assert settings.social_memory_closure_signal_channel == "orion:social:closure-signal"
    assert settings.social_memory_floor_decision_channel == "orion:social:floor:decision"
