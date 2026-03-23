from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from fastapi.testclient import TestClient
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
    assert summary["episode_snapshot"]["summary"]
    assert summary["reentry_anchor"]["reentry_style"] in {"grounded", "warm"}
    assert any(
        item["candidate_kind"] in {"episode_snapshot", "reentry_anchor"} for item in summary["context_candidates"]
    )
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


def test_get_inspection_returns_operator_snapshot() -> None:
    svc, _ = _service_and_session()

    asyncio.run(svc.process_social_turn(_payload()))
    inspection = asyncio.run(svc.get_inspection(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    section_kinds = {section["section_kind"] for section in inspection["sections"]}
    assert inspection["platform"] == "callsyne"
    assert "context_window" in section_kinds
    assert "resumptive" in section_kinds
    assert inspection["metadata"]["tool_execution_available"] == "false"


def test_inspection_endpoint_serves_snapshot() -> None:
    import app.main as main_mod

    class _FakeInspectionService:
        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        async def get_inspection(self, *, platform: str, room_id: str, participant_id: str | None):
            return {"platform": platform, "room_id": room_id, "participant_id": participant_id, "sections": [], "summary": "ok", "metadata": {}}

    original_service = main_mod.service
    main_mod.service = _FakeInspectionService()
    try:
        with TestClient(main_mod.app) as client:
            resp = client.get("/inspection", params={"platform": "callsyne", "room_id": "room-alpha", "participant_id": "peer-1"})
        assert resp.status_code == 200
        assert resp.json()["room_id"] == "room-alpha"
    finally:
        main_mod.service = original_service


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


def test_partial_agreement_can_stay_plain_when_bridge_summary_is_not_needed() -> None:
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

    decision = summary["room"]["deliberation_decision"]
    assert summary["room"]["bridge_summary"] is None
    assert decision["decision_kind"] in {"normal_peer_reply", "normal_room_reply"}


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
    assert "Open edge:" in bridge["summary_text"]
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
    assert "how does that land for you" in summary["room"]["turn_handoff"]["handoff_text"].lower()


def test_bridge_summary_is_suppressed_when_plain_peer_reply_is_better() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="I think the pacing is the problem here.",
                response="I can stay with that read.",
            )
        )
    )
    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-plain-peer-reply-2",
                correlation_id="corr-social-plain-peer-reply-2",
                prompt="I disagree — it feels more like thread split than pacing.",
                response="That helps; I can answer that directly without trying to bridge the whole room.",
            )
        )
    )

    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert summary["room"]["deliberation_decision"]["decision_kind"] == "normal_peer_reply"
    assert summary["room"]["bridge_summary"] is None


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
    assert "what part feels most live" in summary["room"]["turn_handoff"]["handoff_text"].lower()


def test_low_ambiguity_peer_thread_does_not_force_clarifying_question_from_boundary() -> None:
    svc, Session = _service_and_session()
    sess = Session()
    sess.add(
        service_mod.SocialRoomContinuitySQL(
            room_key="callsyne:room-alpha",
            platform="callsyne",
            room_id="room-alpha",
            recurring_topics=["pacing"],
            active_participants=["CallSyne Peer"],
            recent_thread_summary="Pacing thread.",
            room_tone_summary="Warm and curious.",
            open_threads=["thread:thread-1"],
            evidence_refs=["social-turn-old"],
            evidence_count=2,
            last_updated_at="2026-03-22T12:00:00+00:00",
            current_thread_key="callsyne:room-alpha:thread:thread-1",
            current_thread_summary="Pacing thread.",
            trust_boundaries=[
                {
                    "boundary_id": "boundary-1",
                    "platform": "callsyne",
                    "room_id": "room-alpha",
                    "thread_key": "callsyne:room-alpha:thread:thread-1",
                    "scope": "room_thread",
                    "calibration_kind": "disagreement_prone",
                    "treat_claims_as_provisional": True,
                    "summary_anchor": False,
                    "use_narrower_attribution": True,
                    "require_clarification_before_shared_ground": True,
                    "rationale": "Stay narrow when disagreement is active.",
                    "reasons": ["recent_disagreement"],
                    "updated_at": "2026-03-22T12:00:00+00:00",
                    "metadata": {"source": "social-memory"},
                }
            ],
        )
    )
    sess.commit()
    sess.close()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-low-ambiguity-1",
                correlation_id="corr-social-low-ambiguity-1",
                prompt="I think it's just pacing, not thread split.",
                response="Got it — I’ll stay with the pacing read.",
            )
        )
    )

    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert summary["room"]["clarifying_question"] is None
    assert summary["room"]["deliberation_decision"]["decision_kind"] in {"normal_peer_reply", "stay_narrow"}


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

    assert summary["room"]["floor_decision"]["decision_kind"] == "leave_open"
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


def test_repeated_corrected_claims_create_cautious_peer_calibration() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="Fast pacing is clearly the room norm.",
                response="I read it that way too.",
            )
        )
    )
    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-2",
                correlation_id="corr-social-2",
                prompt="Actually, I was wrong — fast pacing is not settled here.",
                response="Right, let's keep that provisional.",
            )
        )
    )
    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-3",
                correlation_id="corr-social-3",
                prompt="To correct that again, the room keeps revising the pacing read.",
                response="Yes, attribute that carefully instead of treating it as shared ground.",
            )
        )
    )

    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))
    participant = summary["participant"]
    room = summary["room"]

    assert participant["peer_calibration"]["calibration_kind"] == "revised_often"
    assert participant["trust_boundary"]["treat_claims_as_provisional"] is True
    assert participant["trust_boundary"]["use_narrower_attribution"] is True
    assert participant["trust_boundary"]["require_clarification_before_shared_ground"] is True
    assert participant["trust_boundary"]["metadata"]["authority_shortcut"] == "disabled"
    assert any(boundary["calibration_kind"] == "revised_often" for boundary in room["trust_boundaries"])


def test_repeated_aligned_summaries_create_continuity_anchor_without_authority_shortcut() -> None:
    svc, _ = _service_and_session()

    asyncio.run(svc.process_social_turn(_payload(prompt="We keep coming back to grounded collaboration.", response="Yes, that still fits.")))
    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-2",
                correlation_id="corr-social-2",
                prompt="Grounded collaboration still feels like the stable thread here.",
                response="Agreed — that's still the core thread.",
            )
        )
    )
    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-3",
                correlation_id="corr-social-3",
                prompt="Quick summary: grounded collaboration is still the room's shared core.",
                response="Yes, we're aligned on that summary.",
            )
        )
    )

    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))
    participant = summary["participant"]
    room = summary["room"]

    assert participant["peer_calibration"]["calibration_kind"] == "strong_summary_partner"
    assert participant["trust_boundary"]["summary_anchor"] is True
    assert participant["trust_boundary"]["treat_claims_as_provisional"] is False
    assert participant["trust_boundary"]["metadata"]["authority_shortcut"] == "disabled"
    assert room["peer_calibrations"][0]["calibration_kind"] == "strong_summary_partner"


def test_calibration_stays_local_to_peer_and_thread_scope() -> None:
    svc, _ = _service_and_session()

    asyncio.run(svc.process_social_turn(_payload(prompt="The pacing read is settled.", response="Maybe, but let's see.")))
    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-2",
                correlation_id="corr-social-2",
                prompt="Actually, I need to correct that pacing read.",
                response="Right, that earlier framing was off.",
            )
        )
    )
    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-3",
                correlation_id="corr-social-3",
                prompt="To correct it again, this thread is still revising the pacing claim.",
                response="Yes, keep this thread-specific and provisional.",
            )
        )
    )
    peer_one_summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-4",
                correlation_id="corr-social-4",
                prompt="I just want to talk about greeting style in a different thread.",
                response="Warm and brief works for me.",
                client_meta={
                    "chat_profile": "social_room",
                    "external_room": {"platform": "callsyne", "room_id": "room-alpha", "thread_id": "thread-2", "transport_message_id": "msg-4"},
                    "external_participant": {"participant_id": "peer-2", "participant_name": "Another Peer", "participant_kind": "peer_ai"},
                },
            )
        )
    )
    peer_two_summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-2"))

    assert peer_one_summary["participant"]["peer_calibration"]["thread_key"] == "callsyne:room-alpha:thread:thread-1"
    assert all(boundary["thread_key"] == "callsyne:room-alpha:thread:thread-1" for boundary in peer_one_summary["room"]["trust_boundaries"] if boundary["participant_id"] == "peer-1")
    assert peer_two_summary["participant"]["peer_calibration"] is None or peer_two_summary["participant"]["peer_calibration"]["participant_id"] == "peer-2"
    assert peer_two_summary["participant"]["trust_boundary"] is None or peer_two_summary["participant"]["trust_boundary"]["participant_id"] == "peer-2"
    assert peer_two_summary["participant"]["trust_boundary"] is None or peer_two_summary["participant"]["trust_boundary"]["calibration_kind"] != "revised_often"


def test_pending_artifact_dialogue_is_not_used_as_calibration_evidence_and_private_stays_blocked() -> None:
    svc, _ = _service_and_session()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="Can you keep a short takeaway from this?",
                response="I’d keep it session-only for now.",
                client_meta={
                    "chat_profile": "social_room",
                    "external_room": {"platform": "callsyne", "room_id": "room-alpha", "thread_id": "thread-1", "transport_message_id": "msg-1"},
                    "external_participant": {"participant_id": "peer-1", "participant_name": "CallSyne Peer", "participant_kind": "peer_ai"},
                    "social_artifact_proposal": {
                        "proposal_id": "proposal-calibration-1",
                        "artifact_type": "shared_takeaway",
                        "proposed_summary_text": "grounded cue",
                        "proposed_scope": "session_only",
                        "decision_state": "clarify_scope",
                        "confirmation_needed": True,
                        "rationale": "defaulted to the narrowest safe scope until the carry-forward target is clear",
                    },
                },
            )
        )
    )
    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-2",
                correlation_id="corr-social-2",
                prompt="Actually keep this private and sealed; I was wrong earlier.",
                response="Agreed — we should not carry that forward.",
            )
        )
    )
    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert summary["participant"]["peer_calibration"] is None
    assert summary["participant"]["calibration_signals"] == []
    assert summary["room"]["trust_boundaries"] == []
    assert "sealed_private" not in str(summary)


def test_stale_consensus_softens_and_flags_refresh_needed() -> None:
    svc, Session = _service_and_session()
    sess = Session()
    sess.add(
        service_mod.SocialRoomContinuitySQL(
            room_key="callsyne:room-alpha",
            platform="callsyne",
            room_id="room-alpha",
            recurring_topics=["grounded collaboration"],
            active_participants=["CallSyne Peer"],
            recent_thread_summary="Grounded collaboration used to be the settled thread.",
            room_tone_summary="Warm and curious.",
            open_threads=["thread:thread-1"],
            evidence_refs=["social-turn-old"],
            evidence_count=3,
            last_updated_at="2026-03-10T12:00:00+00:00",
            shared_artifact_scope="room_local",
            shared_artifact_status="accepted",
            shared_artifact_summary="Accepted as room-local continuity around grounded collaboration.",
            shared_artifact_reason="explicit keep/remember language made the scope legible",
            active_claims=[],
            recent_claim_revisions=[],
            claim_attributions=[],
            claim_consensus_states=[
                {
                    "consensus_id": "consensus-old-1",
                    "claim_id": "claim-old-1",
                    "platform": "callsyne",
                    "room_id": "room-alpha",
                    "thread_key": "callsyne:room-alpha:thread:thread-1",
                    "normalized_claim_key": "grounded collaboration is the room's shared core",
                    "consensus_state": "consensus",
                    "supporting_participant_ids": ["peer-1", "peer-2"],
                    "disputing_participant_ids": [],
                    "questioning_participant_ids": [],
                    "orion_stance": "support",
                    "confidence": 0.86,
                    "supporting_evidence_count": 3,
                    "updated_at": "2026-03-10T12:00:00+00:00",
                    "reasons": ["support_count=3"],
                    "metadata": {"source": "social-memory"},
                }
            ],
            claim_divergence_signals=[],
            active_commitments=[],
            peer_calibrations=[],
            trust_boundaries=[],
        )
    )
    sess.commit()
    sess.close()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-stale-consensus",
                correlation_id="corr-stale-consensus",
                prompt="Checking in on logistics only.",
                response="Acknowledged.",
                created_at="2026-03-22T12:00:00+00:00",
                stored_at="2026-03-22T12:00:01+00:00",
            )
        )
    )

    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))
    consensus = summary["room"]["claim_consensus_states"][0]

    assert consensus["consensus_state"] != "consensus"
    assert consensus["metadata"]["regrounding_decision"] in {"soften", "refresh_needed"}
    assert any(item["artifact_kind"] == "claim_consensus" for item in summary["room"]["memory_freshness"])



def test_stale_calibration_decays_toward_unknown_or_cautious() -> None:
    svc, Session = _service_and_session()
    sess = Session()
    sess.add(
        service_mod.SocialParticipantContinuitySQL(
            peer_key="callsyne:room-alpha:peer-1",
            platform="callsyne",
            room_id="room-alpha",
            participant_id="peer-1",
            participant_name="CallSyne Peer",
            aliases=["CallSyne Peer"],
            participant_kind="peer_ai",
            recent_shared_topics=["pacing"],
            interaction_tone_summary="warm, direct, grounded",
            safe_continuity_summary="Recurring peer.",
            evidence_refs=["social-turn-old"],
            evidence_count=3,
            last_seen_at="2026-03-10T12:00:00+00:00",
            confidence=0.6,
            trust_tier="known",
            shared_artifact_scope="peer_local",
            shared_artifact_status="accepted",
            shared_artifact_summary="Accepted as peer-local continuity around pacing.",
            shared_artifact_reason="explicit keep/remember language made the scope legible",
            calibration_signals=[],
            peer_calibration={
                "calibration_id": "cal-old-1",
                "platform": "callsyne",
                "room_id": "room-alpha",
                "participant_id": "peer-1",
                "participant_name": "CallSyne Peer",
                "thread_key": "callsyne:room-alpha:thread:thread-1",
                "scope": "peer_thread",
                "calibration_kind": "strong_summary_partner",
                "confidence": 0.82,
                "evidence_count": 3,
                "reversible": True,
                "decay_hint": "decay_after_topic_shift",
                "rationale": "Older summary calibration.",
                "reasons": ["aligned_summary_language"],
                "active_signal_ids": ["signal-1"],
                "caution_bias": 0.08,
                "attribution_bias": 0.12,
                "clarification_bias": 0.1,
                "updated_at": "2026-03-10T12:00:00+00:00",
                "metadata": {"source": "social-memory", "authority_shortcut": "disabled"},
            },
            trust_boundary={
                "boundary_id": "boundary-old-1",
                "platform": "callsyne",
                "room_id": "room-alpha",
                "participant_id": "peer-1",
                "participant_name": "CallSyne Peer",
                "thread_key": "callsyne:room-alpha:thread:thread-1",
                "scope": "peer_thread",
                "calibration_kind": "strong_summary_partner",
                "confidence": 0.76,
                "evidence_count": 3,
                "reversible": True,
                "decay_hint": "decay_after_topic_shift",
                "treat_claims_as_provisional": False,
                "summary_anchor": True,
                "use_narrower_attribution": False,
                "require_clarification_before_shared_ground": False,
                "caution_bias": 0.08,
                "attribution_bias": 0.12,
                "clarification_bias": 0.1,
                "rationale": "Older trust boundary.",
                "reasons": ["aligned_summary_language"],
                "updated_at": "2026-03-10T12:00:00+00:00",
                "metadata": {"source": "social-memory", "authority_shortcut": "disabled"},
            },
        )
    )
    sess.add(
        service_mod.SocialRoomContinuitySQL(
            room_key="callsyne:room-alpha",
            platform="callsyne",
            room_id="room-alpha",
            recurring_topics=["pacing"],
            active_participants=["CallSyne Peer"],
            recent_thread_summary="Pacing thread.",
            room_tone_summary="Warm and curious.",
            open_threads=["thread:thread-2"],
            evidence_refs=["social-turn-old"],
            evidence_count=2,
            last_updated_at="2026-03-10T12:00:00+00:00",
            shared_artifact_scope="room_local",
            shared_artifact_status="unknown",
            shared_artifact_summary="",
            shared_artifact_reason="",
            active_claims=[],
            recent_claim_revisions=[],
            claim_attributions=[],
            claim_consensus_states=[],
            claim_divergence_signals=[],
            active_commitments=[],
            peer_calibrations=[],
            trust_boundaries=[],
        )
    )
    sess.commit()
    sess.close()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-stale-calibration",
                correlation_id="corr-stale-calibration",
                prompt="We can focus on logistics here.",
                response="Acknowledged.",
                created_at="2026-03-22T12:00:00+00:00",
                stored_at="2026-03-22T12:00:01+00:00",
                client_meta={
                    "chat_profile": "social_room",
                    "external_room": {"platform": "callsyne", "room_id": "room-alpha", "thread_id": "thread-2", "transport_message_id": "msg-2"},
                    "external_participant": {"participant_id": "peer-1", "participant_name": "CallSyne Peer", "participant_kind": "peer_ai"},
                },
            )
        )
    )

    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))
    calibration = summary["participant"]["peer_calibration"]
    boundary = summary["participant"]["trust_boundary"]

    assert calibration is not None
    assert calibration["calibration_kind"] in {"unknown", "cautious_scope"}
    assert boundary is not None
    assert boundary["summary_anchor"] is False
    assert any(item["artifact_kind"] == "peer_calibration" for item in summary["participant"]["memory_freshness"])



def test_stale_room_ritual_hint_fades_when_not_repeated() -> None:
    svc, Session = _service_and_session()
    sess = Session()
    sess.add(
        service_mod.SocialRoomRitualSummarySQL(
            ritual_key="callsyne:room-alpha",
            platform="callsyne",
            room_id="room-alpha",
            greeting_style="warm",
            reentry_style="grounded",
            thread_revival_style="direct",
            pause_handoff_style="brief",
            summary_cadence_preference=0.6,
            room_tone_summary="Warm and curious.",
            culture_summary="The room tends toward warm greeting and brief pause cues.",
            evidence_count=3,
            confidence=0.72,
            last_updated_at="2026-03-10T12:00:00+00:00",
        )
    )
    sess.commit()
    sess.close()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-stale-ritual",
                correlation_id="corr-stale-ritual",
                prompt="This is purely logistical.",
                response="Acknowledged.",
                created_at="2026-03-22T12:00:00+00:00",
                stored_at="2026-03-22T12:00:01+00:00",
            )
        )
    )

    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))
    ritual = summary["room_ritual"]

    assert ritual["confidence"] < 0.72
    assert "fading" in ritual["culture_summary"]
    assert any(item["artifact_kind"] == "room_ritual" for item in summary["room"]["memory_freshness"])



def test_stale_commitment_expires_cleanly_and_is_recorded_for_regrounding() -> None:
    svc, Session = _service_and_session()
    sess = Session()
    sess.add(
        service_mod.SocialRoomContinuitySQL(
            room_key="callsyne:room-alpha",
            platform="callsyne",
            room_id="room-alpha",
            recurring_topics=["summary"],
            active_participants=["CallSyne Peer"],
            recent_thread_summary="Summary thread.",
            room_tone_summary="Warm and curious.",
            open_threads=["thread:thread-1"],
            evidence_refs=["social-turn-old"],
            evidence_count=2,
            last_updated_at="2026-03-21T12:00:00+00:00",
            shared_artifact_scope="room_local",
            shared_artifact_status="unknown",
            shared_artifact_summary="",
            shared_artifact_reason="",
            active_claims=[],
            recent_claim_revisions=[],
            claim_attributions=[],
            claim_consensus_states=[],
            claim_divergence_signals=[],
            active_commitments=[
                {
                    "commitment_id": "commitment-old-1",
                    "platform": "callsyne",
                    "room_id": "room-alpha",
                    "thread_key": "callsyne:room-alpha:thread:thread-1",
                    "commitment_type": "summarize_room",
                    "audience_scope": "summary",
                    "summary": "Provide the room summary.",
                    "state": "open",
                    "source_turn_id": "social-turn-old",
                    "source_correlation_id": "corr-old",
                    "created_at": "2026-03-10T12:00:00+00:00",
                    "expires_at": "2026-03-10T12:05:00+00:00",
                    "due_state": "stale",
                    "resolution_reason": "",
                    "metadata": {"source": "social-memory"},
                }
            ],
            peer_calibrations=[],
            trust_boundaries=[],
        )
    )
    sess.commit()
    sess.close()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-expire-commitment",
                correlation_id="corr-expire-commitment",
                prompt="New logistics only.",
                response="Acknowledged.",
                created_at="2026-03-22T12:00:00+00:00",
                stored_at="2026-03-22T12:00:01+00:00",
            )
        )
    )

    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))

    assert summary["room"]["active_commitments"] == []
    assert any(item["artifact_kind"] == "commitment" and item["regrounding_decision"] == "expire" for item in summary["room"]["memory_freshness"])



def test_refreshed_evidence_restrengthens_softened_consensus() -> None:
    svc, Session = _service_and_session()
    sess = Session()
    sess.add(
        service_mod.SocialRoomContinuitySQL(
            room_key="callsyne:room-alpha",
            platform="callsyne",
            room_id="room-alpha",
            recurring_topics=["grounded collaboration"],
            active_participants=["CallSyne Peer"],
            recent_thread_summary="Grounded collaboration thread.",
            room_tone_summary="Warm and curious.",
            open_threads=["thread:thread-1"],
            evidence_refs=["social-turn-old"],
            evidence_count=3,
            last_updated_at="2026-03-10T12:00:00+00:00",
            shared_artifact_scope="room_local",
            shared_artifact_status="accepted",
            shared_artifact_summary="Accepted as room-local continuity around grounded collaboration.",
            shared_artifact_reason="explicit keep/remember language made the scope legible",
            active_claims=[
                {
                    "stance_id": "stance-old-1",
                    "claim_id": "claim-old-1",
                    "platform": "callsyne",
                    "room_id": "room-alpha",
                    "thread_key": "callsyne:room-alpha:thread:thread-1",
                    "source_participant_id": "peer-1",
                    "source_participant_name": "CallSyne Peer",
                    "claim_kind": "shared_summary",
                    "normalized_summary": "grounded collaboration is still the room's shared core",
                    "current_stance": "accepted",
                    "confidence": 0.4,
                    "source_basis": "social_memory",
                    "related_claim_ids": [],
                    "reasons": ["older_softened_state"],
                    "created_at": "2026-03-10T12:00:00+00:00",
                    "updated_at": "2026-03-10T12:00:00+00:00",
                    "metadata": {"source": "social-memory"},
                }
            ],
            recent_claim_revisions=[],
            claim_attributions=[
                {
                    "attribution_id": "attr-old-1",
                    "claim_id": "claim-old-1",
                    "platform": "callsyne",
                    "room_id": "room-alpha",
                    "thread_key": "callsyne:room-alpha:thread:thread-1",
                    "normalized_claim_key": "grounded collaboration is still the room's shared core",
                    "attributed_participant_ids": ["peer-1"],
                    "attributed_participant_names": {"peer-1": "CallSyne Peer"},
                    "participant_stances": {"peer-1": "support"},
                    "orion_stance": "unknown",
                    "confidence": 0.35,
                    "supporting_evidence_count": 1,
                    "updated_at": "2026-03-10T12:00:00+00:00",
                    "reasons": ["older_softened_state"],
                    "metadata": {"source": "social-memory"},
                }
            ],
            claim_consensus_states=[
                {
                    "consensus_id": "consensus-soft-1",
                    "claim_id": "claim-old-1",
                    "platform": "callsyne",
                    "room_id": "room-alpha",
                    "thread_key": "callsyne:room-alpha:thread:thread-1",
                    "normalized_claim_key": "grounded collaboration is still the room's shared core",
                    "consensus_state": "emerging",
                    "supporting_participant_ids": ["peer-1"],
                    "disputing_participant_ids": [],
                    "questioning_participant_ids": [],
                    "orion_stance": "unknown",
                    "confidence": 0.3,
                    "supporting_evidence_count": 1,
                    "updated_at": "2026-03-10T12:00:00+00:00",
                    "reasons": ["older_softened_state"],
                    "metadata": {"source": "social-memory", "freshness_state": "stale", "regrounding_decision": "soften"},
                }
            ],
            claim_divergence_signals=[],
            active_commitments=[],
            peer_calibrations=[],
            trust_boundaries=[],
        )
    )
    sess.commit()
    sess.close()

    asyncio.run(
        svc.process_social_turn(
            _payload(
                turn_id="social-turn-refresh-consensus",
                correlation_id="corr-refresh-consensus",
                prompt="Quick summary: grounded collaboration is still the room's shared core.",
                response="Yes, we're aligned on that summary.",
                created_at="2026-03-22T12:00:00+00:00",
                stored_at="2026-03-22T12:00:01+00:00",
            )
        )
    )

    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))
    consensus = summary["room"]["claim_consensus_states"][0]

    assert consensus["confidence"] >= 0.3
    assert consensus["supporting_evidence_count"] >= 1
    assert consensus["metadata"].get("regrounding_decision") != "soften"


def test_addressed_peer_context_is_preferred_over_generic_room_context() -> None:
    svc, _ = _service_and_session()
    asyncio.run(svc.process_social_turn(_payload()))

    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))
    candidates = summary["context_candidates"]
    decision = summary["context_selection_decision"]

    peer_candidate = next(item for item in candidates if item["candidate_kind"] == "peer_continuity")
    thread_candidate = next(item for item in candidates if item["candidate_kind"] == "thread")

    assert peer_candidate["inclusion_decision"] == "include"
    assert thread_candidate["inclusion_decision"] == "include"
    assert peer_candidate["relevance_score"] > 0.9
    assert "addressed_peer_context_preferred" in decision["reasons"]



def test_stale_consensus_candidate_is_excluded_in_favor_of_fresher_divergence() -> None:
    svc, Session = _service_and_session()
    sess = Session()
    sess.add(
        service_mod.SocialRoomContinuitySQL(
            room_key="callsyne:room-alpha",
            platform="callsyne",
            room_id="room-alpha",
            recurring_topics=["pacing"],
            active_participants=["CallSyne Peer"],
            recent_thread_summary="Pacing thread.",
            room_tone_summary="Warm and curious.",
            open_threads=["thread:thread-1"],
            evidence_refs=["social-turn-old"],
            evidence_count=3,
            last_updated_at="2026-03-22T12:00:00+00:00",
            shared_artifact_scope="room_local",
            shared_artifact_status="accepted",
            shared_artifact_summary="Accepted as room-local continuity around pacing.",
            shared_artifact_reason="explicit keep/remember language made the scope legible",
            claim_consensus_states=[
                {
                    "consensus_id": "consensus-old-1",
                    "claim_id": "claim-old-1",
                    "platform": "callsyne",
                    "room_id": "room-alpha",
                    "thread_key": "callsyne:room-alpha:thread:thread-1",
                    "normalized_claim_key": "the pacing read is settled",
                    "consensus_state": "consensus",
                    "supporting_participant_ids": ["peer-1", "peer-2"],
                    "disputing_participant_ids": [],
                    "questioning_participant_ids": [],
                    "orion_stance": "support",
                    "confidence": 0.8,
                    "supporting_evidence_count": 3,
                    "updated_at": "2026-03-10T12:00:00+00:00",
                    "reasons": ["support_count=3"],
                    "metadata": {"source": "social-memory", "regrounding_decision": "soften"},
                }
            ],
            claim_divergence_signals=[
                {
                    "divergence_id": "diverge-1",
                    "claim_id": "claim-new-1",
                    "platform": "callsyne",
                    "room_id": "room-alpha",
                    "thread_key": "callsyne:room-alpha:thread:thread-1",
                    "normalized_claim_key": "the pacing read is still contested",
                    "divergence_detected": True,
                    "consensus_state": "contested",
                    "participant_stances": {"peer-1": "support", "peer-2": "dispute"},
                    "orion_stance": "unknown",
                    "confidence": 0.72,
                    "supporting_evidence_count": 2,
                    "updated_at": "2026-03-22T12:00:00+00:00",
                    "reasons": ["divergence_requires_attribution"],
                    "metadata": {"source": "social-memory"},
                }
            ],
            memory_freshness=[
                {
                    "freshness_id": "fresh-1",
                    "platform": "callsyne",
                    "room_id": "room-alpha",
                    "artifact_kind": "claim_consensus",
                    "freshness_state": "stale",
                    "decay_level": "moderate",
                    "regrounding_decision": "soften",
                    "confidence": 0.5,
                    "evidence_count": 2,
                    "last_updated_at": "2026-03-10T12:00:00+00:00",
                    "rationale": "Older consensus is stale.",
                    "reasons": ["support_is_stale"],
                    "metadata": {"source": "social-memory"},
                }
            ],
            active_claims=[],
            recent_claim_revisions=[],
            claim_attributions=[],
            active_commitments=[],
            peer_calibrations=[],
            trust_boundaries=[],
        )
    )
    sess.commit()
    sess.close()

    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))
    candidates = summary["context_candidates"]

    divergence = next(item for item in candidates if item["candidate_kind"] == "divergence")
    consensus = next(item for item in candidates if item["candidate_kind"] == "consensus")
    episode = next(item for item in candidates if item["candidate_kind"] == "episode_snapshot")
    reentry = next(item for item in candidates if item["candidate_kind"] == "reentry_anchor")

    assert divergence["inclusion_decision"] == "include"
    assert consensus["inclusion_decision"] == "exclude"
    assert episode["inclusion_decision"] in {"soften", "exclude"}
    assert reentry["inclusion_decision"] in {"soften", "exclude"}



def test_active_commitment_is_selected_over_old_ritual_hint() -> None:
    svc, Session = _service_and_session()
    sess = Session()
    sess.add(
        service_mod.SocialRoomContinuitySQL(
            room_key="callsyne:room-alpha",
            platform="callsyne",
            room_id="room-alpha",
            recurring_topics=["summary"],
            active_participants=["CallSyne Peer"],
            recent_thread_summary="Summary thread.",
            room_tone_summary="Warm and curious.",
            open_threads=["thread:thread-1"],
            evidence_refs=["social-turn-old"],
            evidence_count=2,
            last_updated_at="2026-03-22T12:00:00+00:00",
            shared_artifact_scope="room_local",
            shared_artifact_status="unknown",
            shared_artifact_summary="",
            shared_artifact_reason="",
            active_claims=[],
            recent_claim_revisions=[],
            claim_attributions=[],
            claim_consensus_states=[],
            claim_divergence_signals=[],
            active_commitments=[
                {
                    "commitment_id": "commitment-1",
                    "platform": "callsyne",
                    "room_id": "room-alpha",
                    "thread_key": "callsyne:room-alpha:thread:thread-1",
                    "commitment_type": "summarize_room",
                    "audience_scope": "summary",
                    "summary": "Provide the room summary next.",
                    "state": "open",
                    "created_at": "2026-03-22T11:55:00+00:00",
                    "expires_at": "2026-03-22T12:30:00+00:00",
                    "due_state": "due_soon",
                    "resolution_reason": "",
                    "metadata": {"source": "social-memory"},
                }
            ],
            memory_freshness=[],
            peer_calibrations=[],
            trust_boundaries=[],
        )
    )
    sess.add(
        service_mod.SocialRoomRitualSummarySQL(
            ritual_key="callsyne:room-alpha",
            platform="callsyne",
            room_id="room-alpha",
            greeting_style="warm",
            reentry_style="grounded",
            thread_revival_style="direct",
            pause_handoff_style="brief",
            summary_cadence_preference=0.5,
            room_tone_summary="Warm and curious.",
            culture_summary="Older room ritual read is fading.",
            evidence_count=2,
            confidence=0.32,
            last_updated_at="2026-03-10T12:00:00+00:00",
        )
    )
    sess.commit()
    sess.close()

    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))
    selected_kinds = [item["candidate_kind"] for item in summary["context_window"]["selected_candidates"]]
    ritual_candidate = next(item for item in summary["context_candidates"] if item["candidate_kind"] == "ritual")

    assert "commitment" in selected_kinds
    assert ritual_candidate["inclusion_decision"] == "exclude"



def test_stale_calibration_candidate_softens_and_refresh_needed_hint_is_kept() -> None:
    svc, Session = _service_and_session()
    sess = Session()
    sess.add(
        service_mod.SocialParticipantContinuitySQL(
            peer_key="callsyne:room-alpha:peer-1",
            platform="callsyne",
            room_id="room-alpha",
            participant_id="peer-1",
            participant_name="CallSyne Peer",
            aliases=["CallSyne Peer"],
            participant_kind="peer_ai",
            recent_shared_topics=["pacing"],
            interaction_tone_summary="warm, direct, grounded",
            safe_continuity_summary="Recurring peer.",
            evidence_refs=["social-turn-old"],
            evidence_count=3,
            last_seen_at="2026-03-22T12:00:00+00:00",
            confidence=0.6,
            trust_tier="known",
            shared_artifact_scope="peer_local",
            shared_artifact_status="accepted",
            shared_artifact_summary="Accepted as peer-local continuity around pacing.",
            shared_artifact_reason="explicit keep/remember language made the scope legible",
            peer_calibration={
                "calibration_id": "cal-stale-1",
                "platform": "callsyne",
                "room_id": "room-alpha",
                "participant_id": "peer-1",
                "participant_name": "CallSyne Peer",
                "thread_key": "callsyne:room-alpha:thread:thread-1",
                "scope": "peer_thread",
                "calibration_kind": "unknown",
                "confidence": 0.28,
                "evidence_count": 1,
                "reversible": True,
                "decay_hint": "decay_after_topic_shift",
                "rationale": "Older calibration is weak.",
                "reasons": ["refresh_before_reusing_calibration"],
                "active_signal_ids": ["signal-1"],
                "caution_bias": 0.18,
                "attribution_bias": 0.18,
                "clarification_bias": 0.22,
                "updated_at": "2026-03-10T12:00:00+00:00",
                "metadata": {"source": "social-memory"},
            },
            trust_boundary=None,
            memory_freshness=[
                {
                    "freshness_id": "fresh-cal-1",
                    "platform": "callsyne",
                    "room_id": "room-alpha",
                    "participant_id": "peer-1",
                    "thread_key": "callsyne:room-alpha:thread:thread-1",
                    "artifact_kind": "peer_calibration",
                    "freshness_state": "refresh_needed",
                    "decay_level": "strong",
                    "regrounding_decision": "refresh_needed",
                    "confidence": 0.4,
                    "evidence_count": 1,
                    "last_updated_at": "2026-03-10T12:00:00+00:00",
                    "rationale": "Calibration should be refreshed before reuse.",
                    "reasons": ["low_evidence_count"],
                    "metadata": {"source": "social-memory"},
                }
            ],
            calibration_signals=[],
            decay_signals=[],
            regrounding_decisions=[],
        )
    )
    sess.commit()
    sess.close()

    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))
    candidates = summary["context_candidates"]
    calibration = next(item for item in candidates if item["candidate_kind"] == "calibration")
    freshness_hint = next(item for item in candidates if item["candidate_kind"] == "freshness_hint")

    assert calibration["inclusion_decision"] in {"soften", "exclude"}
    assert freshness_hint["inclusion_decision"] == "include"



def test_context_window_keeps_blocked_private_material_out() -> None:
    svc, _ = _service_and_session()
    asyncio.run(
        svc.process_social_turn(
            _payload(
                prompt="Keep this private and sealed.",
                response="Agreed, we won't carry it forward.",
            )
        )
    )

    summary = asyncio.run(svc.get_summary(platform="callsyne", room_id="room-alpha", participant_id="peer-1"))
    rendered = str(summary)

    assert "sealed_private" not in rendered
    assert "do not surface this" not in rendered
