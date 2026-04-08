from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from jinja2 import Environment
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

from orion.schemas.social_scenario import (
    SocialScenarioEvaluationResultV1,
    SocialScenarioExpectationV1,
    SocialScenarioFixtureV1,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO_PACK = ROOT / "tests" / "fixtures" / "social_room" / "scenario_replay.json"
_PROMPT_TEMPLATE_PATH = ROOT / "orion" / "cognition" / "prompts" / "chat_social_room.j2"
_SOCIAL_MEMORY_ROOT = ROOT / "services" / "orion-social-memory"
_SOCIAL_BRIDGE_ROOT = ROOT / "services" / "orion-social-room-bridge"
_HUB_SCRIPTS_ROOT = ROOT / "services" / "orion-hub"

if str(_HUB_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_HUB_SCRIPTS_ROOT))

from scripts.cortex_request_builder import build_chat_request


def _clear_app_modules() -> None:
    for module_name in [name for name in sys.modules if name == "app" or name.startswith("app.")]:
        sys.modules.pop(module_name)


def _load_service_modules(service_root: Path, module_names: Sequence[str]) -> dict[str, Any]:
    _clear_app_modules()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(service_root))
    loaded: dict[str, Any] = {}
    try:
        for name in module_names:
            loaded[name] = importlib.import_module(name)
    finally:
        try:
            sys.path.remove(str(service_root))
        except ValueError:
            pass
        try:
            sys.path.remove(str(ROOT))
        except ValueError:
            pass
        _clear_app_modules()
    return loaded


_MEMORY_MODULES = _load_service_modules(
    _SOCIAL_MEMORY_ROOT,
    ["app.db", "app.models", "app.service", "app.settings"],
)
_BRIDGE_MODULES = _load_service_modules(
    _SOCIAL_BRIDGE_ROOT,
    ["app.service", "app.settings"],
)

_MEMORY_DB = _MEMORY_MODULES["app.db"]
_MEMORY_MODELS = _MEMORY_MODULES["app.models"]
_MEMORY_SERVICE_MOD = _MEMORY_MODULES["app.service"]
_MEMORY_SETTINGS_MOD = _MEMORY_MODULES["app.settings"]
_BRIDGE_SERVICE_MOD = _BRIDGE_MODULES["app.service"]
_BRIDGE_SETTINGS_MOD = _BRIDGE_MODULES["app.settings"]


class _FakeBus:
    enabled = True

    def __init__(self) -> None:
        self.published: list[tuple[str, Any]] = []

    async def publish(self, channel: str, envelope: Any) -> None:
        self.published.append((channel, envelope))

    async def close(self) -> None:
        return None


class _FakeHubClient:
    def __init__(self, reply_text: str = "I’m here in the room.") -> None:
        self.reply_text = reply_text
        self.calls: list[tuple[dict[str, Any], str]] = []

    async def chat(self, *, payload: dict[str, Any], session_id: str):
        self.calls.append((payload, session_id))
        return {"text": self.reply_text, "correlation_id": f"corr-{len(self.calls)}"}


class _FakeCallSyneClient:
    def __init__(self) -> None:
        self.posts: list[Any] = []

    async def post_message(self, request: Any):
        self.posts.append(request)
        return {"message_id": f"posted-{len(self.posts)}", "status": "posted"}


class _SocialMemoryAdapter:
    def __init__(self, service: Any) -> None:
        self.service = service

    async def get_summary(self, *, platform: str, room_id: str, participant_id: str | None):
        return await self.service.get_summary(platform=platform, room_id=room_id, participant_id=participant_id)


class SocialScenarioReplayHarness:
    def __init__(self) -> None:
        self._template = Environment().from_string(_PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8"))

    def run(self, scenarios: Sequence[SocialScenarioFixtureV1]) -> dict[str, Any]:
        results = [self.run_scenario(item) for item in scenarios]
        return {
            "summary": {
                "scenario_count": len(results),
                "passed_count": sum(1 for item in results if item.passed),
                "failed_scenarios": [item.scenario_id for item in results if not item.passed],
            },
            "results": [item.model_dump(mode="json") for item in results],
        }

    def run_scenario(self, fixture: SocialScenarioFixtureV1) -> SocialScenarioEvaluationResultV1:
        summary, inspection, bridge_result, policy_payload, hub_payload, request_metadata, rendered_prompt = asyncio.run(
            self._run_fixture(fixture)
        )
        observed = self._observed_outcomes(
            fixture=fixture,
            summary=summary,
            inspection=inspection,
            bridge_result=bridge_result,
            policy_payload=policy_payload,
            hub_payload=hub_payload,
            request_metadata=request_metadata,
            rendered_prompt=rendered_prompt,
        )
        mismatches, safety = self._evaluate_expectation(fixture.expectation, observed)
        return SocialScenarioEvaluationResultV1(
            platform=fixture.platform,
            room_id=fixture.room_id,
            scenario_id=fixture.scenario_id,
            passed=not mismatches,
            mismatch_reasons=mismatches,
            seams_exercised=observed.get("seams_exercised") or [],
            transcript_turn_count=len(fixture.transcript_turns),
            observed_outcomes=observed,
            safety_observations=safety,
            metadata={"description": fixture.description} | dict(fixture.metadata),
        )

    async def _run_fixture(
        self, fixture: SocialScenarioFixtureV1
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str]:
        memory_service, Session = self._memory_service()
        self._seed_state(Session=Session, fixture=fixture)

        live_message = None
        for index, turn in enumerate(fixture.transcript_turns, start=1):
            if turn.fixture_kind == "social_turn":
                await memory_service.process_social_turn(self._social_turn_payload(fixture=fixture, turn=turn, index=index))
            else:
                live_message = turn
        summary = await memory_service.get_summary(
            platform=fixture.platform,
            room_id=fixture.room_id,
            participant_id=fixture.participant_id,
        )
        inspection = await memory_service.get_inspection(
            platform=fixture.platform,
            room_id=fixture.room_id,
            participant_id=fixture.participant_id,
        )

        bridge_result: dict[str, Any] = {}
        policy_payload: dict[str, Any] = {}
        hub_payload: dict[str, Any] = {}
        request_metadata: dict[str, Any] = {}
        rendered_prompt = ""

        if live_message is not None:
            bridge_bus = _FakeBus()
            hub_client = _FakeHubClient()
            callsyne_client = _FakeCallSyneClient()
            bridge_service = _BRIDGE_SERVICE_MOD.SocialRoomBridgeService(
                settings=_BRIDGE_SETTINGS_MOD.Settings(ORION_BUS_ENABLED=False, SOCIAL_BRIDGE_SELF_PARTICIPANT_IDS="orion-room-bot"),
                hub_client=hub_client,
                callsyne_client=callsyne_client,
                social_memory_client=_SocialMemoryAdapter(memory_service),
                bus=bridge_bus,
            )
            bridge_result = await bridge_service.process_callsyne_message(self._bridge_message_payload(fixture=fixture, turn=live_message))
            policy_payload = self._latest_published_payload(bridge_bus, "orion:social:turn-policy")
            if hub_client.calls:
                hub_payload = dict(hub_client.calls[-1][0])

        prompt_payload = self._prompt_payload(
            fixture=fixture,
            hub_payload=hub_payload,
            summary=summary,
            policy_payload=policy_payload,
            live_message=live_message,
        )
        if prompt_payload:
            prompt_text = str((live_message.text if live_message is not None else "") or ((prompt_payload.get("messages") or [{}])[0].get("content") or ""))
            request, debug, _ = build_chat_request(
                payload=prompt_payload,
                session_id=f"scenario:{fixture.scenario_id}",
                user_id=fixture.participant_id,
                trace_id=f"trace:{fixture.scenario_id}",
                default_mode="brain",
                auto_default_enabled=True,
                source_label="scenario_replay",
                prompt=prompt_text,
            )
            request_metadata = dict(request.metadata or {})
            inspection = dict(debug.get("social_inspection") or inspection)
            rendered_prompt = self._template.render(metadata=request_metadata, memory_digest="")

        return summary, inspection, bridge_result, policy_payload, hub_payload, request_metadata, rendered_prompt

    def _memory_service(self):
        engine = create_engine("sqlite:///:memory:")
        Session = scoped_session(sessionmaker(bind=engine, autocommit=False, autoflush=False))
        _MEMORY_DB.Base.metadata.create_all(bind=engine)
        _MEMORY_SERVICE_MOD.get_session = lambda: Session()
        _MEMORY_SERVICE_MOD.remove_session = lambda: None
        svc = _MEMORY_SERVICE_MOD.SocialMemoryService(
            settings=_MEMORY_SETTINGS_MOD.Settings(ORION_BUS_ENABLED=False),
            bus=_FakeBus(),
        )
        return svc, Session

    def _seed_state(self, *, Session: Any, fixture: SocialScenarioFixtureV1) -> None:
        sess = Session()
        try:
            seed = fixture.seeded_state
            if seed.participant:
                data = {
                    "peer_key": f"{fixture.platform}:{fixture.room_id}:{fixture.participant_id}",
                    "platform": fixture.platform,
                    "room_id": fixture.room_id,
                    "participant_id": fixture.participant_id,
                    "participant_name": "CallSyne Peer",
                    "participant_kind": "peer_ai",
                    "recent_shared_topics": [],
                    "interaction_tone_summary": "",
                    "safe_continuity_summary": "",
                    "evidence_refs": [],
                    "evidence_count": 0,
                    "last_seen_at": "2026-03-22T12:00:00+00:00",
                    "confidence": 0.0,
                    "trust_tier": "known",
                    "shared_artifact_scope": "peer_local",
                    "shared_artifact_status": "unknown",
                    "shared_artifact_summary": "",
                    "shared_artifact_reason": "",
                    "calibration_signals": [],
                    "memory_freshness": [],
                    "decay_signals": [],
                    "regrounding_decisions": [],
                }
                data.update(seed.participant)
                sess.add(_MEMORY_MODELS.SocialParticipantContinuitySQL(**data))
            if seed.room:
                data = {
                    "room_key": f"{fixture.platform}:{fixture.room_id}",
                    "platform": fixture.platform,
                    "room_id": fixture.room_id,
                    "recurring_topics": [],
                    "active_participants": fixture.active_participants,
                    "recent_thread_summary": "",
                    "room_tone_summary": "",
                    "open_threads": [],
                    "evidence_refs": [],
                    "evidence_count": 0,
                    "last_updated_at": "2026-03-22T12:00:00+00:00",
                    "shared_artifact_scope": "room_local",
                    "shared_artifact_status": "unknown",
                    "shared_artifact_summary": "",
                    "shared_artifact_reason": "",
                    "active_threads": [],
                    "active_claims": [],
                    "recent_claim_revisions": [],
                    "claim_attributions": [],
                    "claim_consensus_states": [],
                    "claim_divergence_signals": [],
                    "active_commitments": [],
                    "calibration_signals": [],
                    "peer_calibrations": [],
                    "trust_boundaries": [],
                    "memory_freshness": [],
                    "decay_signals": [],
                    "regrounding_decisions": [],
                }
                data.update(seed.room)
                normalized_threads = []
                for idx, item in enumerate(list(data.get("active_threads") or []), start=1):
                    if not isinstance(item, dict):
                        continue
                    thread = {
                        "platform": fixture.platform,
                        "room_id": fixture.room_id,
                        "thread_id": item.get("thread_id") or str(item.get("thread_key") or f"thread-{idx}").split(":")[-1],
                        "active_participants": list(item.get("active_participants") or fixture.active_participants),
                        "last_speaker": item.get("last_speaker") or item.get("target_participant_name") or item.get("target_participant_id") or "peer",
                        "last_activity_at": item.get("last_activity_at") or data.get("last_updated_at") or "2026-03-22T12:00:00+00:00",
                        "expires_at": item.get("expires_at") or "2026-03-22T13:00:00+00:00",
                    }
                    thread.update(item)
                    normalized_threads.append(thread)
                data["active_threads"] = normalized_threads
                sess.add(_MEMORY_MODELS.SocialRoomContinuitySQL(**data))
            if seed.stance:
                data = {
                    "stance_id": "orion-social-room",
                    "curiosity": 0.7,
                    "warmth": 0.8,
                    "directness": 0.65,
                    "playfulness": 0.35,
                    "caution": 0.4,
                    "depth_preference": 0.6,
                    "recent_social_orientation_summary": "Recent social stance leans warm, curious, direct.",
                    "evidence_refs": [],
                    "evidence_count": 0,
                    "last_updated_at": "2026-03-22T12:00:00+00:00",
                }
                data.update(seed.stance)
                sess.add(_MEMORY_MODELS.SocialStanceSnapshotSQL(**data))
            if seed.peer_style:
                data = {
                    "peer_style_key": f"{fixture.platform}:{fixture.room_id}:{fixture.participant_id}",
                    "platform": fixture.platform,
                    "room_id": fixture.room_id,
                    "participant_id": fixture.participant_id,
                    "participant_name": "CallSyne Peer",
                    "style_hints_summary": "",
                    "preferred_directness": 0.5,
                    "preferred_depth": 0.5,
                    "question_appetite": 0.5,
                    "playfulness_tendency": 0.3,
                    "formality_tendency": 0.5,
                    "summarization_preference": 0.3,
                    "evidence_count": 0,
                    "confidence": 0.0,
                    "last_updated_at": "2026-03-22T12:00:00+00:00",
                }
                data.update(seed.peer_style)
                sess.add(_MEMORY_MODELS.SocialPeerStyleHintSQL(**data))
            if seed.room_ritual:
                data = {
                    "ritual_key": f"{fixture.platform}:{fixture.room_id}",
                    "platform": fixture.platform,
                    "room_id": fixture.room_id,
                    "greeting_style": "warm",
                    "reentry_style": "grounded",
                    "thread_revival_style": "direct",
                    "pause_handoff_style": "brief",
                    "summary_cadence_preference": 0.3,
                    "room_tone_summary": "",
                    "culture_summary": "",
                    "evidence_count": 0,
                    "confidence": 0.0,
                    "last_updated_at": "2026-03-22T12:00:00+00:00",
                }
                data.update(seed.room_ritual)
                sess.add(_MEMORY_MODELS.SocialRoomRitualSummarySQL(**data))
            sess.commit()
        finally:
            sess.close()

    def _social_turn_payload(
        self,
        *,
        fixture: SocialScenarioFixtureV1,
        turn: Any,
        index: int,
    ) -> dict[str, Any]:
        client_meta = {
            "chat_profile": "social_room",
            "external_room": {
                "platform": fixture.platform,
                "room_id": fixture.room_id,
                "thread_id": turn.thread_id or f"thread-{index}",
                "transport_message_id": turn.message_id or f"msg-social-{index}",
                "target_participant_id": turn.target_participant_id,
                "target_participant_name": turn.target_participant_name,
            },
            "external_participant": {
                "participant_id": turn.participant_id,
                "participant_name": turn.participant_name,
                "participant_kind": turn.participant_kind,
            },
        }
        client_meta.update(turn.client_meta)
        return {
            "turn_id": turn.turn_id or f"social-turn-{fixture.scenario_id}-{index}",
            "correlation_id": turn.correlation_id or f"corr-{fixture.scenario_id}-{index}",
            "session_id": f"scenario:{fixture.scenario_id}",
            "user_id": turn.participant_id,
            "source": "scenario_replay",
            "profile": "social_room",
            "prompt": turn.prompt,
            "response": turn.response,
            "text": f"User: {turn.prompt}\nOrion: {turn.response}",
            "created_at": turn.created_at or "2026-03-22T12:00:00+00:00",
            "stored_at": turn.stored_at or "2026-03-22T12:00:01+00:00",
            "recall_profile": "social.room.v1",
            "trace_verb": "chat_social_room",
            "tags": ["social_room", "chat_social_room", fixture.scenario_id],
            "concept_evidence": [],
            "grounding_state": {
                "profile": "social_room",
                "identity_label": "Oríon",
                "relationship_frame": "peer",
                "self_model_hint": "distributed social presence",
                "continuity_anchor": f"{fixture.platform}:{fixture.room_id}",
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
            "client_meta": client_meta,
        }

    def _bridge_message_payload(self, *, fixture: SocialScenarioFixtureV1, turn: Any) -> dict[str, Any]:
        return {
            "platform": fixture.platform,
            "room_id": fixture.room_id,
            "thread_id": turn.thread_id or "thread-bridge",
            "message_id": turn.message_id or f"msg-{fixture.scenario_id}",
            "sender_id": turn.participant_id,
            "sender_name": turn.participant_name,
            "sender_kind": turn.participant_kind,
            "text": turn.text,
            "mentions_orion": turn.mentions_orion,
            "created_at": turn.created_at or "2026-03-22T12:00:00+00:00",
            "reply_to_message_id": turn.reply_to_message_id,
            "reply_to_sender_id": turn.reply_to_sender_id,
            "reply_to_sender_name": turn.reply_to_sender_name,
            "target_participant_id": turn.target_participant_id,
            "target_participant_name": turn.target_participant_name,
            "mentioned_participant_ids": list(turn.mentioned_participant_ids or []),
            "mentioned_participant_names": list(turn.mentioned_participant_names or []),
            "metadata": {"transport": "scenario-replay"} | dict(turn.metadata),
        }

    def _prompt_payload(
        self,
        *,
        fixture: SocialScenarioFixtureV1,
        hub_payload: dict[str, Any],
        summary: dict[str, Any],
        policy_payload: dict[str, Any],
        live_message: Any | None,
    ) -> dict[str, Any]:
        if fixture.prompt_fixture_kind == "custom_payload":
            payload = {
                "chat_profile": "social_room",
                "mode": "brain",
                "messages": [{"role": "user", "content": live_message.text if live_message is not None else fixture.description}],
            }
            payload.update(fixture.prompt_payload_overrides)
            return payload
        payload = dict(hub_payload)
        if not payload and live_message is not None:
            payload = {
                "chat_profile": "social_room",
                "mode": "brain",
                "messages": [{"role": "user", "content": live_message.text}],
                "user_id": live_message.participant_id,
                "social_thread_routing": (policy_payload.get("thread_routing") or None),
                "social_repair_signal": policy_payload.get("repair_signal") or None,
                "social_repair_decision": policy_payload.get("repair_decision") or None,
                "social_epistemic_signal": policy_payload.get("epistemic_signal") or None,
                "social_epistemic_decision": policy_payload.get("epistemic_decision") or None,
                "social_peer_continuity": summary.get("participant"),
                "social_room_continuity": summary.get("room"),
                "social_stance_snapshot": summary.get("stance"),
                "social_peer_style_hint": summary.get("peer_style"),
                "social_room_ritual_summary": summary.get("room_ritual"),
                "social_context_window": summary.get("context_window"),
                "social_context_selection_decision": summary.get("context_selection_decision"),
                "social_context_candidates": summary.get("context_candidates"),
                "external_room": {
                    "platform": fixture.platform,
                    "room_id": fixture.room_id,
                    "thread_id": live_message.thread_id,
                    "transport_message_id": live_message.message_id,
                    "target_participant_id": live_message.target_participant_id,
                    "target_participant_name": live_message.target_participant_name,
                },
                "external_participant": {
                    "participant_id": live_message.participant_id,
                    "participant_name": live_message.participant_name,
                    "participant_kind": live_message.participant_kind,
                },
            }
        if fixture.prompt_payload_overrides:
            payload.update(fixture.prompt_payload_overrides)
        return payload

    def _observed_outcomes(
        self,
        *,
        fixture: SocialScenarioFixtureV1,
        summary: dict[str, Any],
        inspection: dict[str, Any],
        bridge_result: dict[str, Any],
        policy_payload: dict[str, Any],
        hub_payload: dict[str, Any],
        request_metadata: dict[str, Any],
        rendered_prompt: str,
    ) -> dict[str, Any]:
        room = dict(summary.get("room") or {})
        context_candidates = list(summary.get("context_candidates") or [])
        selected_context = [item.get("candidate_kind") for item in ((summary.get("context_window") or {}).get("selected_candidates") or []) if isinstance(item, dict)]
        softened_context = [item.get("candidate_kind") for item in context_candidates if item.get("inclusion_decision") == "soften"]
        excluded_context = [item.get("candidate_kind") for item in context_candidates if item.get("inclusion_decision") == "exclude"]
        inspection_sections = [item.get("section_kind") for item in (inspection.get("sections") or []) if isinstance(item, dict)]
        phrase_hint = dict(request_metadata.get("social_epistemic_phrase_hint") or {})
        safety_blob = json.dumps(
            {
                "summary": summary,
                "inspection": inspection,
                "request_metadata": request_metadata,
                "rendered_prompt": rendered_prompt,
            },
            default=str,
            sort_keys=True,
        )
        seams = ["social-memory-summary", "social-memory-inspection"]
        if policy_payload:
            seams.append("bridge-routing-policy")
        if request_metadata:
            seams.extend(["hub-request-builder", "prompt-grounding"])
        return {
            "bridge_result": bridge_result,
            "turn_policy": policy_payload,
            "hub_payload": hub_payload,
            "request_metadata": request_metadata,
            "rendered_prompt": rendered_prompt,
            "summary": summary,
            "inspection": inspection,
            "routing_decision": ((policy_payload.get("thread_routing") or {}).get("routing_decision")),
            "audience_scope": ((policy_payload.get("thread_routing") or {}).get("audience_scope")),
            "turn_policy_decision": policy_payload.get("decision"),
            "repair_decision": ((policy_payload.get("repair_decision") or {}).get("decision")),
            "repair_type": ((policy_payload.get("repair_signal") or {}).get("repair_type")),
            "epistemic_claim_kind": ((policy_payload.get("epistemic_signal") or {}).get("claim_kind")),
            "epistemic_decision": ((policy_payload.get("epistemic_decision") or {}).get("decision")),
            "epistemic_lead_in": phrase_hint.get("lead_in") or "",
            "deliberation_decision_kind": (room.get("deliberation_decision") or {}).get("decision_kind"),
            "floor_decision_kind": (room.get("floor_decision") or {}).get("decision_kind"),
            "bridge_summary_present": bool(room.get("bridge_summary")),
            "clarifying_question_present": bool(room.get("clarifying_question")),
            "handoff_decision_kind": (room.get("turn_handoff") or {}).get("decision_kind"),
            "closure_present": bool(room.get("closure_signal")),
            "gif_decision_kind": (request_metadata.get("social_gif_policy") or {}).get("decision_kind"),
            "gif_allowed": bool((request_metadata.get("social_gif_policy") or {}).get("gif_allowed")),
            "gif_intent_kind": (request_metadata.get("social_gif_policy") or {}).get("intent_kind"),
            "peer_gif_present": bool((request_metadata.get("social_gif_observed_signal") or {}).get("media_present")),
            "peer_gif_reaction_class": (request_metadata.get("social_gif_interpretation") or {}).get("reaction_class"),
            "peer_gif_confidence": (request_metadata.get("social_gif_interpretation") or {}).get("confidence_level"),
            "peer_gif_ambiguity": (request_metadata.get("social_gif_interpretation") or {}).get("ambiguity_level"),
            "peer_gif_cue_disposition": (request_metadata.get("social_gif_interpretation") or {}).get("cue_disposition"),
            "gif_reasons": list((request_metadata.get("social_gif_policy") or {}).get("reasons") or []),
            "gif_score": ((request_metadata.get("social_gif_policy") or {}).get("metadata") or {}).get("score"),
            "gif_transport_degraded": (((request_metadata.get("social_gif_policy") or {}).get("metadata") or {}).get("transport_degraded")),
            "gif_transport_metadata": bridge_result.get("post_metadata") or {},
            "gif_media_hint_present": bool((bridge_result.get("post_metadata") or {}).get("media_hint")),
            "selected_context_kinds": [item for item in selected_context if item],
            "softened_context_kinds": [item for item in softened_context if item],
            "excluded_context_kinds": [item for item in excluded_context if item],
            "inspection_section_kinds": [item for item in inspection_sections if item],
            "pending_artifact_non_active": self._pending_artifact_non_active(summary, rendered_prompt),
            "private_material_blocked": self._private_material_blocked(fixture.expectation, safety_blob),
            "blocked_strings_absent": {item: item not in safety_blob for item in fixture.expectation.blocked_strings},
            "seams_exercised": seams,
        }

    def _evaluate_expectation(
        self,
        expectation: SocialScenarioExpectationV1,
        observed: dict[str, Any],
    ) -> tuple[list[str], list[str]]:
        mismatches: list[str] = []
        safety: list[str] = []
        scalar_expectations = {
            "turn_policy_decision": expectation.turn_policy_decision,
            "routing_decision": expectation.routing_decision,
            "audience_scope": expectation.audience_scope,
            "repair_decision": expectation.repair_decision,
            "repair_type": expectation.repair_type,
            "epistemic_claim_kind": expectation.epistemic_claim_kind,
            "epistemic_decision": expectation.epistemic_decision,
            "deliberation_decision_kind": expectation.deliberation_decision_kind,
            "floor_decision_kind": expectation.floor_decision_kind,
            "handoff_decision_kind": expectation.handoff_decision_kind,
            "gif_decision_kind": expectation.gif_decision_kind,
            "gif_intent_kind": expectation.gif_intent_kind,
        }
        for key, expected in scalar_expectations.items():
            if expected is not None and observed.get(key) != expected:
                mismatches.append(f"{key}: expected {expected!r}, observed {observed.get(key)!r}")

        if expectation.epistemic_lead_in_contains and expectation.epistemic_lead_in_contains not in str(observed.get("epistemic_lead_in") or ""):
            mismatches.append(
                f"epistemic_lead_in: expected substring {expectation.epistemic_lead_in_contains!r}, observed {observed.get('epistemic_lead_in')!r}"
            )

        bool_expectations = {
            "bridge_summary_present": expectation.bridge_summary_expected,
            "clarifying_question_present": expectation.clarifying_question_expected,
            "closure_present": expectation.closure_expected,
            "gif_allowed": expectation.gif_allowed_expected,
        }
        for key, expected in bool_expectations.items():
            if expected is not None and bool(observed.get(key)) != expected:
                mismatches.append(f"{key}: expected {expected!r}, observed {bool(observed.get(key))!r}")

        for label, expected_values in {
            "selected_context_kinds": expectation.selected_context_kinds,
            "softened_context_kinds": expectation.softened_context_kinds,
            "excluded_context_kinds": expectation.excluded_context_kinds,
            "inspection_section_kinds": expectation.inspection_section_kinds,
        }.items():
            actual = set(observed.get(label) or [])
            for value in expected_values:
                if value not in actual:
                    mismatches.append(f"{label}: expected {value!r} in {sorted(actual)!r}")

        rendered_prompt = str(observed.get("rendered_prompt") or "")
        for text in expectation.prompt_must_contain:
            if text not in rendered_prompt:
                mismatches.append(f"rendered_prompt missing required text {text!r}")
        for text in expectation.prompt_must_not_contain:
            if text in rendered_prompt:
                mismatches.append(f"rendered_prompt unexpectedly contained {text!r}")

        for blocked in expectation.blocked_strings:
            is_absent = bool((observed.get("blocked_strings_absent") or {}).get(blocked, False))
            if is_absent:
                safety.append(f"blocked string remained absent: {blocked}")
            else:
                mismatches.append(f"blocked string leaked into evaluation surfaces: {blocked!r}")

        if expectation.require_pending_artifact_non_active:
            if observed.get("pending_artifact_non_active"):
                safety.append("pending artifact dialogue stayed non-active")
            else:
                mismatches.append("pending artifact dialogue became active continuity")

        if expectation.require_private_material_blocked:
            if observed.get("private_material_blocked"):
                safety.append("private material stayed blocked in replay output")
            else:
                mismatches.append("private material appeared in replay output")

        return mismatches, safety

    def _pending_artifact_non_active(self, summary: dict[str, Any], rendered_prompt: str) -> bool:
        participant = dict(summary.get("participant") or {})
        room = dict(summary.get("room") or {})
        pending_present = any(
            isinstance(container.get(key), dict)
            for container in (participant, room)
            for key in ("shared_artifact_proposal", "shared_artifact_revision")
        )
        if not pending_present:
            return True
        statuses = {str(container.get("shared_artifact_status") or "unknown") for container in (participant, room)}
        if any(status == "accepted" for status in statuses):
            return False
        accepted_cue = "status=accepted" in rendered_prompt
        return not accepted_cue

    def _private_material_blocked(self, expectation: SocialScenarioExpectationV1, safety_blob: str) -> bool:
        if expectation.blocked_strings:
            return all(text not in safety_blob for text in expectation.blocked_strings)
        return True

    def _latest_published_payload(self, bus: _FakeBus, channel: str) -> dict[str, Any]:
        for published_channel, envelope in reversed(bus.published):
            if published_channel == channel:
                return dict(getattr(envelope, "payload", None) or {})
        return {}


def load_scenarios(path: Path | str = DEFAULT_SCENARIO_PACK, *, only: Iterable[str] | None = None) -> list[SocialScenarioFixtureV1]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    scenarios = [SocialScenarioFixtureV1.model_validate(item) for item in raw.get("scenarios") or []]
    if only is None:
        return scenarios
    allowed = set(only)
    return [item for item in scenarios if item.scenario_id in allowed]
