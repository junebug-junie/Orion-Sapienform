from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from orion.cognition.verb_activation import is_active
from orion.cognition.workflows import (
    derive_workflow_execution_policy,
    resolve_user_workflow_invocation,
    resolve_workflow_schedule_management,
    workflow_registry_payload,
)
from orion.schemas.cortex.contracts import CortexChatRequest
from orion.cognition.answer_contract_normalize import build_answer_contract_draft_for_hub
from orion.schemas.social_memory import (
    SocialParticipantContinuityV1,
    SocialRoomContinuityV1,
    SocialStanceSnapshotV1,
)
from orion.schemas.social_context import (
    SocialContextCandidateV1,
    SocialContextSelectionDecisionV1,
    SocialContextWindowV1,
    SocialEpisodeSnapshotV1,
    SocialReentryAnchorV1,
)
from orion.schemas.social_gif import (
    SocialGifIntentV1,
    SocialGifInterpretationV1,
    SocialGifObservedSignalV1,
    SocialGifPolicyDecisionV1,
    SocialGifProxyContextV1,
)

logger = logging.getLogger("orion-hub.request-builder")

try:
    from scripts.social_room import (
        SOCIAL_ROOM_PROFILE,
        SOCIAL_ROOM_RECALL_PROFILE,
        SOCIAL_ROOM_VERB,
        build_social_artifact_dialogue,
        build_style_adaptation_snapshot,
        build_social_concept_evidence,
        build_social_grounding_state,
        build_social_inspection_debug,
        is_social_room_payload,
        resolve_social_skill_allowlist,
        select_social_room_skill,
    )
except ImportError:  # pragma: no cover - test/module-loader compatibility
    HERE = Path(__file__).resolve().parent
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from social_room import (  # type: ignore
        SOCIAL_ROOM_PROFILE,
        SOCIAL_ROOM_RECALL_PROFILE,
        SOCIAL_ROOM_VERB,
        build_social_artifact_dialogue,
        build_style_adaptation_snapshot,
        build_social_concept_evidence,
        build_social_grounding_state,
        build_social_inspection_debug,
        is_social_room_payload,
        resolve_social_skill_allowlist,
        select_social_room_skill,
    )


def _hub_social_skills_enabled() -> bool:
    return _normalize_flag(os.getenv("HUB_SOCIAL_SKILLS_ENABLED"), default=True)


def _hub_social_skill_allowlist() -> str:
    return os.getenv(
        "HUB_SOCIAL_SKILLS_ALLOWLIST",
        "social_artifact_dialogue,social_summarize_thread,social_safe_recall,social_self_ground,social_followup_question,social_room_reflection,social_exit_or_pause",
    )


def _hub_social_style_adaptation_enabled() -> bool:
    return _normalize_flag(os.getenv("HUB_SOCIAL_STYLE_ADAPTATION_ENABLED"), default=True)


def _hub_social_room_rituals_enabled() -> bool:
    return _normalize_flag(os.getenv("HUB_SOCIAL_ROOM_RITUALS_ENABLED"), default=True)


def _hub_social_style_confidence_floor() -> float:
    try:
        value = float(os.getenv("HUB_SOCIAL_STYLE_CONFIDENCE_FLOOR", "0.35"))
    except Exception:
        value = 0.35
    return max(0.0, min(value, 1.0))


def _normalize_mode(value: Any, *, default_mode: str, auto_default_enabled: bool) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"brain", "agent", "council", "auto"}:
        return mode
    return "auto" if auto_default_enabled else default_mode


def _normalize_skill_runner_lane(
    *,
    payload: Dict[str, Any],
    selected_ui_route: str,
    selected_verbs: list[str],
) -> tuple[str, list[str]]:
    if not _normalize_flag(payload.get("skill_runner_origin"), default=False):
        return selected_ui_route, selected_verbs
    lane = str(payload.get("skill_runner_lane") or "").strip().lower()
    if lane == "quick":
        if len(selected_verbs) == 1 and str(selected_verbs[0]).strip().startswith("skills."):
            return "brain", selected_verbs
        return "brain", ["chat_quick"]
    if lane == "agent":
        return "agent", []
    if lane == "brain":
        return "brain", selected_verbs
    return selected_ui_route, selected_verbs


def _import_skill_runner_catalogue_resolver():
    """Load resolve_skill_runner_catalogue_verb for both package and script-local imports."""
    try:
        from scripts.skill_runner_catalogue import resolve_skill_runner_catalogue_verb

        return resolve_skill_runner_catalogue_verb
    except ImportError:  # pragma: no cover - test/module-loader compatibility
        HERE = Path(__file__).resolve().parent
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        from skill_runner_catalogue import resolve_skill_runner_catalogue_verb  # type: ignore

        return resolve_skill_runner_catalogue_verb


def _normalize_flag(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def build_continuity_messages(
    *,
    history: Any,
    latest_user_prompt: str,
    turns: int = 10,
) -> List[Dict[str, str]]:
    """Build bounded user/assistant/system turns for CortexChatRequest.messages."""
    max_turns = max(0, int(turns or 0))
    max_msgs = 2 * max_turns if max_turns else 0
    normalized: List[Dict[str, str]] = []

    if isinstance(history, list):
        for item in history:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip().lower()
            if role not in {"user", "assistant", "system"}:
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            normalized.append({"role": role, "content": content})

    if max_msgs > 0 and len(normalized) > max_msgs:
        normalized = normalized[-max_msgs:]

    if not normalized:
        prompt = str(latest_user_prompt or "").strip()
        if prompt:
            normalized = [{"role": "user", "content": prompt}]

    return normalized


def _build_recall_payload(payload: Dict[str, Any], *, use_recall: bool, route_mode: str | None = None) -> Dict[str, Any]:
    recall_mode = str(payload.get("recall_mode") or "hybrid").strip() or "hybrid"
    recall_required = _normalize_flag(payload.get("recall_required"), default=False)
    recall_profile = payload.get("recall_profile")
    if isinstance(recall_profile, str):
        recall_profile = recall_profile.strip() or None
    # Agent-mode continuity should not inherit broad reflective recall unless the caller
    # intentionally selected a profile.
    if use_recall and recall_profile is None and str(route_mode or "").strip().lower() != "agent":
        recall_profile = "reflect.v1"

    return {
        "enabled": use_recall,
        "required": recall_required,
        "mode": recall_mode,
        "profile": recall_profile,
    }


def _build_social_epistemic_phrase_hint(
    *,
    epistemic_signal: Dict[str, Any],
    epistemic_decision: Dict[str, Any],
    room_continuity: Dict[str, Any],
) -> Dict[str, str] | None:
    if not epistemic_signal and not epistemic_decision:
        return None

    claim_kind = str(epistemic_signal.get("claim_kind") or "").strip().lower()
    decision = str(epistemic_decision.get("decision") or "").strip().lower()
    confidence = str(epistemic_signal.get("confidence_level") or "").strip().lower()
    ambiguity = str(epistemic_signal.get("ambiguity_level") or "").strip().lower()

    lead_in = ""
    caution = ""
    if claim_kind == "recall":
        lead_in = "Lead naturally with a memory frame such as 'From what I remember,' or 'As I remember it,'."
        caution = "Keep it to what is actually supported; don't overstate fuzzy details."
    elif claim_kind == "summary":
        lead_in = "Lead naturally with a compact summary frame such as 'Quick summary:' or 'Where we seem to be is...'."
        caution = "Stay with the active thread and don't broaden the room's ask."
    elif claim_kind == "inference":
        lead_in = "Lead naturally with an interpretive frame such as 'My read is...' or 'It seems like...'."
        caution = "Keep it sounding like a read, not a fact claim."
    elif claim_kind == "speculation":
        lead_in = "Lead naturally with a tentative frame such as 'Tentatively,' or 'My best guess is...'."
        caution = "Stay tentative and stick to visible room context."
    elif claim_kind == "clarification_needed" or decision == "ask_clarifying_question":
        lead_in = "Ask one short clarifying question first."
        caution = "Clarify scope, thread, or target before making a claim."
    elif claim_kind == "proposal":
        lead_in = "Keep the reply narrow and treat the topic as still pending rather than settled."
        caution = "Do not frame pending or declined shared-artifact state as accepted memory."

    if ambiguity in {"medium", "high"} and decision != "ask_clarifying_question":
        caution = (
            f"{caution} Default to the narrower thread and audience."
        ).strip()
    if confidence == "low" and claim_kind in {"recall", "inference", "speculation"}:
        caution = (
            f"{caution} Keep the wording modest."
        ).strip()
    active_claims = room_continuity.get("active_claims") if isinstance(room_continuity, dict) else []
    claim_states = {
        str(item.get("current_stance") or "").strip().lower()
        for item in (active_claims or [])
        if isinstance(item, dict)
    }
    if claim_states & {"provisional", "disputed", "corrected", "revised", "withdrawn"}:
        caution = (
            f"{caution} Track what was claimed versus what was corrected; don't turn provisional claims into settled fact."
        ).strip()
    consensus_states = {
        str(item.get("consensus_state") or "").strip().lower()
        for item in ((room_continuity.get("claim_consensus_states") or []) if isinstance(room_continuity, dict) else [])
        if isinstance(item, dict)
    }
    if consensus_states & {"partial", "contested", "corrected"}:
        caution = (
            f"{caution} Attribute who holds which view and avoid flattening contested or partial alignment into room consensus."
        ).strip()

    if not lead_in and not caution:
        return None
    return {
        "lead_in": lead_in,
        "caution": " ".join(part for part in (caution,) if part).strip(),
    }


def build_cortex_chat_request(
    *,
    payload: Dict[str, Any],
    session_id: str | None,
    user_id: str | None,
    trace_id: str | None,
    default_mode: str,
    auto_default_enabled: bool,
    source_label: str,
    prompt: str,
    messages: List[Dict[str, Any]] | None = None,
) -> Tuple[CortexChatRequest, Dict[str, Any], bool]:
    selected_ui_route = _normalize_mode(
        payload.get("mode"),
        default_mode=default_mode,
        auto_default_enabled=auto_default_enabled,
    )
    social_room = is_social_room_payload(payload)
    selected_verbs = [str(v).strip() for v in (payload.get("verbs") or []) if str(v).strip()]
    skill_runner_catalogue_verb: str | None = None
    if _normalize_flag(payload.get("skill_runner_origin"), default=False):
        resolve_skill_runner_catalogue_verb = _import_skill_runner_catalogue_resolver()
        resolved = resolve_skill_runner_catalogue_verb(
            prompt=str(prompt or "").strip(),
            skill_runner_origin=True,
        )
        if resolved:
            selected_verbs = [resolved]
            skill_runner_catalogue_verb = resolved
    selected_ui_route, selected_verbs = _normalize_skill_runner_lane(
        payload=payload,
        selected_ui_route=selected_ui_route,
        selected_verbs=selected_verbs,
    )
    if social_room:
        selected_ui_route = "brain"
    mode = selected_ui_route

    raw_recall = payload.get("use_recall", None)
    if raw_recall is None:
        use_recall = True
    elif isinstance(raw_recall, bool):
        use_recall = raw_recall
    elif isinstance(raw_recall, (int, float)):
        use_recall = bool(raw_recall)
    else:
        use_recall = str(raw_recall).strip().lower() in {"1", "true", "yes", "y", "on"}

    recall_payload = _build_recall_payload(payload, use_recall=use_recall, route_mode=selected_ui_route)
    if social_room:
        recall_payload["enabled"] = use_recall
        recall_payload["required"] = False
        recall_payload["mode"] = "hybrid"
        recall_payload["profile"] = SOCIAL_ROOM_RECALL_PROFILE if use_recall else None
    recall_payload["lane"] = "social" if social_room else "chat"
    recall_payload["profile_explicit"] = _normalize_flag(
        payload.get("recall_profile_explicit"), default=False
    )

    options = dict(payload.get("options") or {}) if isinstance(payload.get("options"), dict) else {}
    no_write_active = bool(payload.get("no_write", False))
    if selected_ui_route == "agent":
        options.setdefault("supervised", True)
    if social_room:
        options["tool_execution_policy"] = "none"
        options["action_execution_policy"] = "none"
    if no_write_active:
        options["tool_execution_policy"] = "none"
        options["action_execution_policy"] = "none"
        options["no_write_active"] = True

    workflow_match = None
    workflow_management = None
    workflow_request_override = payload.get("workflow_request_override") if isinstance(payload.get("workflow_request_override"), dict) else None
    workflow_resolution_reason = "social_room_profile"
    if not social_room:
        if isinstance(workflow_request_override, dict) and str(workflow_request_override.get("workflow_id") or "").strip():
            workflow_resolution_reason = "workflow_request_override"
        else:
            workflow_management = resolve_workflow_schedule_management(prompt, user_id=user_id, session_id=session_id)
            if workflow_management is not None:
                workflow_resolution_reason = "explicit_schedule_management_match"
            else:
                workflow_match = resolve_user_workflow_invocation(prompt)
                if workflow_match is not None:
                    workflow_resolution_reason = "explicit_named_workflow_match"
                else:
                    workflow_resolution_reason = "no_workflow_match"

    verb_override: str | None = None
    if workflow_match is not None or workflow_management is not None or workflow_request_override is not None:
        mode = "brain"
        selected_ui_route = "brain" if selected_ui_route == "auto" else selected_ui_route
        options.pop("route_intent", None)
        if selected_verbs:
            logger.info(
                "workflow_resolution_result %s",
                {
                    "matched_workflow_id": workflow_match.workflow_id if workflow_match is not None else None,
                    "fallback_route": "workflow_lane",
                    "reason": "named_workflow_precedence_over_selected_verbs",
                    "selected_verbs": selected_verbs,
                },
            )
    elif social_room and not selected_verbs:
        verb_override = SOCIAL_ROOM_VERB
    elif len(selected_verbs) == 1:
        verb_override = selected_verbs[0]
        options.pop("allowed_verbs", None)
        if mode == "auto":
            if verb_override == "agent_runtime":
                mode = "agent"
            elif verb_override == "council_runtime":
                mode = "council"
            else:
                mode = "brain"
    elif len(selected_verbs) > 1:
        options["allowed_verbs"] = selected_verbs

    route_intent = "none"
    if mode == "auto" and not verb_override and len(selected_verbs) == 0:
        route_intent = "auto"
        options["route_intent"] = "auto"
    else:
        options.pop("route_intent", None)
    if social_room:
        route_intent = "none"
        options.pop("route_intent", None)
    if workflow_match is not None or workflow_management is not None:
        route_intent = "none"
        options.pop("route_intent", None)

    metadata: Dict[str, Any] = {
        "source": source_label,
        "hub_route": {
            "selected_ui_route": selected_ui_route,
        },
        "available_workflows": workflow_registry_payload(user_invocable_only=True),
    }
    if isinstance(payload.get("presence_context"), dict):
        metadata["presence_context"] = payload.get("presence_context")
    if isinstance(payload.get("surface_context"), dict):
        metadata["surface_context"] = payload.get("surface_context")
    if payload.get("browser_client_id"):
        metadata["browser_client_id"] = str(payload.get("browser_client_id"))
    mutation_cognition_context = payload.get("mutation_cognition_context")
    if isinstance(mutation_cognition_context, dict) and mutation_cognition_context:
        metadata["mutation_cognition_context"] = mutation_cognition_context
    draft_ac = build_answer_contract_draft_for_hub(prompt)
    metadata["answer_contract_draft"] = draft_ac
    logger.info(
        "answer_contract_built source=hub request_kind=%s",
        draft_ac.get("request_kind"),
    )
    if workflow_request_override is not None:
        workflow_id = str(workflow_request_override.get("workflow_id") or "").strip()
        if workflow_id:
            metadata["workflow_request"] = dict(workflow_request_override)
            metadata["workflow_request"]["workflow_id"] = workflow_id
            metadata["workflow_request"]["invoked_from_chat"] = True
            metadata["workflow_request"].setdefault(
                "execution_policy",
                derive_workflow_execution_policy(
                    workflow_id=workflow_id,
                    prompt=prompt,
                    session_id=session_id,
                    user_id=user_id,
                ).model_dump(mode="json"),
            )
    elif workflow_match is not None:
        metadata["workflow_request"] = workflow_match.model_dump(mode="json")
        metadata["workflow_request"]["invoked_from_chat"] = True
        metadata["workflow_request"]["execution_policy"] = derive_workflow_execution_policy(
            workflow_id=workflow_match.workflow_id,
            prompt=prompt,
            session_id=session_id,
            user_id=user_id,
        ).model_dump(mode="json")
    elif workflow_management is not None:
        metadata["workflow_schedule_management"] = workflow_management.request.model_dump(mode="json")

    if social_room:
        peer_continuity = payload.get("social_peer_continuity") or {}
        room_continuity = payload.get("social_room_continuity") or {}
        stance_snapshot = payload.get("social_stance_snapshot") or {}
        peer_style_hint = payload.get("social_peer_style_hint") or {}
        room_ritual_summary = payload.get("social_room_ritual_summary") or {}
        bridge_summary = payload.get("social_bridge_summary") or ((room_continuity or {}).get("bridge_summary")) or {}
        clarifying_question = payload.get("social_clarifying_question") or ((room_continuity or {}).get("clarifying_question")) or {}
        deliberation_decision = payload.get("social_deliberation_decision") or ((room_continuity or {}).get("deliberation_decision")) or {}
        turn_handoff = payload.get("social_turn_handoff") or ((room_continuity or {}).get("turn_handoff")) or {}
        closure_signal = payload.get("social_closure_signal") or ((room_continuity or {}).get("closure_signal")) or {}
        floor_decision = payload.get("social_floor_decision") or ((room_continuity or {}).get("floor_decision")) or {}
        open_commitments = payload.get("social_open_commitments") or ((room_continuity or {}).get("active_commitments")) or []
        context_window = payload.get("social_context_window") or {}
        context_selection_decision = payload.get("social_context_selection_decision") or {}
        context_candidates = payload.get("social_context_candidates") or []
        episode_snapshot = payload.get("social_episode_snapshot") or {}
        reentry_anchor = payload.get("social_reentry_anchor") or {}
        gif_policy = payload.get("social_gif_policy") or {}
        gif_intent = payload.get("social_gif_intent") or {}
        gif_observed_signal = payload.get("social_gif_observed_signal") or {}
        gif_proxy_context = payload.get("social_gif_proxy_context") or {}
        gif_interpretation = payload.get("social_gif_interpretation") or {}
        thread_routing = payload.get("social_thread_routing") or ((payload.get("social_turn_policy") or {}).get("thread_routing")) or {}
        handoff_signal = payload.get("social_handoff_signal") or ((payload.get("social_turn_policy") or {}).get("handoff_signal")) or {}
        repair_signal = payload.get("social_repair_signal") or ((payload.get("social_turn_policy") or {}).get("repair_signal")) or {}
        repair_decision = payload.get("social_repair_decision") or ((payload.get("social_turn_policy") or {}).get("repair_decision")) or {}
        epistemic_signal = payload.get("social_epistemic_signal") or ((payload.get("social_turn_policy") or {}).get("epistemic_signal")) or {}
        epistemic_decision = payload.get("social_epistemic_decision") or ((payload.get("social_turn_policy") or {}).get("epistemic_decision")) or {}
        epistemic_phrase_hint = _build_social_epistemic_phrase_hint(
            epistemic_signal=epistemic_signal if isinstance(epistemic_signal, dict) else {},
            epistemic_decision=epistemic_decision if isinstance(epistemic_decision, dict) else {},
            room_continuity=room_continuity if isinstance(room_continuity, dict) else {},
        )
        skill_cfg = payload.get("social_skill_selection_config") or {}
        allowlist_raw = (
            skill_cfg.get("allowlist")
            if isinstance(skill_cfg, dict) and skill_cfg.get("allowlist") is not None
            else _hub_social_skill_allowlist()
        )
        skill_allowlist = resolve_social_skill_allowlist(
            ",".join(allowlist_raw) if isinstance(allowlist_raw, list) else allowlist_raw
        )
        artifact_proposal, artifact_revision, artifact_confirmation, _, _ = build_social_artifact_dialogue(
            payload=payload,
            prompt=prompt,
        )
        skills_enabled = _hub_social_skills_enabled()
        if isinstance(skill_cfg, dict) and skill_cfg.get("enabled") is not None:
            skills_enabled = _normalize_flag(skill_cfg.get("enabled"), default=_hub_social_skills_enabled())
        selection, skill_result, skill_request = select_social_room_skill(
            payload=payload,
            prompt=prompt,
            skills_enabled=skills_enabled,
            allowlist=skill_allowlist,
        )
        style_cfg = payload.get("social_style_config") or {}
        adaptation_enabled = _hub_social_style_adaptation_enabled()
        rituals_enabled = _hub_social_room_rituals_enabled()
        if isinstance(style_cfg, dict) and style_cfg.get("enabled") is not None:
            adaptation_enabled = _normalize_flag(style_cfg.get("enabled"), default=adaptation_enabled)
        if isinstance(style_cfg, dict) and style_cfg.get("rituals_enabled") is not None:
            rituals_enabled = _normalize_flag(style_cfg.get("rituals_enabled"), default=rituals_enabled)
        confidence_floor = _hub_social_style_confidence_floor()
        if isinstance(style_cfg, dict) and style_cfg.get("confidence_floor") is not None:
            try:
                confidence_floor = max(0.0, min(float(style_cfg.get("confidence_floor")), 1.0))
            except Exception:
                confidence_floor = _hub_social_style_confidence_floor()
        adaptation_snapshot = build_style_adaptation_snapshot(
            payload=payload,
            confidence_floor=confidence_floor,
            adaptation_enabled=adaptation_enabled,
            rituals_enabled=rituals_enabled,
        )
        metadata.update(
            {
                "chat_profile": SOCIAL_ROOM_PROFILE,
                "social_grounding_state": build_social_grounding_state(payload=payload).model_dump(mode="json"),
                "social_concept_evidence": [
                    item.model_dump(mode="json") for item in build_social_concept_evidence(payload.get("concept_evidence"))
                ],
                "social_peer_continuity": SocialParticipantContinuityV1.model_validate(peer_continuity).model_dump(mode="json") if peer_continuity else None,
                "social_room_continuity": SocialRoomContinuityV1.model_validate(room_continuity).model_dump(mode="json") if room_continuity else None,
                "social_stance_snapshot": SocialStanceSnapshotV1.model_validate(stance_snapshot).model_dump(mode="json") if stance_snapshot else None,
                "social_peer_style_hint": peer_style_hint or None,
                "social_room_ritual_summary": room_ritual_summary or None,
                "social_bridge_summary": bridge_summary or None,
                "social_clarifying_question": clarifying_question or None,
                "social_deliberation_decision": deliberation_decision or None,
                "social_turn_handoff": turn_handoff or None,
                "social_closure_signal": closure_signal or None,
                "social_floor_decision": floor_decision or None,
                "social_context_window": SocialContextWindowV1.model_validate(context_window).model_dump(mode="json") if context_window else None,
                "social_context_selection_decision": SocialContextSelectionDecisionV1.model_validate(context_selection_decision).model_dump(mode="json") if context_selection_decision else None,
                "social_context_candidates": [SocialContextCandidateV1.model_validate(item).model_dump(mode="json") for item in context_candidates[:8] if isinstance(item, dict)] if isinstance(context_candidates, list) else None,
                "social_episode_snapshot": SocialEpisodeSnapshotV1.model_validate(episode_snapshot).model_dump(mode="json") if episode_snapshot else None,
                "social_reentry_anchor": SocialReentryAnchorV1.model_validate(reentry_anchor).model_dump(mode="json") if reentry_anchor else None,
                "social_open_commitments": open_commitments[:2] if isinstance(open_commitments, list) else None,
                "social_thread_routing": thread_routing or None,
                "social_handoff_signal": handoff_signal or None,
                "social_repair_signal": repair_signal or None,
                "social_repair_decision": repair_decision or None,
                "social_epistemic_signal": epistemic_signal or None,
                "social_epistemic_decision": epistemic_decision or None,
                "social_gif_policy": SocialGifPolicyDecisionV1.model_validate(gif_policy).model_dump(mode="json") if gif_policy else None,
                "social_gif_intent": SocialGifIntentV1.model_validate(gif_intent).model_dump(mode="json") if gif_intent else None,
                "social_gif_observed_signal": SocialGifObservedSignalV1.model_validate(gif_observed_signal).model_dump(mode="json") if gif_observed_signal else None,
                "social_gif_proxy_context": SocialGifProxyContextV1.model_validate(gif_proxy_context).model_dump(mode="json") if gif_proxy_context else None,
                "social_gif_interpretation": SocialGifInterpretationV1.model_validate(gif_interpretation).model_dump(mode="json") if gif_interpretation else None,
                "social_epistemic_phrase_hint": epistemic_phrase_hint,
                "social_style_adaptation": adaptation_snapshot.model_dump(mode="json"),
                "social_artifact_proposal": artifact_proposal.model_dump(mode="json") if artifact_proposal else None,
                "social_artifact_revision": artifact_revision.model_dump(mode="json") if artifact_revision else None,
                "social_artifact_confirmation": artifact_confirmation.model_dump(mode="json") if artifact_confirmation else None,
                "social_skill_request": skill_request.model_dump(mode="json"),
                "social_skill_selection": selection.model_dump(mode="json"),
                "social_skill_result": skill_result.model_dump(mode="json") if skill_result else None,
            }
        )

    req = CortexChatRequest(
        prompt=prompt,
        messages=messages,
        mode=mode,
        route_intent=route_intent,
        session_id=session_id,
        user_id=user_id,
        trace_id=trace_id,
        packs=payload.get("packs"),
        verb=verb_override,
        options=options,
        recall=recall_payload,
        metadata=metadata,
    )
    diagnostic_value = payload.get("diagnostic")
    if diagnostic_value is None and isinstance(payload.get("options"), dict):
        diagnostic_value = payload.get("options", {}).get("diagnostic")

    debug = {
        "selected_ui_route": selected_ui_route,
        "mode": req.mode,
        "verb": req.verb,
        "route_intent": (req.options or {}).get("route_intent") or "none",
        "allowed_verbs_count": len(((req.options or {}).get("allowed_verbs") or [])),
        "packs": req.packs or [],
        "options": dict(req.options or {}),
        "recall_enabled": bool((req.recall or {}).get("enabled", True)),
        "recall_required": bool((req.recall or {}).get("required", False)),
        "recall_profile": (req.recall or {}).get("profile"),
        "supervised": _normalize_flag((req.options or {}).get("supervised"), default=False),
        "force_agent_chain": _normalize_flag((req.options or {}).get("force_agent_chain"), default=False),
        "diagnostic": _normalize_flag(diagnostic_value, default=False),
        "chat_profile": SOCIAL_ROOM_PROFILE if social_room else None,
        "workflow_id": metadata.get("workflow_request", {}).get("workflow_id") if isinstance(metadata.get("workflow_request"), dict) else None,
        "workflow_request": metadata.get("workflow_request"),
        "workflow_execution_policy": metadata.get("workflow_request", {}).get("execution_policy") if isinstance(metadata.get("workflow_request"), dict) else None,
        "workflow_management_operation": metadata.get("workflow_schedule_management", {}).get("operation") if isinstance(metadata.get("workflow_schedule_management"), dict) else None,
        "workflow_resolution_reason": workflow_resolution_reason,
        "workflow_requested": bool(workflow_match is not None or workflow_management is not None),
        "fallback_route": "workflow_lane" if (workflow_match is not None or workflow_management is not None) else "chat_or_auto_route",
        "skill_runner_catalogue_verb": skill_runner_catalogue_verb,
    }
    if social_room:
        debug["social_skill_allowlist"] = metadata.get("social_skill_request", {}).get("allowlist") or []
        debug["social_skill_selection"] = metadata.get("social_skill_selection")
        debug["social_skill_request"] = metadata.get("social_skill_request")
        debug["social_skill_result"] = metadata.get("social_skill_result")
        debug["social_style_adaptation"] = metadata.get("social_style_adaptation")
        debug["social_artifact_proposal"] = metadata.get("social_artifact_proposal")
        debug["social_artifact_revision"] = metadata.get("social_artifact_revision")
        debug["social_artifact_confirmation"] = metadata.get("social_artifact_confirmation")
        debug["social_inspection"] = build_social_inspection_debug(payload=payload, route_debug=metadata, metadata=metadata)
    return req, debug, use_recall


def validate_single_verb_override(
    payload: Dict[str, Any],
    *,
    node_name: str,
    prompt: str | None = None,
) -> Optional[Dict[str, Any]]:
    selected_verbs = [str(v).strip() for v in (payload.get("verbs") or []) if str(v).strip()]
    if len(selected_verbs) != 1:
        return None
    verb = selected_verbs[0]
    if _normalize_flag(payload.get("skill_runner_origin"), default=False) and prompt is not None:
        resolve_skill_runner_catalogue_verb = _import_skill_runner_catalogue_resolver()
        cat = resolve_skill_runner_catalogue_verb(
            prompt=str(prompt).strip(),
            skill_runner_origin=True,
        )
        if cat:
            verb = cat
    if is_active(verb, node_name=node_name):
        return None
    return {
        "error": f"inactive_verb:{verb}",
        "message": f"Verb '{verb}' is inactive on node {node_name}.",
        "verb": verb,
        "node": node_name,
    }


def build_chat_request(
    *,
    payload: Dict[str, Any],
    session_id: str | None,
    user_id: str | None,
    trace_id: str | None,
    default_mode: str,
    auto_default_enabled: bool,
    source_label: str,
    prompt: str,
    messages: List[Dict[str, Any]] | None = None,
) -> Tuple[CortexChatRequest, Dict[str, Any], bool]:
    """Compatibility wrapper: canonical Hub chat request builder for HTTP + WS."""
    return build_cortex_chat_request(
        payload=payload,
        session_id=session_id,
        user_id=user_id,
        trace_id=trace_id,
        default_mode=default_mode,
        auto_default_enabled=auto_default_enabled,
        source_label=source_label,
        prompt=prompt,
        messages=messages,
    )
