from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

from orion.schemas.social_inspection import (
    SocialInspectionDecisionTraceV1,
    SocialInspectionSectionV1,
    SocialInspectionSnapshotV1,
)

_BLOCKED_RE = re.compile(r"\b(sealed|private|password|secret|ssn|mirror|journal)\b", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def _compact_text(value: Any, *, limit: int = 180) -> str:
    text = _WS_RE.sub(" ", str(value or "")).strip()
    return text[:limit]


def _safe_text(value: Any, *, limit: int = 180) -> str:
    text = _compact_text(value, limit=limit)
    if not text:
        return ""
    if _BLOCKED_RE.search(text):
        return ""
    return text


def _append_unique(target: list[str], *values: str, limit: int = 6) -> None:
    for value in values:
        text = _safe_text(value)
        if text and text not in target:
            target.append(text)
        if len(target) >= limit:
            break


def _trace(
    *,
    trace_kind: str,
    decision_state: str,
    summary: str,
    why: str,
    source_ref: str = "",
    freshness_hint: str | None = None,
    confidence_hint: float | None = None,
    metadata: dict[str, str] | None = None,
) -> SocialInspectionDecisionTraceV1:
    return SocialInspectionDecisionTraceV1(
        trace_kind=trace_kind,
        decision_state=decision_state,  # type: ignore[arg-type]
        summary=_safe_text(summary, limit=220),
        why_it_mattered=_safe_text(why, limit=220),
        source_ref=_compact_text(source_ref, limit=120),
        freshness_hint=_compact_text(freshness_hint, limit=60) or None,
        confidence_hint=confidence_hint,
        metadata=metadata or {},
    )


def _make_section(
    *,
    section_kind: str,
    included: list[str],
    selected: list[str],
    softened: list[str],
    excluded: list[str],
    why: str,
    traces: list[SocialInspectionDecisionTraceV1],
    freshness_hints: list[str] | None = None,
    confidence_hints: list[str] | None = None,
    metadata: dict[str, str] | None = None,
) -> SocialInspectionSectionV1 | None:
    if not any((included, selected, softened, excluded, traces)):
        return None
    return SocialInspectionSectionV1(
        section_kind=section_kind,  # type: ignore[arg-type]
        included_artifact_summaries=included[:10],
        selected_state=selected[:6],
        softened_state=softened[:6],
        excluded_state=excluded[:6],
        freshness_hints=(freshness_hints or [])[:6],
        confidence_hints=(confidence_hints or [])[:6],
        why_this_mattered=_safe_text(why, limit=220),
        decision_traces=traces[:8],
        metadata=metadata or {},
    )


def _selected_candidates(window: Dict[str, Any]) -> list[Dict[str, Any]]:
    raw = window.get("selected_candidates")
    return raw if isinstance(raw, list) else []


def _candidate_bucket(candidates: Iterable[Dict[str, Any]], *, decision: str) -> list[Dict[str, Any]]:
    return [item for item in candidates if str(item.get("inclusion_decision") or "") == decision]


def build_social_inspection_snapshot(
    *,
    platform: str,
    room_id: str,
    participant_id: str | None,
    thread_key: str | None,
    surfaces: Dict[str, Any],
    source_surface: str,
    source_service: str,
) -> SocialInspectionSnapshotV1:
    window = dict(surfaces.get("social_context_window") or {})
    decision = dict(surfaces.get("social_context_selection_decision") or {})
    candidates = [item for item in (surfaces.get("social_context_candidates") or []) if isinstance(item, dict)]
    room = dict(surfaces.get("social_room_continuity") or {})
    peer = dict(surfaces.get("social_peer_continuity") or {})
    episode = dict(surfaces.get("social_episode_snapshot") or {})
    reentry = dict(surfaces.get("social_reentry_anchor") or {})
    routing = dict(surfaces.get("social_thread_routing") or {})
    repair_signal = dict(surfaces.get("social_repair_signal") or {})
    repair_decision = dict(surfaces.get("social_repair_decision") or {})
    epistemic_signal = dict(surfaces.get("social_epistemic_signal") or {})
    epistemic_decision = dict(surfaces.get("social_epistemic_decision") or {})
    artifact_proposal = dict(surfaces.get("social_artifact_proposal") or {})
    artifact_revision = dict(surfaces.get("social_artifact_revision") or {})
    artifact_confirmation = dict(surfaces.get("social_artifact_confirmation") or {})
    gif_policy = dict(surfaces.get("social_gif_policy") or {})
    gif_intent = dict(surfaces.get("social_gif_intent") or {})
    gif_observed = dict(surfaces.get("social_gif_observed_signal") or {})
    gif_proxy = dict(surfaces.get("social_gif_proxy_context") or {})
    gif_interpretation = dict(surfaces.get("social_gif_interpretation") or {})
    gif_usage = dict(room.get("gif_usage_state") or {})
    selected_candidates = _selected_candidates(window)

    sections: list[SocialInspectionSectionV1] = []
    top_traces: list[SocialInspectionDecisionTraceV1] = []
    safety_omissions = 0

    def _safe(value: Any, *, limit: int = 180) -> str:
        nonlocal safety_omissions
        text = _compact_text(value, limit=limit)
        if text and _BLOCKED_RE.search(text):
            safety_omissions += 1
            return ""
        return text

    included: list[str] = []
    softened: list[str] = []
    excluded: list[str] = []
    traces: list[SocialInspectionDecisionTraceV1] = []
    for item in selected_candidates[:6]:
        summary = _safe(item.get("summary"), limit=180)
        if summary:
            _append_unique(included, f"{item.get('candidate_kind')}: {summary}")
            if str(item.get("inclusion_decision")) == "soften":
                _append_unique(softened, summary)
            traces.append(
                _trace(
                    trace_kind="context_candidate",
                    decision_state="softened" if str(item.get("inclusion_decision")) == "soften" else "selected",
                    summary=f"{item.get('candidate_kind')}: {summary}",
                    why=item.get("rationale") or "Selected for the active context window.",
                    source_ref=item.get("reference_key") or "",
                    freshness_hint=str(item.get("freshness_band") or "") or None,
                    metadata={"priority_band": str(item.get("priority_band") or "")},
                )
            )
    for item in _candidate_bucket(candidates, decision="soften")[:4]:
        summary = _safe(item.get("summary"), limit=160)
        if summary:
            _append_unique(softened, f"{item.get('candidate_kind')}: {summary}")
    for item in _candidate_bucket(candidates, decision="exclude")[:4]:
        summary = _safe(item.get("summary"), limit=160)
        if summary:
            _append_unique(excluded, f"{item.get('candidate_kind')}: {summary}")
    if decision:
        top_traces.append(
            _trace(
                trace_kind="context_selection",
                decision_state="selected",
                summary=decision.get("rationale") or "Compact context window selected.",
                why="This determined which social-memory artifacts were allowed to govern the turn.",
                source_ref=decision.get("decision_id") or "",
            )
        )
    section = _make_section(
        section_kind="context_window",
        included=included,
        selected=[text for text in included if text not in softened][:6],
        softened=softened,
        excluded=excluded,
        why="Shows which context artifacts governed the turn and which stale/background artifacts were softened or excluded.",
        traces=traces,
        freshness_hints=[
            _safe(f"budget={decision.get('budget_max')}"),
            _safe(f"considered={window.get('total_candidates_considered')}"),
        ],
        metadata={"source_surface": source_surface},
    )
    if section:
        sections.append(section)

    claims_included: list[str] = []
    claims_softened: list[str] = []
    claims_excluded: list[str] = []
    claims_traces: list[SocialInspectionDecisionTraceV1] = []
    for claim in (room.get("active_claims") or [])[:2]:
        if not isinstance(claim, dict):
            continue
        summary = _safe(claim.get("normalized_summary") or claim.get("normalized_claim_key"))
        if summary:
            _append_unique(claims_included, f"claim: {summary}")
    for revision in (room.get("recent_claim_revisions") or [])[:2]:
        if not isinstance(revision, dict):
            continue
        summary = _safe(revision.get("revised_summary"))
        if summary:
            _append_unique(claims_included, f"revision: {summary}")
    for divergence in (room.get("claim_divergence_signals") or [])[:2]:
        if not isinstance(divergence, dict):
            continue
        summary = _safe(divergence.get("normalized_claim_key"))
        if summary:
            _append_unique(claims_included, f"divergence: {summary}")
            claims_traces.append(
                _trace(
                    trace_kind="claim_divergence",
                    decision_state="active",
                    summary=summary,
                    why="Active divergence keeps Orion from overstating consensus.",
                    source_ref=divergence.get("claim_id") or "",
                )
            )
    for consensus in (room.get("claim_consensus_states") or [])[:2]:
        if not isinstance(consensus, dict):
            continue
        summary = _safe(consensus.get("normalized_claim_key"))
        if summary:
            _append_unique(claims_softened, f"consensus: {summary}")
    for candidate in [item for item in candidates if item.get("candidate_kind") == "consensus" and item.get("inclusion_decision") == "exclude"][:2]:
        summary = _safe(candidate.get("summary"))
        if summary:
            _append_unique(claims_excluded, f"consensus: {summary}")
    section = _make_section(
        section_kind="claims",
        included=claims_included,
        selected=claims_included[:6],
        softened=claims_softened,
        excluded=claims_excluded,
        why="Captures claim state, revisions, attribution pressure, and divergence/consensus cues that shaped the reply.",
        traces=claims_traces,
        metadata={"source_surface": source_surface},
    )
    if section:
        sections.append(section)

    commitment_items: list[str] = []
    for commitment in (room.get("active_commitments") or [])[:3]:
        if not isinstance(commitment, dict):
            continue
        summary = _safe(commitment.get("summary"))
        if summary and str(commitment.get("state") or "") == "open":
            _append_unique(commitment_items, summary)
    section = _make_section(
        section_kind="commitments",
        included=commitment_items,
        selected=commitment_items[:6],
        softened=[],
        excluded=[],
        why="Open commitments stay visible because they create concrete obligations for the next turn.",
        traces=[],
        metadata={"source_surface": source_surface},
    )
    if section:
        sections.append(section)

    routing_items: list[str] = []
    routing_traces: list[SocialInspectionDecisionTraceV1] = []
    routing_summary = _safe(routing.get("thread_summary") or room.get("current_thread_summary"))
    if routing_summary:
        _append_unique(routing_items, routing_summary)
    if routing:
        routing_traces.append(
            _trace(
                trace_kind="routing",
                decision_state="active",
                summary=routing.get("routing_decision") or "routing_active",
                why=routing.get("rationale") or "The turn was routed against the current audience/thread interpretation.",
                source_ref=routing.get("thread_key") or "",
            )
        )
    section = _make_section(
        section_kind="routing",
        included=routing_items,
        selected=routing_items[:6],
        softened=[],
        excluded=[],
        why="Shows how the active audience/thread interpretation shaped the response target.",
        traces=routing_traces,
        metadata={"source_surface": source_surface},
    )
    if section:
        sections.append(section)

    repair_items: list[str] = []
    repair_traces: list[SocialInspectionDecisionTraceV1] = []
    for value in (
        repair_signal.get("repair_summary"),
        repair_signal.get("repair_need"),
        repair_decision.get("decision_kind"),
        repair_decision.get("rationale"),
    ):
        text = _safe(value)
        if text:
            _append_unique(repair_items, text)
    if repair_items:
        repair_traces.append(
            _trace(
                trace_kind="repair",
                decision_state="active",
                summary=repair_decision.get("decision_kind") or repair_signal.get("repair_need") or "repair_active",
                why=repair_decision.get("rationale") or "Repair state shaped tone or caution for the turn.",
            )
        )
    section = _make_section(
        section_kind="repair",
        included=repair_items,
        selected=repair_items[:6],
        softened=[],
        excluded=[],
        why="Repair signals explain why Orion may have slowed down, clarified, or repaired relational footing.",
        traces=repair_traces,
        metadata={"source_surface": source_surface},
    )
    if section:
        sections.append(section)

    deliberation_items: list[str] = []
    for value in (
        ((room.get("bridge_summary") or {}).get("summary_text") if isinstance(room.get("bridge_summary"), dict) else None),
        ((room.get("clarifying_question") or {}).get("question_text") if isinstance(room.get("clarifying_question"), dict) else None),
        ((room.get("deliberation_decision") or {}).get("decision_kind") if isinstance(room.get("deliberation_decision"), dict) else None),
    ):
        text = _safe(value)
        if text:
            _append_unique(deliberation_items, text)
    section = _make_section(
        section_kind="deliberation",
        included=deliberation_items,
        selected=deliberation_items[:6],
        softened=[],
        excluded=[],
        why="Bridge summaries, clarifying questions, and deliberation hints explain how Orion framed unresolved room meaning.",
        traces=[],
        metadata={"source_surface": source_surface},
    )
    if section:
        sections.append(section)

    floor_items: list[str] = []
    for value in (
        ((room.get("turn_handoff") or {}).get("handoff_text") if isinstance(room.get("turn_handoff"), dict) else None),
        ((room.get("closure_signal") or {}).get("closure_text") if isinstance(room.get("closure_signal"), dict) else None),
        ((room.get("floor_decision") or {}).get("decision_kind") if isinstance(room.get("floor_decision"), dict) else None),
    ):
        text = _safe(value)
        if text:
            _append_unique(floor_items, text)
    section = _make_section(
        section_kind="floor",
        included=floor_items,
        selected=floor_items[:6],
        softened=[],
        excluded=[],
        why="Floor state shows whether Orion was handing off, closing, or preserving turn-taking structure.",
        traces=[],
        metadata={"source_surface": source_surface},
    )
    if section:
        sections.append(section)

    calibration_items: list[str] = []
    for value in (
        ((peer.get("peer_calibration") or {}).get("rationale") if isinstance(peer.get("peer_calibration"), dict) else None),
        *((boundary.get("rationale") for boundary in (room.get("trust_boundaries") or [])[:2] if isinstance(boundary, dict))),
    ):
        text = _safe(value)
        if text:
            _append_unique(calibration_items, text)
    section = _make_section(
        section_kind="calibration",
        included=calibration_items,
        selected=calibration_items[:6],
        softened=[],
        excluded=[],
        why="Calibration and trust-boundary hints explain local caution, attribution, and clarification choices without changing truth conditions.",
        traces=[],
        metadata={"source_surface": source_surface},
    )
    if section:
        sections.append(section)

    freshness_items: list[str] = []
    freshness_excluded: list[str] = []
    for item in (peer.get("memory_freshness") or [])[:2] + (room.get("memory_freshness") or [])[:3]:
        if not isinstance(item, dict):
            continue
        summary = _safe(item.get("rationale"))
        if summary:
            _append_unique(freshness_items, f"{item.get('artifact_kind')}: {summary}")
    for candidate in [item for item in candidates if item.get("inclusion_decision") in {"soften", "exclude"} and item.get("freshness_band") in {"stale", "refresh_needed", "expired"}][:4]:
        summary = _safe(candidate.get("summary"))
        if summary:
            _append_unique(freshness_excluded, f"{candidate.get('candidate_kind')}: {summary}")
    section = _make_section(
        section_kind="freshness",
        included=freshness_items,
        selected=freshness_items[:6],
        softened=[],
        excluded=freshness_excluded,
        why="Freshness and re-grounding signals show what older state was intentionally softened, excluded, or reopened.",
        traces=[],
        metadata={"source_surface": source_surface},
    )
    if section:
        sections.append(section)

    resumptive_items: list[str] = []
    for value in (episode.get("summary"), episode.get("resumptive_hint"), reentry.get("anchor_text")):
        text = _safe(value)
        if text:
            _append_unique(resumptive_items, text)
    section = _make_section(
        section_kind="resumptive",
        included=resumptive_items,
        selected=resumptive_items[:6],
        softened=[],
        excluded=[],
        why="Episode snapshots and re-entry anchors show what resumptive context was available without letting it outrank live state.",
        traces=[],
        metadata={"source_surface": source_surface},
    )
    if section:
        sections.append(section)

    epistemic_items: list[str] = []
    epistemic_traces: list[SocialInspectionDecisionTraceV1] = []
    for value in (
        epistemic_signal.get("signal_kind"),
        epistemic_signal.get("signal_summary"),
        epistemic_decision.get("decision_kind"),
        epistemic_decision.get("rationale"),
    ):
        text = _safe(value)
        if text:
            _append_unique(epistemic_items, text)
    if epistemic_items:
        epistemic_traces.append(
            _trace(
                trace_kind="epistemic",
                decision_state="active",
                summary=epistemic_decision.get("decision_kind") or epistemic_signal.get("signal_kind") or "epistemic_active",
                why=epistemic_decision.get("rationale") or "Epistemic stance tuned certainty, attribution, or qualification for the turn.",
            )
        )
    section = _make_section(
        section_kind="epistemic",
        included=epistemic_items,
        selected=epistemic_items[:6],
        softened=[],
        excluded=[],
        why="Epistemic signals explain why Orion framed confidence, attribution, or uncertainty a certain way.",
        traces=epistemic_traces,
        metadata={"source_surface": source_surface},
    )
    if section:
        sections.append(section)

    artifact_excluded: list[str] = []
    artifact_traces: list[SocialInspectionDecisionTraceV1] = []
    for payload, label, state in (
        (artifact_proposal, "proposal", artifact_proposal.get("decision_state")),
        (artifact_revision, "revision", artifact_revision.get("decision_state")),
        (artifact_confirmation, "confirmation", artifact_confirmation.get("decision_state")),
    ):
        if not payload:
            continue
        _safe(
            payload.get("proposed_summary_text")
            or payload.get("revised_summary_text")
            or payload.get("confirmed_summary_text")
            or payload.get("rationale")
        )
        if str(state or "") in {"proposed", "clarify_scope", "declined", "deferred"}:
            artifact_excluded.append(f"{label}: pending or declined artifact dialogue stayed non-active")
            artifact_traces.append(
                _trace(
                    trace_kind="artifact_dialogue",
                    decision_state="omitted",
                    summary=f"{label} remained non-active",
                    why="Pending, deferred, or declined artifact dialogue must not be represented as active continuity.",
                )
            )
    section = _make_section(
        section_kind="artifact_dialogue",
        included=[],
        selected=[],
        softened=[],
        excluded=artifact_excluded,
        why="Makes it explicit when artifact dialogue existed but was intentionally kept out of active continuity.",
        traces=artifact_traces,
        metadata={"source_surface": source_surface},
    )
    if section:
        sections.append(section)

    gif_items: list[str] = []
    gif_excluded: list[str] = []
    gif_traces: list[SocialInspectionDecisionTraceV1] = []
    if gif_policy:
        _append_unique(
            gif_items,
            f"decision: {gif_policy.get('decision_kind')}",
            f"allowed={gif_policy.get('gif_allowed')}",
            f"intent={gif_policy.get('intent_kind')}",
        )
        if gif_policy.get("rationale"):
            _append_unique(gif_items, str(gif_policy.get("rationale")))
        if gif_intent.get("gif_query"):
            _append_unique(gif_items, f"query: {gif_intent.get('gif_query')}")
        if not gif_policy.get("gif_allowed"):
            for reason in (gif_policy.get("reasons") or [])[:4]:
                text = _safe(reason)
                if text:
                    _append_unique(gif_excluded, text)
        gif_traces.append(
            _trace(
                trace_kind="gif_policy",
                decision_state="active" if gif_policy.get("gif_allowed") else "excluded",
                summary=str(gif_policy.get("decision_kind") or "text_only"),
                why=gif_policy.get("rationale") or "GIF policy can only add bounded social garnish after the turn meaning is already understood.",
                source_ref=gif_policy.get("policy_id") or "",
            )
        )
    if gif_interpretation:
        _append_unique(
            gif_items,
            f"peer reaction={gif_interpretation.get('reaction_class')}",
            f"confidence={gif_interpretation.get('confidence_level')}",
            f"ambiguity={gif_interpretation.get('ambiguity_level')}",
        )
        disposition = str(gif_interpretation.get("cue_disposition") or "").strip()
        if disposition in {"softened", "ignored"}:
            _append_unique(gif_excluded, f"peer gif cue {disposition}")
        gif_traces.append(
            _trace(
                trace_kind="gif_interpretation",
                decision_state="selected" if disposition == "used" else "softened" if disposition == "softened" else "excluded",
                summary=str(gif_interpretation.get("reaction_class") or "unknown"),
                why=gif_interpretation.get("rationale") or "GIF proxy interpretation is metadata-only and should remain a soft social cue.",
                source_ref=gif_interpretation.get("interpretation_id") or "",
            )
        )
    if gif_observed:
        _append_unique(
            gif_items,
            f"peer_gif_present={gif_observed.get('media_present')}",
            f"proxy sources: {', '.join(gif_proxy.get('proxy_inputs_present') or []) or 'transport-only'}",
        )
        if gif_observed.get("provider"):
            _append_unique(gif_items, f"provider={gif_observed.get('provider')}")
        if gif_observed.get("transport_source"):
            _append_unique(gif_items, f"transport={gif_observed.get('transport_source')}")
    if gif_usage:
        _append_unique(
            gif_items,
            f"recent density={gif_usage.get('recent_gif_density')}",
            f"recent turns={gif_usage.get('recent_gif_turn_count')}",
            f"turns since last gif={gif_usage.get('turns_since_last_orion_gif')}",
        )
        last_intent = gif_usage.get("last_intent_kind")
        if last_intent:
            _append_unique(gif_items, f"last intent={last_intent}")
        recent_intents = [str(item) for item in (gif_usage.get("recent_intent_kinds") or []) if str(item).strip()]
        if recent_intents:
            _append_unique(gif_excluded, f"recent intent loop watch={', '.join(recent_intents[-3:])}")
        gif_traces.append(
            _trace(
                trace_kind="gif_usage_state",
                decision_state="active",
                summary=f"density={gif_usage.get('recent_gif_density')}",
                why="Live GIF shakedown needs recent density and intent history visible so repetitive or chaotic behavior is inspectable.",
                source_ref=gif_usage.get("usage_state_id") or "",
            )
        )
    section = _make_section(
        section_kind="gif",
        included=gif_items,
        selected=gif_items[:6] if (gif_policy.get("gif_allowed") or gif_interpretation.get("cue_disposition") == "used") else [],
        softened=[],
        excluded=gif_excluded,
        why="GIF policy keeps expressive garnish inspectable and subordinate to routing, safety, repair, and epistemic constraints.",
        traces=gif_traces,
        metadata={"source_surface": source_surface},
    )
    if section:
        sections.append(section)

    if safety_omissions:
        section = _make_section(
            section_kind="safety",
            included=[],
            selected=[],
            softened=[],
            excluded=[f"{safety_omissions} blocked/private summaries omitted from inspection"],
            why="Inspection stays bounded to already-safe social-room state and does not widen memory exposure.",
            traces=[
                _trace(
                    trace_kind="safety_filter",
                    decision_state="omitted",
                    summary="Blocked/private material omitted",
                    why="Inspection must not expose sealed, private, or similarly blocked text.",
                )
            ],
            metadata={"source_surface": source_surface},
        )
        if section:
            sections.append(section)

    summary_parts = [
        f"{len(sections)} sections",
        f"{len(selected_candidates)} context candidates selected",
        f"{len(_candidate_bucket(candidates, decision='soften'))} softened",
        f"{len(_candidate_bucket(candidates, decision='exclude'))} excluded",
    ]
    return SocialInspectionSnapshotV1(
        snapshot_id=f"{platform}:{room_id}:{thread_key or participant_id or 'room'}:inspection",
        platform=platform,
        room_id=room_id,
        thread_key=thread_key,
        participant_id=participant_id,
        summary=", ".join(summary_parts),
        sections=sections,
        decision_traces=(top_traces + [trace for section in sections for trace in section.decision_traces])[:16],
        metadata={
            "source_surface": source_surface,
            "source_service": source_service,
            "safety_omissions": str(safety_omissions),
            "tool_execution_available": "false",
            "action_execution_available": "false",
        },
    )
