from __future__ import annotations

import re
from typing import Iterable

from orion.core.schemas.substrate_mutation import MutationSignalV1
from orion.core.schemas.substrate_review_telemetry import GraphReviewTelemetryRecordV1


class MutationDetectors:
    """Convert runtime telemetry into typed mutation signals."""

    def __init__(self, *, allow_cognitive_lane: bool = False) -> None:
        self.allow_cognitive_lane = allow_cognitive_lane

    def from_review_telemetry(self, records: Iterable[GraphReviewTelemetryRecordV1]) -> list[MutationSignalV1]:
        signals: list[MutationSignalV1] = []
        for record in records:
            if not record.anchor_scope or not record.subject_ref or not record.target_zone:
                continue
            target_surface = _target_surface_for_zone(record.target_zone)
            pressure_event_signals = _signals_from_pressure_events(record=record, target_surface=target_surface)
            if pressure_event_signals:
                signals.extend(pressure_event_signals)
            signals.append(_build_base_signal(record=record, target_surface=target_surface))
            # Rich pressure path remains routing-only in this phase.
            if target_surface == "routing":
                signals.extend(_build_rich_routing_signals(record=record, target_surface=target_surface))
            if self.allow_cognitive_lane:
                signals.extend(_build_cognitive_signals_from_artifacts(record=record))
        return signals


def _build_base_signal(*, record: GraphReviewTelemetryRecordV1, target_surface: str) -> MutationSignalV1:
    strength = 0.3
    event_kind = "runtime_noop"
    if record.execution_outcome == "failed":
        strength = 0.9
        event_kind = "runtime_failure"
    elif record.execution_outcome in {"suppressed", "terminated"}:
        strength = 0.7
        event_kind = "runtime_suppression"
    elif record.execution_outcome == "executed":
        strength = 0.4 if "requeue_review" in record.consolidation_outcomes else 0.2
        event_kind = "runtime_review_churn" if "requeue_review" in record.consolidation_outcomes else "runtime_executed"
    return MutationSignalV1(
        event_kind=event_kind,
        anchor_scope=record.anchor_scope or "orion",
        subject_ref=record.subject_ref or "entity:orion",
        target_zone=record.target_zone or "autonomy_graph",
        target_surface=target_surface,
        strength=strength,
        evidence_refs=[
            f"telemetry:{record.telemetry_id}",
            f"runtime_outcome:{record.execution_outcome}",
        ],
        source_ref=f"review-telemetry:{record.telemetry_id}",
        metadata={
            "source_kind": "review_telemetry_runtime_outcome",
            "derived_signal_kind": event_kind,
            "confidence": strength,
        },
    )


def _build_rich_routing_signals(*, record: GraphReviewTelemetryRecordV1, target_surface: str) -> list[MutationSignalV1]:
    joined_notes = " ".join(str(item or "").lower() for item in record.notes)
    joined_outcomes = " ".join(str(item or "").lower() for item in record.consolidation_outcomes)
    selection_reason = str(record.selection_reason or "").lower()
    payload = f"{joined_notes} {joined_outcomes} {selection_reason}"
    rows: list[tuple[str, float, str, list[str]]] = []
    if _contains_any(payload, ("false_escalation", "operator_correction:downgrade", "false_downgrade", "operator_correction:escalate")):
        rows.append(
            (
                "routing_decision_mismatch",
                0.78,
                "routing_mismatch_signal",
                _evidence_tokens(payload, ("false_escalation", "false_downgrade", "operator_correction:downgrade", "operator_correction:escalate")),
            )
        )
    if _contains_any(payload, ("recall_miss", "missing_recall", "not_helpful", "dissatisfied", "dissatisfaction")):
        rows.append(
            (
                "routing_recall_dissatisfaction",
                0.62,
                "recall_dissatisfaction_signal",
                _evidence_tokens(payload, ("recall_miss", "missing_recall", "not_helpful", "dissatisfied", "dissatisfaction")),
            )
        )
    if record.degraded or record.runtime_duration_ms >= 1200 or _contains_any(payload, ("truncated", "finish_reason:length", "timeout")):
        rows.append(
            (
                "routing_runtime_degradation",
                0.68,
                "runtime_degradation_signal",
                _evidence_tokens(payload, ("truncated", "finish_reason:length", "timeout")),
            )
        )
    if _contains_any(payload, ("not_addressed", "addressed_only", "peer_targeted_elsewhere", "self_message_loop")):
        rows.append(
            (
                "routing_social_addressedness_gap",
                0.58,
                "social_addressedness_signal",
                _evidence_tokens(payload, ("not_addressed", "addressed_only", "peer_targeted_elsewhere", "self_message_loop")),
            )
        )
    signals: list[MutationSignalV1] = []
    for event_kind, strength, source_kind, evidence_tokens in rows:
        evidence_refs = [
            f"telemetry:{record.telemetry_id}",
            f"source_kind:{source_kind}",
            f"selection_reason:{(record.selection_reason or '')[:120]}",
        ]
        evidence_refs.extend(f"signal_hint:{item}" for item in evidence_tokens[:4])
        signals.append(
            MutationSignalV1(
                event_kind=event_kind,
                anchor_scope=record.anchor_scope or "orion",
                subject_ref=record.subject_ref or "entity:orion",
                target_zone=record.target_zone or "autonomy_graph",
                target_surface=target_surface,
                strength=strength,
                evidence_refs=evidence_refs[:32],
                source_ref=f"review-telemetry:{record.telemetry_id}",
                metadata={
                    "source_kind": source_kind,
                    "derived_signal_kind": event_kind,
                    "confidence": strength,
                    "telemetry_id": record.telemetry_id,
                },
            )
        )
    return signals


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _evidence_tokens(text: str, needles: tuple[str, ...]) -> list[str]:
    found = [needle for needle in needles if needle in text]
    if found:
        return found
    matches = re.findall(r"(?:operator_correction:[a-z_]+|false_[a-z_]+|recall_[a-z_]+|not_addressed|timeout|truncated)", text)
    return list(dict.fromkeys(matches))[:6]


def _target_surface_for_zone(zone: str) -> str:
    if zone == "autonomy_graph":
        return "routing"
    if zone == "world_ontology":
        return "graph_consolidation"
    if zone == "self_relationship_graph":
        return "prompt_profile"
    return "recall"


def _signals_from_pressure_events(*, record: GraphReviewTelemetryRecordV1, target_surface: str) -> list[MutationSignalV1]:
    if not record.pressure_events:
        return []
    routing_categories = {
        "routing_false_escalation",
        "routing_false_downgrade",
        "response_truncation_or_length_finish",
        "runtime_degradation_or_timeout",
        "social_addressedness_gap",
    }
    recall_surface_by_category = {
        "recall_miss_or_dissatisfaction": "recall_strategy_profile",
        "unsupported_memory_claim": "recall_strategy_profile",
        "irrelevant_semantic_neighbor": "recall_graph_expansion_policy",
        "missing_exact_anchor": "recall_anchor_policy",
        "stale_memory_selected": "recall_page_index_profile",
    }
    signals: list[MutationSignalV1] = []
    for event in record.pressure_events:
        category = str(event.pressure_category)
        if category in routing_categories:
            mapped_surface = "routing"
            mapped_zone = "autonomy_graph"
        else:
            mapped_surface = recall_surface_by_category.get(category)
            mapped_zone = "concept_graph"
        if not mapped_surface:
            continue
        event_kind = f"pressure_event:{event.pressure_category}"
        metadata = dict(event.metadata or {})
        compare_summary = metadata.get("v1_v2_compare") if isinstance(metadata.get("v1_v2_compare"), dict) else {}
        anchor_plan = metadata.get("anchor_plan") if isinstance(metadata.get("anchor_plan"), dict) else {}
        selected_cards = metadata.get("selected_evidence_cards") if isinstance(metadata.get("selected_evidence_cards"), list) else []
        evidence_refs = [
            f"pressure_event:{event.pressure_event_id}",
            f"source_service:{event.source_service}",
            f"source_event_id:{event.source_event_id}",
            f"pressure_category:{event.pressure_category}",
            *list(event.evidence_refs),
        ]
        if compare_summary:
            evidence_refs.extend(
                [
                    f"recall_compare:v1_latency_ms={compare_summary.get('v1_latency_ms')}",
                    f"recall_compare:v2_latency_ms={compare_summary.get('v2_latency_ms')}",
                    f"recall_compare:selected_count_delta={compare_summary.get('selected_count_delta')}",
                ]
            )
        if anchor_plan:
            evidence_refs.extend(
                [
                    f"anchor_plan:temporal_anchor={anchor_plan.get('temporal_anchor')}",
                    f"anchor_plan:time_window_days={anchor_plan.get('time_window_days')}",
                    f"anchor_plan:exact_anchor_tokens={len(list(anchor_plan.get('exact_anchor_tokens') or []))}",
                ]
            )
        for card in selected_cards[:4]:
            if isinstance(card, dict):
                evidence_refs.append(f"selected_card:{card.get('id')}")
        signals.append(
            MutationSignalV1(
                event_kind=event_kind,
                anchor_scope=record.anchor_scope or "orion",
                subject_ref=record.subject_ref or "entity:orion",
                target_zone=mapped_zone,
                target_surface=mapped_surface,
                strength=max(0.1, min(event.confidence, 1.0)),
                evidence_refs=evidence_refs[:32],
                source_ref=f"pressure-event:{event.pressure_event_id}",
                metadata={
                    "source_kind": "producer_pressure_event",
                    "derived_signal_kind": event_kind,
                    "confidence": event.confidence,
                    "source_service": event.source_service,
                    "source_event_id": event.source_event_id,
                    "source_correlation_id": event.correlation_id,
                    "pressure_category": event.pressure_category,
                    "pressure_event_id": event.pressure_event_id,
                    "recall_compare": compare_summary,
                    "anchor_plan": anchor_plan,
                    "selected_evidence_cards": selected_cards[:8],
                    "failure_category": category,
                    "recall_evidence_kind": str((event.metadata or {}).get("recall_evidence_kind") or "live_shadow"),
                    **(
                        {"recall_eval_case": (event.metadata or {}).get("recall_eval_case")}
                        if isinstance((event.metadata or {}).get("recall_eval_case"), dict)
                        else {}
                    ),
                    **(
                        {"suite_run_id": (event.metadata or {}).get("suite_run_id")}
                        if (event.metadata or {}).get("suite_run_id") is not None
                        else {}
                    ),
                },
            )
        )
    return signals


_COGNITIVE_SIGNAL_SPECS: tuple[tuple[str, str, tuple[str, ...], float, str], ...] = (
    (
        "contradiction_pressure",
        "cognitive_contradiction_reconciliation",
        ("routing_false_escalation", "routing_false_downgrade", "contradiction"),
        0.7,
        "contradiction_reconciliation_signal",
    ),
    (
        "identity_continuity_pressure",
        "cognitive_identity_continuity_adjustment",
        ("recall_miss_or_dissatisfaction", "identity_continuity", "memory_drift"),
        0.64,
        "identity_continuity_signal",
    ),
    (
        "stance_drift_pressure",
        "cognitive_stance_continuity_adjustment",
        ("response_truncation_or_length_finish", "runtime_degradation_or_timeout", "stance_drift"),
        0.6,
        "stance_drift_signal",
    ),
    (
        "social_continuity_pressure",
        "cognitive_social_continuity_repair",
        ("social_addressedness_gap", "not_addressed", "addressedness_gap"),
        0.62,
        "social_continuity_signal",
    ),
)


def _build_cognitive_signals_from_artifacts(*, record: GraphReviewTelemetryRecordV1) -> list[MutationSignalV1]:
    pressure_categories = {str(event.pressure_category) for event in record.pressure_events}
    notes_blob = " ".join(str(item or "").lower() for item in record.notes)
    reason_blob = str(record.selection_reason or "").lower()
    payload = f"{notes_blob} {reason_blob}"
    signals: list[MutationSignalV1] = []
    for event_kind, target_surface, needles, default_strength, source_kind in _COGNITIVE_SIGNAL_SPECS:
        matched = any(needle in pressure_categories for needle in needles) or _contains_any(payload, needles)
        if not matched:
            continue
        event_evidence = [
            f"telemetry:{record.telemetry_id}",
            f"signal_kind:{event_kind}",
            f"source_kind:{source_kind}",
        ]
        for event in record.pressure_events:
            if any(needle in str(event.pressure_category) for needle in needles):
                event_evidence.append(f"pressure_event:{event.pressure_event_id}")
        signals.append(
            MutationSignalV1(
                event_kind=event_kind,
                anchor_scope=record.anchor_scope or "orion",
                subject_ref=record.subject_ref or "entity:orion",
                target_zone="self_relationship_graph",
                target_surface=target_surface,
                strength=default_strength,
                evidence_refs=event_evidence[:32],
                source_ref=f"review-telemetry:{record.telemetry_id}",
                metadata={
                    "source_kind": source_kind,
                    "derived_signal_kind": event_kind,
                    "confidence": default_strength,
                    "telemetry_id": record.telemetry_id,
                    "pressure_categories": sorted(pressure_categories),
                },
            )
        )
    return signals
