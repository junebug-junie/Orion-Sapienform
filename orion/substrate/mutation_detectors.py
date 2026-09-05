from __future__ import annotations

from typing import Iterable

from orion.core.schemas.substrate_mutation import MutationSignalV1
from orion.core.schemas.substrate_review_telemetry import GraphReviewTelemetryRecordV1
from orion.substrate.mutation_contracts import PARKED_TARGET_SURFACES

# Single source of truth for zone -> surface, shared with _target_surface_for_zone
# below (which is the only caller in this module) and with PARKED_TELEMETRY_ZONES.
TARGET_SURFACE_BY_ZONE: dict[str, str] = {
    "world_ontology": "graph_consolidation",
    "self_relationship_graph": "prompt_profile",
}
_DEFAULT_TARGET_SURFACE = "recall"

# "autonomy_graph" zone retired 2026-09-05 along with the "routing" surface
# it existed solely to feed (see mutation_contracts.py's note on
# PARKED_TARGET_SURFACES and mutation_proposals.py's SURFACE_TO_CLASS).
# Unlike a parked surface (still a real, known target, just refused), this
# zone now has no live target at all -- falling through to
# _DEFAULT_TARGET_SURFACE would mislabel autonomy-graph telemetry as a
# recall signal, so it is skipped explicitly instead, same as
# _build_rich_routing_signals() (removed) used to be skipped for parked
# surfaces: no store write, no pressure-accumulation cycle, for a signal
# with nowhere to go.
_RETIRED_TELEMETRY_ZONES: frozenset[str] = frozenset({"autonomy_graph"})

# Telemetry zones whose base-signal target_surface is entirely parked (see
# mutation_contracts.PARKED_TARGET_SURFACES) -- i.e. a record in one of these
# zones can never produce a live base or rich-routing signal, only possibly a
# producer_pressure_event signal on a different, non-parked surface. Consumed
# by services/orion-hub/scripts/api_routes.py's signal-intake health check so
# it does not report a cycle "healthy" on the strength of rows that can only
# ever resolve to a parked surface. Currently always empty (nothing is
# parked as of 2026-09-05) -- kept as the mechanism for whatever is parked
# next, not routing-specific.
PARKED_TELEMETRY_ZONES: frozenset[str] = frozenset(
    zone for zone, surface in TARGET_SURFACE_BY_ZONE.items() if surface in PARKED_TARGET_SURFACES
) | _RETIRED_TELEMETRY_ZONES


class MutationDetectors:
    """Convert runtime telemetry into typed mutation signals."""

    def __init__(self, *, allow_cognitive_lane: bool = False) -> None:
        self.allow_cognitive_lane = allow_cognitive_lane

    def from_review_telemetry(self, records: Iterable[GraphReviewTelemetryRecordV1]) -> list[MutationSignalV1]:
        signals: list[MutationSignalV1] = []
        for record in records:
            if not record.anchor_scope or not record.subject_ref or not record.target_zone:
                continue
            # `_signals_from_pressure_events` can still route an individual
            # pressure_event to a non-parked surface (e.g. a recall category)
            # even for a retired/parked zone, so it always runs; only the
            # zone-derived base signal, which is retired-or-parked-or-not as
            # a whole, is worth skipping early.
            zone_has_no_live_surface = record.target_zone in _RETIRED_TELEMETRY_ZONES
            target_surface = None if zone_has_no_live_surface else _target_surface_for_zone(record.target_zone)
            pressure_event_signals = _signals_from_pressure_events(record=record)
            if pressure_event_signals:
                signals.extend(pressure_event_signals)
            if target_surface is not None and target_surface not in PARKED_TARGET_SURFACES:
                signals.append(_build_base_signal(record=record, target_surface=target_surface))
            if self.allow_cognitive_lane:
                signals.extend(_build_cognitive_signals_from_artifacts(record=record))
        # Trailing filter, kept even with the early skip above: a parked
        # surface is parked as of 2026-09-03 because its evidence (a review-
        # pipeline consolidation-outcome signal) has nothing to do with what
        # it gates, and mutation_proposals.py refuses it unconditionally
        # regardless -- so a signal for it must never reach the store or a
        # pressure-accumulation cycle just because it arrived via
        # `_signals_from_pressure_events`'s category-based routing instead of
        # the zone-based path the early skip above covers. CLAUDE.md 0A: a
        # signal excluded from one consumer but still ticking for every other
        # one is hiding, not retired.
        return [signal for signal in signals if signal.target_surface not in PARKED_TARGET_SURFACES]


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


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _target_surface_for_zone(zone: str) -> str:
    return TARGET_SURFACE_BY_ZONE.get(zone, _DEFAULT_TARGET_SURFACE)


# These categories used to map to the "routing" surface (retired 2026-09-05,
# see mutation_contracts.py/mutation_proposals.py) -- there is no live
# surface to redirect them to, so they now fall through
# `_signals_from_pressure_events`'s `if not mapped_surface: continue` guard
# instead of spending a store write on a signal that would only be refused
# three steps later at plan_for_pressure(). Module-scoped like
# TARGET_SURFACE_BY_ZONE above, not rebuilt per call/per record.
_RETIRED_PRESSURE_CATEGORIES: frozenset[str] = frozenset(
    {
        "routing_false_escalation",
        "routing_false_downgrade",
        "response_truncation_or_length_finish",
        "runtime_degradation_or_timeout",
        "social_addressedness_gap",
    }
)
_RECALL_SURFACE_BY_CATEGORY: dict[str, str] = {
    "recall_miss_or_dissatisfaction": "recall_strategy_profile",
    "unsupported_memory_claim": "recall_strategy_profile",
    "irrelevant_semantic_neighbor": "recall_graph_expansion_policy",
    "missing_exact_anchor": "recall_anchor_policy",
    "stale_memory_selected": "recall_page_index_profile",
}


def _signals_from_pressure_events(*, record: GraphReviewTelemetryRecordV1) -> list[MutationSignalV1]:
    if not record.pressure_events:
        return []
    signals: list[MutationSignalV1] = []
    for event in record.pressure_events:
        category = str(event.pressure_category)
        if category in _RETIRED_PRESSURE_CATEGORIES:
            continue
        mapped_surface = _RECALL_SURFACE_BY_CATEGORY.get(category)
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
