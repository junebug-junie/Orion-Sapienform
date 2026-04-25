from __future__ import annotations

from typing import Iterable

from orion.core.schemas.substrate_mutation import MutationSignalV1
from orion.core.schemas.substrate_review_telemetry import GraphReviewTelemetryRecordV1


class MutationDetectors:
    """Convert runtime telemetry into typed mutation signals."""

    def from_review_telemetry(self, records: Iterable[GraphReviewTelemetryRecordV1]) -> list[MutationSignalV1]:
        signals: list[MutationSignalV1] = []
        for record in records:
            if not record.anchor_scope or not record.subject_ref or not record.target_zone:
                continue
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
            signals.append(
                MutationSignalV1(
                    event_kind=event_kind,
                    anchor_scope=record.anchor_scope,
                    subject_ref=record.subject_ref,
                    target_zone=record.target_zone,
                    target_surface=_target_surface_for_zone(record.target_zone),
                    strength=strength,
                    evidence_refs=[
                        f"telemetry:{record.telemetry_id}",
                        f"runtime_outcome:{record.execution_outcome}",
                    ],
                    source_ref=f"review-telemetry:{record.telemetry_id}",
                )
            )
        return signals


def _target_surface_for_zone(zone: str) -> str:
    if zone == "autonomy_graph":
        return "routing"
    if zone == "world_ontology":
        return "graph_consolidation"
    if zone == "self_relationship_graph":
        return "prompt_profile"
    return "recall"
