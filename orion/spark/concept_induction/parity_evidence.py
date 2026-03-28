from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any


@dataclass
class ParityReadinessThresholds:
    min_comparisons: int = 50
    max_mismatch_rate: float = 0.05
    max_unavailable_rate: float = 0.02
    critical_mismatch_classes: tuple[str, ...] = (
        "profile_missing_on_graph",
        "profile_missing_on_local",
        "query_error",
    )


@dataclass
class ConsumerParityEvidence:
    consumer: str
    total_comparisons: int = 0
    exact_matches: int = 0
    mismatches: int = 0
    graph_unavailable_count: int = 0
    empty_on_local_only: int = 0
    empty_on_graph_only: int = 0
    mismatch_class_counts: Counter[str] = field(default_factory=Counter)
    recent_subjects: deque[str] = field(default_factory=lambda: deque(maxlen=20))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        mismatch_rate = (self.mismatches / self.total_comparisons) if self.total_comparisons else 0.0
        unavailable_rate = (
            self.graph_unavailable_count / self.total_comparisons if self.total_comparisons else 0.0
        )
        return {
            "consumer": self.consumer,
            "total_comparisons": self.total_comparisons,
            "exact_matches": self.exact_matches,
            "mismatches": self.mismatches,
            "graph_unavailable_count": self.graph_unavailable_count,
            "empty_on_local_only": self.empty_on_local_only,
            "empty_on_graph_only": self.empty_on_graph_only,
            "mismatch_rate": round(mismatch_rate, 6),
            "unavailable_rate": round(unavailable_rate, 6),
            "mismatch_class_counts": dict(self.mismatch_class_counts),
            "recent_subjects": list(self.recent_subjects),
            "last_updated": self.last_updated.isoformat(),
        }


class ParityEvidenceStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._by_consumer: dict[str, ConsumerParityEvidence] = {}
        self._thresholds = ParityReadinessThresholds()
        self._summary_interval = 25

    def configure(
        self,
        *,
        thresholds: ParityReadinessThresholds,
        summary_interval: int,
    ) -> None:
        with self._lock:
            self._thresholds = thresholds
            self._summary_interval = max(1, int(summary_interval))

    def record(
        self,
        *,
        consumer: str,
        subject_outcomes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        with self._lock:
            evidence = self._by_consumer.setdefault(consumer, ConsumerParityEvidence(consumer=consumer))
            for outcome in subject_outcomes:
                evidence.total_comparisons += 1
                subject = str(outcome.get("subject") or "")
                if subject:
                    evidence.recent_subjects.append(subject)
                if outcome.get("graph_unavailable"):
                    evidence.graph_unavailable_count += 1
                if outcome.get("empty_on_local_only"):
                    evidence.empty_on_local_only += 1
                if outcome.get("empty_on_graph_only"):
                    evidence.empty_on_graph_only += 1

                mismatch_classes = list(outcome.get("mismatch_classes") or [])
                if mismatch_classes:
                    evidence.mismatches += 1
                    evidence.mismatch_class_counts.update(mismatch_classes)
                else:
                    evidence.exact_matches += 1
                evidence.last_updated = datetime.now(timezone.utc)

            readiness = self._readiness_for_consumer_locked(consumer)
            summary = {
                "consumer": consumer,
                "evidence": evidence.to_dict(),
                "readiness": readiness,
                "should_emit_summary": evidence.total_comparisons % self._summary_interval == 0,
            }
            return summary

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            consumers = {name: evidence.to_dict() for name, evidence in self._by_consumer.items()}
            readiness = {name: self._readiness_for_consumer_locked(name) for name in self._by_consumer}
            return {
                "thresholds": {
                    "min_comparisons": self._thresholds.min_comparisons,
                    "max_mismatch_rate": self._thresholds.max_mismatch_rate,
                    "max_unavailable_rate": self._thresholds.max_unavailable_rate,
                    "critical_mismatch_classes": list(self._thresholds.critical_mismatch_classes),
                },
                "consumers": consumers,
                "readiness": readiness,
            }

    def _readiness_for_consumer_locked(self, consumer: str) -> dict[str, Any]:
        evidence = self._by_consumer.get(consumer)
        if evidence is None:
            return {
                "consumer": consumer,
                "ready": False,
                "reason": "no_evidence",
                "mismatch_rate": 0.0,
                "unavailable_rate": 0.0,
                "critical_classes_present": [],
            }

        mismatch_rate = (evidence.mismatches / evidence.total_comparisons) if evidence.total_comparisons else 0.0
        unavailable_rate = (
            evidence.graph_unavailable_count / evidence.total_comparisons if evidence.total_comparisons else 0.0
        )

        critical_present = [
            cls for cls in self._thresholds.critical_mismatch_classes if evidence.mismatch_class_counts.get(cls, 0) > 0
        ]

        reason = "ready"
        ready = True
        if evidence.total_comparisons < self._thresholds.min_comparisons:
            ready = False
            reason = "insufficient_samples"
        elif unavailable_rate > self._thresholds.max_unavailable_rate:
            ready = False
            reason = "graph_unavailable_rate_high"
        elif mismatch_rate > self._thresholds.max_mismatch_rate:
            ready = False
            reason = "mismatch_rate_high"
        elif critical_present:
            ready = False
            reason = "critical_mismatch_classes_present"

        return {
            "consumer": consumer,
            "ready": ready,
            "reason": reason,
            "mismatch_rate": round(mismatch_rate, 6),
            "unavailable_rate": round(unavailable_rate, 6),
            "critical_classes_present": critical_present,
        }


_STORE = ParityEvidenceStore()


def configure_parity_evidence_store(
    *,
    thresholds: ParityReadinessThresholds,
    summary_interval: int,
) -> None:
    _STORE.configure(thresholds=thresholds, summary_interval=summary_interval)


def record_parity_evidence(*, consumer: str, subject_outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    return _STORE.record(consumer=consumer, subject_outcomes=subject_outcomes)


def get_parity_evidence_snapshot() -> dict[str, Any]:
    return _STORE.snapshot()


def reset_parity_evidence_store() -> None:
    global _STORE
    _STORE = ParityEvidenceStore()
