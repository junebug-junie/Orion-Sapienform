from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from orion.core.schemas.substrate_mutation import MutationPressureV1, MutationSignalV1


@dataclass(frozen=True)
class PressurePolicy:
    activation_threshold: float = 3.0
    decay_factor: float = 0.92
    cooldown_seconds: int = 600


_RECALL_SURFACES_FOR_SNAPSHOT = frozenset(
    {
        "recall",
        "recall_strategy_profile",
        "recall_anchor_policy",
        "recall_page_index_profile",
        "recall_graph_expansion_policy",
    }
)
_RECALL_HISTORY_MAX = 8


class PressureAccumulator:
    def __init__(self, *, policy: PressurePolicy | None = None) -> None:
        self._policy = policy or PressurePolicy()

    def apply(self, *, current: MutationPressureV1 | None, signal: MutationSignalV1, now: datetime | None = None) -> MutationPressureV1:
        t = now or datetime.now(timezone.utc)
        existing_score = current.pressure_score if current else 0.0
        next_score = max(0.0, (existing_score * self._policy.decay_factor) + (signal.strength * 5.0))
        source_signal_ids = list(current.source_signal_ids) if current else []
        source_signal_ids.append(signal.signal_id)
        evidence_refs = list(current.evidence_refs) if current else []
        for ref in signal.evidence_refs:
            if ref not in evidence_refs:
                evidence_refs.append(ref)
        cooldown_until = current.cooldown_until if current and (current.cooldown_until is None or current.cooldown_until > t) else None
        snapshot: dict[str, object] = dict(current.recall_evidence_snapshot) if current else {}
        history: list[dict[str, Any]] = list(current.recall_evidence_history) if current else []
        if signal.target_surface in _RECALL_SURFACES_FOR_SNAPSHOT and str(signal.event_kind or "").startswith("pressure_event:"):
            meta = signal.metadata or {}
            if isinstance(meta.get("recall_compare"), dict):
                snapshot["recall_compare"] = meta["recall_compare"]
            if isinstance(meta.get("anchor_plan"), dict):
                snapshot["anchor_plan"] = meta["anchor_plan"]
            if isinstance(meta.get("selected_evidence_cards"), list):
                snapshot["selected_evidence_cards"] = meta["selected_evidence_cards"][:12]
            fc = meta.get("failure_category")
            if isinstance(fc, str) and fc.strip():
                snapshot["failure_category"] = fc.strip()
            kind = str(meta.get("recall_evidence_kind") or "live_shadow").strip() or "live_shadow"
            hist_entry: dict[str, Any] = {
                "recorded_at": t.isoformat(),
                "signal_id": signal.signal_id,
                "failure_category": str(meta.get("failure_category") or "").strip(),
                "recall_evidence_kind": kind,
            }
            if isinstance(meta.get("recall_compare"), dict):
                hist_entry["recall_compare"] = dict(meta["recall_compare"])
            if isinstance(meta.get("anchor_plan"), dict):
                hist_entry["anchor_plan"] = dict(meta["anchor_plan"])
            if isinstance(meta.get("selected_evidence_cards"), list):
                hist_entry["selected_evidence_cards"] = list(meta["selected_evidence_cards"][:8])
            if isinstance(meta.get("recall_eval_case"), dict):
                hist_entry["recall_eval_case"] = dict(meta["recall_eval_case"])
            if meta.get("suite_run_id") is not None:
                hist_entry["suite_run_id"] = meta.get("suite_run_id")
            history.append(hist_entry)
            history = history[-_RECALL_HISTORY_MAX:]
        return MutationPressureV1(
            pressure_id=current.pressure_id if current else f"substrate-mutation-pressure-{signal.signal_id}",
            anchor_scope=signal.anchor_scope,
            subject_ref=signal.subject_ref,
            target_surface=signal.target_surface,
            target_zone=signal.target_zone,
            pressure_kind=signal.event_kind,
            pressure_score=min(100.0, next_score),
            evidence_refs=evidence_refs[:64],
            source_signal_ids=source_signal_ids[-64:],
            cooldown_until=cooldown_until,
            updated_at=t,
            recall_evidence_snapshot=snapshot,
            recall_evidence_history=history,
        )

    def ready_for_proposal(self, pressure: MutationPressureV1, *, now: datetime | None = None) -> bool:
        t = now or datetime.now(timezone.utc)
        if pressure.pressure_score < self._policy.activation_threshold:
            return False
        if pressure.cooldown_until and pressure.cooldown_until > t:
            return False
        return True

    def mark_proposal_emitted(self, pressure: MutationPressureV1, *, now: datetime | None = None) -> MutationPressureV1:
        t = now or datetime.now(timezone.utc)
        return pressure.model_copy(update={"cooldown_until": t + timedelta(seconds=self._policy.cooldown_seconds), "updated_at": t})
