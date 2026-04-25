from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from orion.core.schemas.substrate_mutation import MutationPressureV1, MutationSignalV1


@dataclass(frozen=True)
class PressurePolicy:
    activation_threshold: float = 3.0
    decay_factor: float = 0.92
    cooldown_seconds: int = 600


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
        cooldown_until = current.cooldown_until if current else None
        if next_score >= self._policy.activation_threshold:
            cooldown_until = t + timedelta(seconds=self._policy.cooldown_seconds)
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
        )

    def ready_for_proposal(self, pressure: MutationPressureV1, *, now: datetime | None = None) -> bool:
        t = now or datetime.now(timezone.utc)
        if pressure.pressure_score < self._policy.activation_threshold:
            return False
        if pressure.cooldown_until and pressure.cooldown_until > t:
            return False
        return True
