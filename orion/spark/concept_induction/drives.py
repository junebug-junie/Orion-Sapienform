from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable

from orion.core.schemas.drives import DriveStateV1, TensionEventV1

DRIVE_KEYS = ("coherence", "continuity", "capability", "relational", "predictive", "autonomy")


@dataclass
class DriveMathConfig:
    decay_tau_sec: float = 1800.0
    saturation_gain: float = 1.8
    activate_threshold: float = 0.62
    deactivate_threshold: float = 0.42


class DriveEngine:
    def __init__(self, cfg: DriveMathConfig | None = None) -> None:
        self.cfg = cfg or DriveMathConfig()

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _decay_factor(self, elapsed_sec: float) -> float:
        if elapsed_sec <= 0:
            return 1.0
        return math.exp(-elapsed_sec / max(1.0, self.cfg.decay_tau_sec))

    def _soft_saturate(self, value: float) -> float:
        v = self._clamp01(value)
        return self._clamp01(1.0 - math.exp(-self.cfg.saturation_gain * v))

    def update(
        self,
        *,
        previous_pressures: Dict[str, float] | None,
        previous_activations: Dict[str, bool] | None,
        tensions: Iterable[TensionEventV1],
        now: datetime,
        previous_ts: datetime | None,
    ) -> tuple[Dict[str, float], Dict[str, bool]]:
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        prev_p = {k: self._clamp01((previous_pressures or {}).get(k, 0.0)) for k in DRIVE_KEYS}
        prev_a = {k: bool((previous_activations or {}).get(k, False)) for k in DRIVE_KEYS}

        elapsed = (now - previous_ts).total_seconds() if previous_ts else 0.0
        decay = self._decay_factor(elapsed)

        impact_sum = {k: 0.0 for k in DRIVE_KEYS}
        for event in tensions:
            mag = self._clamp01(event.magnitude)
            for drive, weight in sorted(event.drive_impacts.items()):
                if drive not in impact_sum:
                    continue
                impact_sum[drive] += mag * self._clamp01(weight)

        pressures: Dict[str, float] = {}
        activations: Dict[str, bool] = {}
        for drive in DRIVE_KEYS:
            raw = prev_p[drive] * decay + impact_sum[drive]
            pressures[drive] = self._soft_saturate(raw)

            if prev_a[drive]:
                activations[drive] = pressures[drive] >= self.cfg.deactivate_threshold
            else:
                activations[drive] = pressures[drive] >= self.cfg.activate_threshold

        return pressures, activations


def drive_state_from_values(
    *,
    subject: str,
    model_layer: str,
    entity_id: str,
    ts: datetime,
    pressures: Dict[str, float],
    activations: Dict[str, bool],
    confidence: float,
    correlation_id: str | None,
    trace_id: str | None,
    turn_id: str | None,
    provenance,
    related_nodes,
) -> DriveStateV1:
    return DriveStateV1(
        subject=subject,
        model_layer=model_layer,
        entity_id=entity_id,
        kind="memory.drives.state.v1",
        ts=ts,
        confidence=confidence,
        correlation_id=correlation_id,
        trace_id=trace_id,
        turn_id=turn_id,
        provenance=provenance,
        related_nodes=related_nodes,
        pressures={k: float(max(0.0, min(1.0, pressures.get(k, 0.0)))) for k in DRIVE_KEYS},
        activations={k: bool(activations.get(k, False)) for k in DRIVE_KEYS},
        updated_at=ts,
    )
