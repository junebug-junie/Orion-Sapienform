from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from orion.schemas.telemetry.biometrics import BiometricsSummaryV1
from orion.schemas.telemetry.cabinet_ambient_spike import CabinetAmbientSpikeV1


@dataclass
class AmbientSpikeDetectorConfig:
    activity_threshold: float = 0.30
    consecutive_ticks: int = 2
    cooldown_sec: float = 300.0


class AmbientSpikeDetector:
    """Emit at most one spike event per cooldown when activity stays elevated.

    After a spike fires, `_consecutive` resets. During cooldown, consecutive
    ticks still accumulate so sustained elevated activity can re-alert
    immediately once cooldown expires (debounce applies to cold-start only).
    """

    def __init__(self, cfg: AmbientSpikeDetectorConfig) -> None:
        self.cfg = cfg
        self._consecutive = 0
        self._last_spike_at: Optional[datetime] = None

    def observe(
        self,
        *,
        node: str,
        timestamp: datetime,
        summary: BiometricsSummaryV1,
        source_service: str,
        source_node: Optional[str] = None,
    ) -> Optional[CabinetAmbientSpikeV1]:
        activity = summary.pressures.get("cabinet_ambient_audio_activity")
        if activity is None:
            self._consecutive = 0
            return None

        if activity >= self.cfg.activity_threshold:
            self._consecutive += 1
        else:
            self._consecutive = 0
            return None

        if self._consecutive < self.cfg.consecutive_ticks:
            return None

        if self._in_cooldown(timestamp):
            return None

        measurements = summary.measurements or {}
        rms = measurements.get("cabinet_ambient_rms")
        if rms is None:
            return None

        peak = measurements.get("cabinet_ambient_peak")
        ts = timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=timezone.utc)

        self._last_spike_at = ts
        self._consecutive = 0

        return CabinetAmbientSpikeV1(
            spike_id=str(uuid4()),
            node=node,
            timestamp=ts,
            activity=float(activity),
            rms=float(rms),
            peak=float(peak) if peak is not None else None,
            activity_threshold=self.cfg.activity_threshold,
            consecutive_ticks=self.cfg.consecutive_ticks,
            source_service=source_service,
            source_node=source_node,
        )

    def _in_cooldown(self, timestamp: datetime) -> bool:
        if self._last_spike_at is None:
            return False
        ts = timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=timezone.utc)
        elapsed = (ts - self._last_spike_at).total_seconds()
        return elapsed < self.cfg.cooldown_sec
