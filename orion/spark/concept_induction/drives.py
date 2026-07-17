"""Read orion/autonomy/drives_and_autonomy_retrospective.md (sec5b-sec5e) and this
directory's CLAUDE.md before changing DriveEngine.update()'s aggregation math -- the
sum-then-clamp-once collapse bug this module used to have was found and re-derived
independently more than once because that write-up wasn't checked first."""

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
    # When True (ORION_DRIVE_LEAKY_MATH_ENABLED), pressure is a wall-clock leaky
    # integrator that rests at zero and is cadence-invariant. When False, the
    # legacy soft-saturate path is used (kept for rollback). The legacy path has
    # a non-zero fixed point (~0.731 at gain=1.8) that pins every drive under
    # frequent ticks — a cadence artifact, not cognition.
    leaky_math_enabled: bool = True


class DriveEngine:
    def __init__(self, cfg: DriveMathConfig | None = None) -> None:
        self.cfg = cfg or DriveMathConfig()

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _clamp_signed(value: float) -> float:
        """Clamp to [-1, 1] -- same shape as _clamp01 but preserves sign, for
        drive_impacts weights that carry relief (negative) as well as growth
        (positive). Growth-only producers never emit a negative weight, so
        this is a strict superset of _clamp01's behavior for them.
        """
        return max(-1.0, min(1.0, float(value)))

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

        p = {drive: prev_p[drive] * decay for drive in DRIVE_KEYS}
        if self.cfg.leaky_math_enabled:
            # Leaky integrator, applied SEQUENTIALLY once per tension event
            # (not summed across the whole fold batch and clamped once at the
            # end -- that sum-then-clamp design let a batch of many same-sign
            # tensions collapse every drive that exceeded the clamp bound to
            # the exact same value, independent of starting pressure or how
            # much it exceeded by; see docs/superpowers/specs/2026-07-17-
            # drive-engine-fold-batch-clamp-collapse-fix-design.md). Each
            # event's own impulse pushes pressure a fraction of the remaining
            # headroom (1 - p) toward growth, or a fraction of the existing
            # pressure (p) toward relief -- no fixed point: with impulse=0,
            # pressure=base < prev.
            #
            # Unlike the old sum-then-clamp math (commutative -- batch order
            # never mattered), sequential application makes the exact result
            # depend on event order. The caller buffers tensions in arrival
            # order (bus_worker.py's `pending.extend(...)`), which is stable
            # for a normal run but not guaranteed order-identical across bus
            # redelivery/retries (found by code review). Sort by (ts,
            # artifact_id) -- both always present on every TensionEventV1 --
            # so the result is fully order-independent regardless of delivery
            # order, not just deterministic for one fixed arrival order.
            for event in sorted(tensions, key=lambda e: (e.ts, e.artifact_id)):
                mag = self._clamp01(event.magnitude)
                for drive, weight in sorted(event.drive_impacts.items()):
                    if drive not in p:
                        continue
                    # weight is signed: positive = growth (existing
                    # producers), negative = relief (e.g. a satisfaction
                    # tension on a successfully completed action). magnitude
                    # itself stays non-negative (schema-enforced) --
                    # direction lives in weight.
                    #
                    # The outer clamp_signed() is a defensive no-op, not load-
                    # bearing: mag is already in [0,1] and clamp_signed(weight)
                    # is already in [-1,1], so their product is mathematically
                    # already within [-1,1] (found by code review, kept as a
                    # cheap safety net rather than trusting that invariant to
                    # never be violated by a future producer).
                    impulse = self._clamp_signed(mag * self._clamp_signed(weight))
                    if impulse >= 0.0:
                        p[drive] = self._clamp01(p[drive] + impulse * (1.0 - p[drive]))
                    else:
                        # Relief is the mirror image of growth: diminishing
                        # effect as pressure approaches its OTHER bound (0
                        # instead of 1), scaled by how much pressure is
                        # actually there to relieve (p[drive], not
                        # 1-p[drive]) -- at p[drive]=0 a relief impulse is
                        # naturally a no-op without depending on the outer
                        # clamp.
                        p[drive] = self._clamp01(p[drive] + impulse * p[drive])
            pressures: Dict[str, float] = p
        else:
            # Legacy soft_saturate path: UNCHANGED, still sum-then-saturate.
            # Not the live path (default leaky_math_enabled=True), kept only
            # as a documented rollback -- out of scope for the fold-batch
            # collapse fix above.
            impact_sum = {k: 0.0 for k in DRIVE_KEYS}
            for event in tensions:
                mag = self._clamp01(event.magnitude)
                for drive, weight in sorted(event.drive_impacts.items()):
                    if drive not in impact_sum:
                        continue
                    impact_sum[drive] += mag * self._clamp_signed(weight)
            pressures = {}
            for drive in DRIVE_KEYS:
                raw = p[drive] + impact_sum[drive]
                pressures[drive] = self._soft_saturate(raw)

        activations: Dict[str, bool] = {}
        for drive in DRIVE_KEYS:
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
