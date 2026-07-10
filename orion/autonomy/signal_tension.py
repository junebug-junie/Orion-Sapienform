"""Adapt real bus events into deviation-driven TensionEventV1 (spec §4).

Four entry points, all degrade to ``None`` (never raise) on absent/garbage
input:

* ``signal_to_tension`` — an ``OrionSignalV1`` through the deviation gate + map.
  Stub signals are dropped; each mapped dimension impulses only when it deviates
  from its own learned baseline. The 55/s ``scene_state`` flood is unmapped and
  steady, so it mints nothing.
* ``failure_to_tension`` — a failure event maps *directly* (no EWMA): a failure
  is itself the deviation, so gating it behind a warm-up would swallow the first
  real errors. Severity sizes the magnitude; the map supplies drive weights.
* ``equilibrium_to_tension`` — edge-triggered health transition (ok -> degraded
  mints once; degraded -> degraded mints nothing).
* ``chat_evidence_to_tension`` — a pressure-eligible ``AutonomyEvidenceRefV1``
  maps *directly* (no EWMA / DeviationGate), same family as ``failure_to_tension``.

The per-drive contribution is encoded as ``magnitude * drive_impacts[drive]`` so
the existing ``compute_tick_attribution`` and ``DriveEngine.update`` (which both
multiply magnitude by the impact weight) reproduce the intended impulse.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Mapping, Optional

from orion.autonomy.deviation_gate import DeviationGate
from orion.autonomy.signal_drive_map import SignalDriveMap
from orion.core.schemas.drives import ArtifactProvenance, TensionEventV1
from orion.signals.models import OrionSignalV1
from orion.signals.stub_detection import is_stub_signal
from orion.spark.concept_induction.drives import DRIVE_KEYS

if TYPE_CHECKING:
    from orion.autonomy.models import AutonomyEvidenceRefV1

SIGNAL_TENSION_KIND = "tension.signal.v1"
FAILURE_TENSION_KIND = "tension.failure.v1"
HEALTH_TENSION_KIND = "tension.health.v1"
CHAT_EVIDENCE_TENSION_KIND = "tension.chat_evidence.v1"


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _build_tension(
    *,
    kind: str,
    raw_by_drive: Mapping[str, float],
    channel: str,
    correlation_id: Optional[str],
    summary: str,
    related: Optional[list[str]] = None,
) -> Optional[TensionEventV1]:
    raw = {d: _clamp01(v) for d, v in raw_by_drive.items() if d in DRIVE_KEYS and v > 0.0}
    if not raw:
        return None
    magnitude = _clamp01(max(raw.values()))
    if magnitude <= 0.0:
        return None
    # Encode so magnitude * impact == the intended per-drive contribution.
    drive_impacts: Dict[str, float] = {d: _clamp01(v / magnitude) for d, v in raw.items()}
    return TensionEventV1(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        kind=kind,
        magnitude=magnitude,
        drive_impacts=drive_impacts,
        provenance=ArtifactProvenance(
            intake_channel=channel,
            correlation_id=correlation_id,
            evidence_summary=summary[:240],
        ),
        related_nodes=related or [],
    )


def signal_to_tension(
    sig: OrionSignalV1,
    gate: DeviationGate,
    sdm: SignalDriveMap,
    *,
    channel: str = "orion:signals",
) -> Optional[TensionEventV1]:
    """OrionSignalV1 -> deviation-gated tension, or None."""
    try:
        if is_stub_signal(sig):
            return None
        dims = sig.dimensions or {}
        if not dims:
            return None
        confidence = float(dims.get("confidence", 1.0))
        raw_by_drive: Dict[str, float] = {}
        for dim_name, value in dims.items():
            rule = sdm.match(sig.signal_kind, dim_name)
            if rule is None:
                continue
            impulse = gate.observe(
                sig.signal_kind, dim_name, value, confidence=confidence, worse=rule.worse
            )
            if impulse <= 0.0:
                continue
            for drive, weight in rule.drives.items():
                raw_by_drive[drive] = raw_by_drive.get(drive, 0.0) + impulse * weight
        return _build_tension(
            kind=SIGNAL_TENSION_KIND,
            raw_by_drive=raw_by_drive,
            channel=channel,
            correlation_id=getattr(sig, "signal_id", None),
            summary=f"{sig.signal_kind} deviation",
            related=[f"organ:{getattr(sig, 'organ_id', 'unknown')}"],
        )
    except Exception:
        return None


def failure_to_tension(
    *,
    severity: float,
    sdm: SignalDriveMap,
    channel: str,
    correlation_id: Optional[str] = None,
    summary: str = "failure_event",
) -> Optional[TensionEventV1]:
    """A failure maps directly via the ``failure_event.severity`` rule (no EWMA:
    a failure is itself the deviation)."""
    try:
        rule = sdm.match("failure_event", "severity")
        if rule is None:
            return None
        sev = _clamp01(severity)
        if sev <= 0.0:
            return None
        raw_by_drive = {d: sev * w for d, w in rule.drives.items()}
        return _build_tension(
            kind=FAILURE_TENSION_KIND,
            raw_by_drive=raw_by_drive,
            channel=channel,
            correlation_id=correlation_id,
            summary=summary,
        )
    except Exception:
        return None


class SignalTensionSource:
    """Stateful coordinator: holds the deviation gate, structural map, and rate
    limiter so their state persists across ticks. The worker calls the ``from_*``
    methods and merges the returned (already rate-limited) tensions into the tick.

    Every method degrades to ``[]`` (never raises) so a malformed event can never
    stall the drive loop.
    """

    def __init__(
        self,
        *,
        gate: DeviationGate,
        sdm: SignalDriveMap,
        ratelimiter,
    ) -> None:
        self._gate = gate
        self._sdm = sdm
        self._rl = ratelimiter

    def _bounded(self, tension: Optional[TensionEventV1], now: float) -> list[TensionEventV1]:
        if tension is None:
            return []
        return self._rl.bounded([tension], now=now)

    def from_signal(self, sig: OrionSignalV1, *, now: float, channel: str = "orion:signals") -> list[TensionEventV1]:
        return self._bounded(signal_to_tension(sig, self._gate, self._sdm, channel=channel), now)

    def from_failure(self, *, severity: float, now: float, channel: str,
                     correlation_id: Optional[str] = None, summary: str = "failure_event") -> list[TensionEventV1]:
        return self._bounded(
            failure_to_tension(severity=severity, sdm=self._sdm, channel=channel,
                               correlation_id=correlation_id, summary=summary),
            now,
        )

    def from_equilibrium(self, *, healthy: bool, prev_healthy: Optional[bool], now: float,
                         correlation_id: Optional[str] = None) -> list[TensionEventV1]:
        return self._bounded(
            equilibrium_to_tension(healthy=healthy, prev_healthy=prev_healthy, sdm=self._sdm,
                                   correlation_id=correlation_id),
            now,
        )


def equilibrium_to_tension(
    *,
    healthy: bool,
    prev_healthy: Optional[bool],
    sdm: SignalDriveMap,
    channel: str = "orion:equilibrium:snapshot",
    correlation_id: Optional[str] = None,
    severity: float = 0.6,
) -> Optional[TensionEventV1]:
    """Edge-triggered: mint once on ok -> degraded, nothing while it stays
    degraded (or recovers). Uses the ``mesh_health.level`` structural weights."""
    try:
        # Only the falling edge into degraded is a new tension.
        if healthy or prev_healthy is False:
            return None
        rule = sdm.match("mesh_health", "level")
        if rule is None:
            return None
        raw_by_drive = {d: _clamp01(severity) * w for d, w in rule.drives.items()}
        return _build_tension(
            kind=HEALTH_TENSION_KIND,
            raw_by_drive=raw_by_drive,
            channel=channel,
            correlation_id=correlation_id,
            summary="equilibrium degraded",
        )
    except Exception:
        return None


def chat_evidence_to_tension(
    ev: "AutonomyEvidenceRefV1",
    sdm: SignalDriveMap,
    *,
    channel: str = "orion:cortex_exec:chat_stance",
) -> Optional[TensionEventV1]:
    """Map a pressure-eligible AutonomyEvidenceRefV1 directly (no EWMA).

    Requires signal_kind + dimension + value. Unmapped / missing → None.
    Never raises.
    """
    try:
        signal_kind = getattr(ev, "signal_kind", None)
        dimension = getattr(ev, "dimension", None)
        value = getattr(ev, "value", None)
        if not signal_kind or not dimension or value is None:
            return None
        rule = sdm.match(str(signal_kind), str(dimension))
        if rule is None:
            return None
        v = _clamp01(float(value))
        if v <= 0.0:
            return None
        raw_by_drive = {d: v * w for d, w in rule.drives.items()}
        summary = (getattr(ev, "summary", None) or f"{signal_kind}.{dimension}")[:240]
        return _build_tension(
            kind=CHAT_EVIDENCE_TENSION_KIND,
            raw_by_drive=raw_by_drive,
            channel=channel,
            correlation_id=getattr(ev, "evidence_id", None),
            summary=str(summary),
        )
    except Exception:
        return None
