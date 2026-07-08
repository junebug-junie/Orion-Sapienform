"""Deterministic endogenous "spontaneous want" generator.

Reads Orion's continuous internal ``SelfStateV1`` stream and, in the ABSENCE of
external input, mints a ``TensionEventV1(origin="endogenous")`` when internal
dynamics cross an origination band.

Pure substrate math: no LLM, no bus, no I/O, no regex/text routing of self-state
content. Every extraction is defensive — ``observe`` and ``maybe_originate``
never raise; on malformed input they skip or return ``None``. The internal ring
buffer is bounded at ``cfg.window``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from orion.core.schemas.drives import ArtifactProvenance, TensionEventV1
from orion.schemas.self_state import SelfStateV1

# The only drives an endogenous tension may map to.
DRIVE_KEYS = ("coherence", "continuity", "capability", "relational", "predictive", "autonomy")


def _clamp01(x: float) -> float:
    """Clamp a numeric value into [0, 1]; non-numeric -> 0.0."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


@dataclass
class OriginationConfig:
    window: int = 8              # ring buffer size (CAP)
    threshold: float = 0.55      # ORIGINATION_THRESHOLD
    cooldown_sec: float = 900.0  # ORIGINATION_COOLDOWN_SEC
    mag_cap: float = 0.5         # ENDOGENOUS_MAG_CAP
    w_drift: float = 0.4
    w_dwell: float = 0.35
    w_agency: float = 0.25
    exogenous_floor: int = 0     # max exogenous tensions in window to still allow endogeny
    dwell_norm: float = 20.0
    unresolved_norm: float = 4.0


@dataclass
class _Snapshot:
    """Bounded summary of a single self-state pushed into the ring."""

    drift: float                 # mean |value| over dimension_trajectory, clamp01
    dwell_ticks: int
    unresolved: tuple            # tuple of unresolved-pressure strings (bounded copy)
    agency_readiness: float      # score of dimensions['agency_readiness'], clamp01
    overall_intensity: float     # clamp01


class OriginationEngine:
    def __init__(self, cfg: Optional[OriginationConfig] = None) -> None:
        self.cfg = cfg or OriginationConfig()
        window = self.cfg.window if self.cfg.window and self.cfg.window > 0 else 1
        self._ring: deque = deque(maxlen=window)
        self.last_fire_ts: Optional[datetime] = None
        self._last_signal: dict = {}

    # ---- observation ------------------------------------------------------

    def observe(self, self_state: SelfStateV1) -> None:
        """Push a bounded summary of this self-state into the ring (cap cfg.window).

        Never raises on malformed input — the offending snapshot is skipped.
        """
        try:
            snap = self._summarize(self_state)
        except Exception:
            return
        if snap is not None:
            self._ring.append(snap)

    def _summarize(self, self_state: SelfStateV1) -> Optional[_Snapshot]:
        # drift: mean |value| over dimension_trajectory dict
        drift = 0.0
        traj = getattr(self_state, "dimension_trajectory", None)
        if isinstance(traj, dict) and traj:
            vals = []
            for v in traj.values():
                try:
                    vals.append(abs(float(v)))
                except (TypeError, ValueError):
                    continue
            if vals:
                drift = sum(vals) / len(vals)
        drift = _clamp01(drift)

        # dwell ticks
        try:
            dwell_ticks = int(getattr(self_state, "attention_dwell_ticks", 0) or 0)
        except (TypeError, ValueError):
            dwell_ticks = 0

        # unresolved pressures (bounded copy of strings)
        unresolved_raw = getattr(self_state, "unresolved_pressures", None)
        if isinstance(unresolved_raw, (list, tuple)):
            # Local cap so the per-snapshot copy is bounded regardless of upstream.
            unresolved = tuple(str(p) for p in unresolved_raw[:16])
        else:
            unresolved = tuple()

        # agency readiness score (defensive attr OR dict access)
        agency_readiness = self._extract_agency(self_state)

        overall_intensity = _clamp01(getattr(self_state, "overall_intensity", 0.0))

        return _Snapshot(
            drift=drift,
            dwell_ticks=dwell_ticks,
            unresolved=unresolved,
            agency_readiness=agency_readiness,
            overall_intensity=overall_intensity,
        )

    @staticmethod
    def _extract_agency(self_state: SelfStateV1) -> float:
        dims = getattr(self_state, "dimensions", None)
        entry = None
        if isinstance(dims, dict):
            entry = dims.get("agency_readiness")
        else:
            try:
                entry = dims["agency_readiness"]  # type: ignore[index]
            except Exception:
                entry = None
        if entry is None:
            return 0.0
        # attribute access first, then dict access
        score = getattr(entry, "score", None)
        if score is None:
            try:
                score = entry["score"]  # type: ignore[index]
            except Exception:
                score = None
        return _clamp01(score if score is not None else 0.0)

    # ---- origination ------------------------------------------------------

    def maybe_originate(
        self,
        *,
        exogenous_tension_count: int,
        now: datetime,
        subject: str = "orion",
    ) -> Optional[TensionEventV1]:
        """Compute D,W,A -> P over the current window and fire iff:

        exogenous_tension_count <= cfg.exogenous_floor AND cooldown elapsed since
        last fire AND P >= cfg.threshold. Never raises; records ``last_signal``.
        """
        try:
            return self._maybe_originate(
                exogenous_tension_count=exogenous_tension_count, now=now, subject=subject
            )
        except Exception:
            # Absolute guarantee: never raise.
            self._last_signal = {"drift": 0.0, "dwell": 0.0, "agency": 0.0, "P": 0.0, "fired": False}
            return None

    def _maybe_originate(
        self, *, exogenous_tension_count: int, now: datetime, subject: str
    ) -> Optional[TensionEventV1]:
        cfg = self.cfg

        # Empty ring: nothing to originate from.
        if not self._ring:
            self._last_signal = {"drift": 0.0, "dwell": 0.0, "agency": 0.0, "P": 0.0, "fired": False}
            return None

        D = self._drift_signal()
        W = self._dwell_signal()
        A = self._agency_signal()
        P = _clamp01(cfg.w_drift * D + cfg.w_dwell * W + cfg.w_agency * A)

        # Gate 1: exogenous input present -> no endogeny.
        try:
            exo = int(exogenous_tension_count)
        except (TypeError, ValueError):
            exo = 1  # treat garbage as "input present" -> suppress
        exo_ok = exo <= cfg.exogenous_floor

        # Gate 2: cooldown elapsed.
        cooldown_ok = True
        if self.last_fire_ts is not None:
            try:
                elapsed = (now - self.last_fire_ts).total_seconds()
                cooldown_ok = elapsed >= cfg.cooldown_sec
            except Exception:
                cooldown_ok = False

        # Gate 3: threshold.
        threshold_ok = P >= cfg.threshold

        fired = exo_ok and cooldown_ok and threshold_ok
        self._last_signal = {"drift": D, "dwell": W, "agency": A, "P": P, "fired": fired}

        if not fired:
            return None

        drive = self._map_drive(D, W, A)
        magnitude = min(cfg.mag_cap, P)
        self.last_fire_ts = now

        return TensionEventV1(
            subject=subject,
            model_layer="self-model",
            entity_id="self:orion",
            kind="tension.endogenous.v1",
            magnitude=magnitude,
            drive_impacts={drive: 1.0},
            origin="endogenous",
            origination_signal={"drift": D, "dwell": W, "agency": A, "P": P},
            provenance=ArtifactProvenance(
                intake_channel="substrate.self_state.v1",
                evidence_summary=f"endogenous:{drive} P={P:.3f}",
            ),
        )

    # ---- sub-signals ------------------------------------------------------

    def _drift_signal(self) -> float:
        # mean over ring of each snapshot's drift (already per-snapshot mean |value|).
        if not self._ring:
            return 0.0
        total = sum(s.drift for s in self._ring)
        return _clamp01(total / len(self._ring))

    def _dwell_signal(self) -> float:
        # Uses the LATEST snapshot.
        latest = self._ring[-1]
        cfg = self.cfg
        dwell_norm = cfg.dwell_norm if cfg.dwell_norm else 1.0
        unresolved_norm = cfg.unresolved_norm if cfg.unresolved_norm else 1.0
        dwell_part = min(1.0, latest.dwell_ticks / dwell_norm)
        unresolved_part = min(1.0, len(latest.unresolved) / unresolved_norm)
        return _clamp01(0.5 * dwell_part + 0.5 * unresolved_part)

    def _agency_signal(self) -> float:
        # Uses the LATEST snapshot: readiness * (1 - overall_intensity).
        latest = self._ring[-1]
        return _clamp01(latest.agency_readiness * (1.0 - latest.overall_intensity))

    def _map_drive(self, D: float, W: float, A: float) -> str:
        latest = self._ring[-1]
        unresolved = latest.unresolved

        # 1. Override by top unresolved pressure.
        if "social_pressure" in unresolved:
            return "relational"
        if "continuity_pressure" in unresolved:
            return "continuity"

        # 2. Dominant weighted sub-signal.
        cfg = self.cfg
        drift_w = cfg.w_drift * D
        dwell_w = cfg.w_dwell * W
        agency_w = cfg.w_agency * A
        # drift-dominant -> coherence; dwell-dominant -> autonomy; agency-dominant -> capability
        if drift_w >= dwell_w and drift_w >= agency_w:
            return "coherence"
        if dwell_w >= agency_w:
            return "autonomy"
        return "capability"

    # ---- debug surface ----------------------------------------------------

    @property
    def last_signal(self) -> dict:
        """{'drift','dwell','agency','P','fired'} from the last maybe_originate call."""
        return dict(self._last_signal)
