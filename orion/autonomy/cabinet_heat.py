"""Cabinet heat: one source column, two numbers, never the same number twice.

Spec: docs/superpowers/specs/2026-09-29-attend-to-act-loop-design.md, "Amendment
2026-09-29" -> "Signal -- reuse, do not duplicate".

Source: athena's ``orion_biometrics_summary.measurements->>'cabinet_temp_c'``
(``orion/telemetry/cabinet_sensors.py`` -> biometrics pipeline, ~30 s cadence) --
the same rows the hardware-watch reflex reads.

1. ``cabinet_heat_pressure`` -- the OUTCOME (level). clamp01((t - 28.0) / (32.0 - 28.0)):
   0 at the thermal gate's elevated RE-ARM point (28.0 C), 0.375 at the elevated
   TRIP (29.5 C), 1.0 at hot (32.0 C). Live it reads 0 about 11 % of the week, so
   it can return to a genuine calm. Scored t0 -> t0 + 20 min on minute means.

2. ``cabinet_warming_error`` -- the ATTENTION SIGNAL (trigger). Non-zero only while
   the cabinet is elevated (thermal-gate hysteresis, not hot) AND rising by at least
   ``rise_threshold_c`` within 15 minutes, using the SHARED
   ``orion.hardware_watch.rules.cabinet_rise_c`` (the reflex's own definition of
   "rising"). Value = clamp01(rise_c / 1.0): 0.5 at the learned action's 0.5 C
   threshold, 1.0 at the reflex's 1.0 C threshold. Written as the
   ``prediction_error`` of ``node:substrate.cabinet`` so the EXISTING substrate
   dynamics engine (prediction_error -> dynamic_pressure) admits it to the
   workspace competition. Elevated alone is a common daily state (34 % of
   readings 09-29..10-06; the ~65 % figure written here on 09-29 is stale); this
   signal is zero then, so the cabinet cannot become a permanent winner.

3. The reflex verdict (thermal controller v2, docs/superpowers/specs/
   2026-10-06-thermal-controller-redesign-design.md D1/D2). ONE owner for "how hot
   is the cabinet" and for what a missing reading means:
   - a reading up to ``grace_sec`` old (default 300 s, the gate's own staleness
     bound) holds the last state; one failed query is the caller's to bridge by
     re-reading its last good points (hardware-watch does);
   - past grace the state is ``unknown``. For protection it counts as
     ``elevated`` (``effective_state``) -- ``hot`` only when the AC ALSO reads low
     (``ac_low=True``): "sensor dead and AC looks dead" is the one case where we
     assume the worst. AC silent too (monitoring outage) is not "AC low";
   - ``critical`` (>= 34 C, re-arm 33 C, ``thermal_gate.DEFAULT_CRITICAL_C``) is the
     reflex's line. ``reflex`` names the shed board reason hardware-watch asserts:
     ``cabinet_hot`` (critical, or unknown + AC low), ``cabinet_unknown`` (unknown),
     else None.

Theory anchor (why a rise is a prediction error): with the AC healthy the cabinet's
heat balance predicts a steady temperature; the live 15-minute change has SD
0.42 C with nothing done, so a 0.5 C rise is ~1.2 sigma above that prediction
and 1.0 C ~2.4 sigma. The threshold is a tunable knob, recorded on every proposal.

Pure arithmetic, no I/O, no clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence

from orion.autonomy.thermal_gate import (
    DEFAULT_CRITICAL_C,
    DEFAULT_CRITICAL_REARM_C,
    DEFAULT_ELEVATED_C,
    DEFAULT_ELEVATED_REARM_C,
    DEFAULT_HOT_C,
    DEFAULT_MAX_READING_AGE_SEC,
    ThermalState,
    thermal_state,
)
from orion.hardware_watch.rules import TempPoint, cabinet_rise_c

CABINET_HEAT_SIGNAL = "cabinet_heat_pressure"
CABINET_NODE_ID = "node:substrate.cabinet"
CABINET_TEMP_KEY = "cabinet_temp_c"
CABINET_NODE = "athena"

# The learned action's rise threshold (design: tunable knob, ~1.2x the 15-min noise
# SD). The reflex uses 1.0 C via the same function.
DEFAULT_RISE_THRESHOLD_C = 0.5
RISE_WINDOW_SEC = 900.0
# A rise of this size reads as full-strength warming (the reflex's own threshold).
FULL_SCALE_RISE_C = 1.0
# Minute-mean half width for before/after readings.
MINUTE_SEC = 60.0
# D1: how long the last reading's state holds before it becomes unknown.
DEFAULT_READING_GRACE_SEC = DEFAULT_MAX_READING_AGE_SEC
# D5: the cooling rule's "projected to reach hot within" window.
DEFAULT_LOOKAHEAD_MIN = 20.0

REFLEX_CABINET_HOT = "cabinet_hot"
REFLEX_CABINET_UNKNOWN = "cabinet_unknown"


def cabinet_heat_pressure(temp_c: float | None) -> float | None:
    """Level: 0 at re-arm 28.0, 0.375 at trip 29.5, 1.0 at hot 32.0. None for no reading."""
    if temp_c is None or not math.isfinite(temp_c):
        return None
    span = DEFAULT_HOT_C - DEFAULT_ELEVATED_REARM_C
    return max(0.0, min(1.0, (float(temp_c) - DEFAULT_ELEVATED_REARM_C) / span))


def hysteretic_thermal_state(
    points: Sequence[TempPoint], now: datetime, *, max_age_sec: float = DEFAULT_MAX_READING_AGE_SEC,
    initial_state: ThermalState = "normal",
) -> tuple[ThermalState, float | None, float | None]:
    """Fold the thermal gate over the readings (ascending) so hysteresis is honoured.

    Returns (state, newest_temp_c, newest_age_sec). A missing or stale newest reading is
    ``unknown``. ``initial_state`` seeds the fold (a caller that keeps its last verdict passes it,
    so a trip older than the window it re-reads still holds until the re-arm point)."""
    state: ThermalState = initial_state if initial_state != "unknown" else "normal"
    finite = [p for p in points if math.isfinite(p.value)]
    for p in finite:
        state = thermal_state(temp_c=p.value, age_sec=0.0, previous_state=state).state
    if not finite:
        return "unknown", None, None
    newest = finite[-1]
    age = (now - newest.ts).total_seconds()
    verdict = thermal_state(temp_c=newest.value, age_sec=age, previous_state=state, max_age_sec=max_age_sec)
    return verdict.state, newest.value, age


@dataclass(frozen=True)
class CabinetHeatReading:
    temp_c: float | None
    age_sec: float | None
    thermal_state: ThermalState
    rise_c: float | None
    rise_threshold_c: float
    pressure: float | None
    warming_error: float
    # --- thermal controller v2 (D1) ---
    minutes_to_hot: float | None = None   # linear projection from the 15-min rise; None if not rising
    critical: bool = False                # >= 34 C, held until below 33 C
    ac_low: bool | None = None            # what the caller said about the AC (None = not asked / unknown)
    grace_sec: float = DEFAULT_READING_GRACE_SEC

    @property
    def elevated_and_rising(self) -> bool:
        return self.warming_error > 0.0

    @property
    def reading_age_sec(self) -> float | None:
        """Seconds since the newest finite reading (None: there is none)."""
        return self.age_sec

    @property
    def effective_state(self) -> ThermalState:
        """The state protection acts on: unknown counts as elevated, or hot when the AC is low too."""
        if self.thermal_state != "unknown":
            return self.thermal_state
        return "hot" if self.ac_low else "elevated"

    @property
    def reflex(self) -> str | None:
        """The shed board reason hardware-watch's reflex asserts now (D2), or None."""
        if self.thermal_state == "unknown":
            return REFLEX_CABINET_HOT if self.ac_low else REFLEX_CABINET_UNKNOWN
        return REFLEX_CABINET_HOT if self.critical else None

    def as_dict(self) -> dict:
        return {
            "temp_c": self.temp_c,
            "age_sec": None if self.age_sec is None else round(self.age_sec, 1),
            "thermal_state": self.thermal_state,
            "effective_state": self.effective_state,
            "rise_c": self.rise_c,
            "rise_threshold_c": self.rise_threshold_c,
            "rise_window_sec": RISE_WINDOW_SEC,
            CABINET_HEAT_SIGNAL: None if self.pressure is None else round(self.pressure, 4),
            "warming_error": round(self.warming_error, 4),
            "minutes_to_hot": None if self.minutes_to_hot is None else round(self.minutes_to_hot, 1),
            "critical": self.critical,
            "ac_low": self.ac_low,
            "reflex": self.reflex,
        }


def cabinet_warming_error(thermal: ThermalState, rise_c: float | None, rise_threshold_c: float) -> float:
    """Elevated (not hot, not unknown) AND rising >= threshold -> clamp01(rise / 1.0); else 0."""
    if thermal != "elevated" or rise_c is None or rise_c < rise_threshold_c:
        return 0.0
    return max(0.0, min(1.0, rise_c / FULL_SCALE_RISE_C))


def minutes_to_hot(temp_c: float | None, rise_c: float | None, *, hot_c: float = DEFAULT_HOT_C,
                   window_sec: float = RISE_WINDOW_SEC) -> float | None:
    """Linear projection: at the last window's rate, minutes until ``hot_c``. 0 when already there;
    None when there is no reading or the cabinet is not rising."""
    if temp_c is None or not math.isfinite(temp_c):
        return None
    if temp_c >= hot_c:
        return 0.0
    if rise_c is None or rise_c <= 0:
        return None
    return (hot_c - temp_c) / (rise_c / (window_sec / 60.0))


def _critical(points: Sequence[TempPoint], previous: bool, *, critical_c: float, rearm_c: float) -> bool:
    crit = previous
    for p in points:
        if math.isfinite(p.value):
            crit = p.value >= critical_c or (crit and p.value > rearm_c)
    return crit


def read_cabinet_heat(
    points: Sequence[TempPoint],
    now: datetime,
    *,
    rise_threshold_c: float = DEFAULT_RISE_THRESHOLD_C,
    grace_sec: float = DEFAULT_READING_GRACE_SEC,
    ac_low: bool | None = None,
    previous: "CabinetHeatReading | None" = None,
    critical_c: float = DEFAULT_CRITICAL_C,
    critical_rearm_c: float = DEFAULT_CRITICAL_REARM_C,
) -> CabinetHeatReading:
    """Everything the attention bridge, the action's eligibility and the reflex need, from one read.

    ``previous`` (optional): the caller's last reading; seeds both hysteresis folds so a trip that
    left the re-read window still holds until its re-arm point. ``ac_low``: the AC rule's own
    verdict (only consulted when the cabinet is unknown)."""
    seed: ThermalState = previous.thermal_state if previous is not None else "normal"
    state, temp, age = hysteretic_thermal_state(points, now, max_age_sec=grace_sec, initial_state=seed)
    known = state != "unknown"
    rise = cabinet_rise_c(points, now, RISE_WINDOW_SEC) if known else None
    crit = known and _critical(points, bool(previous and previous.critical), critical_c=critical_c,
                               rearm_c=critical_rearm_c)
    return CabinetHeatReading(
        temp_c=temp,
        age_sec=age,
        thermal_state=state,
        rise_c=rise,
        rise_threshold_c=rise_threshold_c,
        pressure=cabinet_heat_pressure(temp) if known else None,
        warming_error=cabinet_warming_error(state, rise, rise_threshold_c),
        minutes_to_hot=minutes_to_hot(temp, rise) if known else None,
        critical=crit,
        ac_low=ac_low,
        grace_sec=grace_sec,
    )


def minute_mean(points: Sequence[TempPoint], at: datetime, *, half_width_sec: float = MINUTE_SEC / 2) -> float | None:
    """Mean of the readings within +-half_width_sec of ``at`` (the design's "minute mean").
    None when no reading falls in the minute: absence is not a zero."""
    lo, hi = at - timedelta(seconds=half_width_sec), at + timedelta(seconds=half_width_sec)
    vals = [p.value for p in points if lo <= p.ts <= hi and math.isfinite(p.value)]
    return sum(vals) / len(vals) if vals else None


__all__ = [
    "CABINET_HEAT_SIGNAL",
    "CABINET_NODE",
    "CABINET_NODE_ID",
    "CABINET_TEMP_KEY",
    "CabinetHeatReading",
    "DEFAULT_ELEVATED_C",
    "DEFAULT_LOOKAHEAD_MIN",
    "DEFAULT_READING_GRACE_SEC",
    "REFLEX_CABINET_HOT",
    "REFLEX_CABINET_UNKNOWN",
    "minutes_to_hot",
    "DEFAULT_RISE_THRESHOLD_C",
    "RISE_WINDOW_SEC",
    "cabinet_heat_pressure",
    "cabinet_warming_error",
    "hysteretic_thermal_state",
    "minute_mean",
    "read_cabinet_heat",
]


# --- one SQL read, shared by every consumer (substrate runtime, proposal runtime, feedback runtime) ---
# orion_biometrics_summary.timestamp is a VARCHAR ("2026-10-01 01:33:45.55652+00"), so the window is a
# string-prefix comparison on the (node, timestamp) index -- the same shape orion-hardware-watch uses.
CABINET_POINTS_SQL = (
    "SELECT timestamp::timestamptz AS ts, (measurements->>'cabinet_temp_c')::float AS v "
    "FROM orion_biometrics_summary "
    "WHERE node = :node AND timestamp >= :since AND timestamp <= :until AND measurements ? 'cabinet_temp_c' "
    "ORDER BY timestamp"
)


def _ts_key(ts: datetime) -> str:
    from datetime import timezone

    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def load_cabinet_points(conn, *, since: datetime, until: datetime, node: str = CABINET_NODE) -> list[TempPoint]:
    """``conn``: a SQLAlchemy Connection. Ascending TempPoints; empty when nothing is recorded."""
    from sqlalchemy import text

    rows = conn.execute(text(CABINET_POINTS_SQL), {
        "node": node, "since": _ts_key(since), "until": _ts_key(until) + "~"}).fetchall()
    return [TempPoint(ts=r[0], value=float(r[1])) for r in rows if r[1] is not None]
