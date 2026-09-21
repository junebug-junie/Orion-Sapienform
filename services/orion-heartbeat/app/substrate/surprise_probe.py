"""Fail criteria for the surprise / situation-change probe.

Pre-reg: docs/research/preregistration/2026-09-20-heartbeat-surprise.md

Pure scoring — no quimb.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import median
from typing import Iterable, Sequence

GAP_SEC = 1800.0
MIN_SESSION_DUR_SEC = 600.0
MIN_HUB_ATOMS = 8
MAX_KEEP_SESSIONS = 12
MIN_CONTEXTS = 8
MIN_SURPRISE_POINTS = 8
SNAPSHOT_EVERY = 10
WARMUP_ATOMS = 200
SESSION_ATOMS = 400
CONTROL_DUR_SEC = 1200.0
WARMUP_LOOKBACK_SEC = 600.0
CONTROL_CLEARANCE_SEC = 1800.0
SETTLE_FRAC = 0.75
DROP_DELTA_MIN = 0.05


@dataclass(frozen=True)
class SessionSpan:
    start: datetime
    stop: datetime
    hub_atoms: int


@dataclass(frozen=True)
class WindowScore:
    kind: str
    slope: float
    drop: float
    n_surprise: int
    negative_slope: bool


@dataclass(frozen=True)
class SideScore:
    n: int
    median_slope: float
    frac_negative: float
    median_drop: float
    settles: bool


@dataclass(frozen=True)
class SurpriseVerdict:
    thermometer: bool
    holds: bool
    mixed: bool
    reason: str
    sessions: SideScore
    controls: SideScore


def sessionize(timestamps: Sequence[datetime], *, gap_sec: float = GAP_SEC) -> list[SessionSpan]:
    if not timestamps:
        return []
    ordered = sorted(timestamps)
    spans: list[SessionSpan] = []
    start = ordered[0]
    prev = ordered[0]
    n = 1
    for ts in ordered[1:]:
        gap = (ts - prev).total_seconds()
        if gap > gap_sec:
            spans.append(SessionSpan(start=start, stop=prev, hub_atoms=n))
            start = ts
            n = 1
        else:
            n += 1
        prev = ts
    spans.append(SessionSpan(start=start, stop=prev, hub_atoms=n))
    return spans


def keep_sessions(spans: Iterable[SessionSpan]) -> list[SessionSpan]:
    kept = [
        s
        for s in spans
        if (s.stop - s.start).total_seconds() >= MIN_SESSION_DUR_SEC and s.hub_atoms >= MIN_HUB_ATOMS
    ]
    kept.sort(key=lambda s: s.start)
    if len(kept) > MAX_KEEP_SESSIONS:
        kept = kept[-MAX_KEEP_SESSIONS:]
    return kept


def ols_slope(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0.0:
        return 0.0
    return float(num / den)


def window_drop(values: Sequence[float]) -> float:
    if len(values) < 6:
        return 0.0
    head = values[:3]
    tail = values[-3:]
    return float(sum(head) / 3.0 - sum(tail) / 3.0)


def score_window(*, kind: str, surprises: Sequence[float]) -> WindowScore | None:
    if len(surprises) < MIN_SURPRISE_POINTS:
        return None
    slope = ols_slope(surprises)
    return WindowScore(
        kind=kind,
        slope=slope,
        drop=window_drop(surprises),
        n_surprise=len(surprises),
        negative_slope=slope < 0.0,
    )


def score_side(windows: Sequence[WindowScore]) -> SideScore:
    if not windows:
        return SideScore(n=0, median_slope=0.0, frac_negative=0.0, median_drop=0.0, settles=False)
    slopes = [w.slope for w in windows]
    drops = [w.drop for w in windows]
    frac = sum(1 for w in windows if w.negative_slope) / len(windows)
    med_slope = float(median(slopes))
    settles = med_slope < 0.0 and frac >= SETTLE_FRAC
    return SideScore(
        n=len(windows),
        median_slope=med_slope,
        frac_negative=frac,
        median_drop=float(median(drops)),
        settles=settles,
    )


def decide_surprise(*, sessions: SideScore, controls: SideScore) -> SurpriseVerdict:
    if sessions.n < MIN_CONTEXTS or controls.n < MIN_CONTEXTS:
        return SurpriseVerdict(
            thermometer=False,
            holds=False,
            mixed=False,
            reason=(
                f"UNVERIFIED: need ≥{MIN_CONTEXTS} scored sessions and controls, "
                f"got sessions={sessions.n} controls={controls.n}"
            ),
            sessions=sessions,
            controls=controls,
        )
    if sessions.settles and controls.settles:
        return SurpriseVerdict(
            thermometer=False,
            holds=False,
            mixed=False,
            reason="UNVERIFIED: saturation — control windows settle too",
            sessions=sessions,
            controls=controls,
        )
    drop_holds = sessions.median_drop >= controls.median_drop + DROP_DELTA_MIN
    settling_holds = sessions.settles and not controls.settles
    if settling_holds and drop_holds:
        return SurpriseVerdict(
            thermometer=False,
            holds=True,
            mixed=False,
            reason="holds: sessions settle and start louder than controls",
            sessions=sessions,
            controls=controls,
        )
    if settling_holds and not drop_holds:
        return SurpriseVerdict(
            thermometer=False,
            holds=False,
            mixed=True,
            reason="mixed: sessions settle but the start is not louder than controls",
            sessions=sessions,
            controls=controls,
        )
    if drop_holds and not settling_holds:
        return SurpriseVerdict(
            thermometer=False,
            holds=False,
            mixed=True,
            reason="mixed: chat start is louder, but no session-specific decline",
            sessions=sessions,
            controls=controls,
        )
    return SurpriseVerdict(
        thermometer=True,
        holds=False,
        mixed=False,
        reason="thermometer: random stretches look like chat stretches",
        sessions=sessions,
        controls=controls,
    )


def l2(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"profile length mismatch {len(a)} vs {len(b)}")
    return float(sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5)


def pick_controls(
    *,
    sessions: Sequence[SessionSpan],
    window_starts: Sequence[datetime],
    clearance: timedelta = timedelta(seconds=CONTROL_CLEARANCE_SEC),
    want: int = MAX_KEEP_SESSIONS,
) -> list[datetime]:
    """Keep candidate starts that sit clear of every kept session."""
    out: list[datetime] = []
    for start in window_starts:
        ok = True
        for sess in sessions:
            if start + timedelta(seconds=CONTROL_DUR_SEC) < sess.start - clearance:
                continue
            if start > sess.stop + clearance:
                continue
            ok = False
            break
        if ok:
            out.append(start)
        if len(out) >= want:
            break
    return out
