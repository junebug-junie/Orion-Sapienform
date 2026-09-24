"""Attention score for one sighting: a fixed weighted sum Orion can explain.

docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md idea 6.
There is no "suspicious" category anywhere in this module, on purpose: the
output is a number plus the named components that produced it, so a false
alarm can be traced to the component that fired. Words like "suspicious",
"stranger", "intruder" are never produced as categories.

Components, each in [0, 1] or None:

- ``unknown``: 1.0 when the individual has no Juniper-given label, else 0.0.
- ``unusual_time``: how far this hour is from the individual's own usual
  hours (falls back to all sightings of the same kind on the stream).
  **None** when the history covers fewer than ``min_support_days`` distinct
  days -- not enough evidence to call any time unusual.
- ``long_dwell``: dwell relative to the zone's ``dwell_rare_sec``;
  ``dwell == dwell_rare_sec`` scores 0.5, twice that scores 1.0. None when
  the zone has no ``dwell_rare_sec``.
- ``few_prior_sightings``: 1.0 for a first-ever sighting, falling linearly
  to 0.0 at ``FEW_SIGHTINGS_SCALE`` prior sightings.

A None component contributes 0 to the score and is recorded as None (null
in JSON), never as 0 -- "no evidence" and "evidence of normal" are kept apart.

Pure: no clock, no DB. The caller passes the history.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence

WEIGHTS: Dict[str, float] = {
    "unknown": 0.25,
    "unusual_time": 0.30,
    "long_dwell": 0.30,
    "few_prior_sightings": 0.15,
}

FEW_SIGHTINGS_SCALE = 10.0
DEFAULT_MIN_SUPPORT_DAYS = 5
# +-1 hour neighbourhood when judging "usual" hours (circular over 24h).
_HOUR_NEIGHBOURHOOD = 1


@dataclass(frozen=True)
class HourHistory:
    """Past sightings as (local hour, local date) pairs, plus where they came from."""

    hours: Sequence[int]
    days: Sequence[str]
    basis: str  # "individual" | "kind"


@dataclass(frozen=True)
class AttentionResult:
    score: float
    components: Dict[str, Optional[float]]
    reasons: List[str]
    basis: Optional[str]


def _neighbourhood_mass(hist: Sequence[int], hour: int) -> int:
    return sum(hist[(hour + d) % 24] for d in range(-_HOUR_NEIGHBOURHOOD, _HOUR_NEIGHBOURHOOD + 1))


def unusual_time_component(
    hour: int, history: Optional[HourHistory], *, min_support_days: int = DEFAULT_MIN_SUPPORT_DAYS
) -> Optional[float]:
    """1 - (mass near this hour / mass near the busiest hour). None under min support."""
    if history is None or len(set(history.days)) < min_support_days or not history.hours:
        return None
    hist = [0] * 24
    for h in history.hours:
        hist[int(h) % 24] += 1
    peak = max(_neighbourhood_mass(hist, h) for h in range(24))
    if peak <= 0:
        return None
    return round(1.0 - _neighbourhood_mass(hist, hour) / peak, 4)


def long_dwell_component(dwell_sec: float, dwell_rare_sec: Optional[float]) -> Optional[float]:
    if dwell_rare_sec is None or dwell_rare_sec <= 0:
        return None
    return round(min(1.0, max(0.0, dwell_sec) / (2.0 * dwell_rare_sec)), 4)


def few_prior_sightings_component(prior_sightings: int) -> float:
    return round(max(0.0, 1.0 - max(0, prior_sightings) / FEW_SIGHTINGS_SCALE), 4)


def score_sighting(
    *,
    labeled: bool,
    local_hour: int,
    dwell_sec: float,
    dwell_rare_sec: Optional[float],
    prior_sightings: int,
    individual_history: Optional[HourHistory],
    kind_history: Optional[HourHistory],
    min_support_days: int = DEFAULT_MIN_SUPPORT_DAYS,
    weights: Optional[Dict[str, float]] = None,
) -> AttentionResult:
    w = weights or WEIGHTS
    basis: Optional[str] = None
    unusual = unusual_time_component(local_hour, individual_history, min_support_days=min_support_days)
    if unusual is not None:
        basis = "individual"
    else:
        unusual = unusual_time_component(local_hour, kind_history, min_support_days=min_support_days)
        if unusual is not None:
            basis = "kind"
    components: Dict[str, Optional[float]] = {
        "unknown": 0.0 if labeled else 1.0,
        "unusual_time": unusual,
        "long_dwell": long_dwell_component(dwell_sec, dwell_rare_sec),
        "few_prior_sightings": few_prior_sightings_component(prior_sightings),
    }
    score = sum(w[k] * (v or 0.0) for k, v in components.items())
    reasons = _reasons(components, dwell_sec=dwell_sec, dwell_rare_sec=dwell_rare_sec,
                       prior_sightings=prior_sightings, basis=basis)
    return AttentionResult(score=round(score, 4), components=components, reasons=reasons, basis=basis)


def _fmt_minutes(sec: float) -> str:
    m = sec / 60.0
    return f"{m:.1f} minutes" if m < 10 else f"{m:.0f} minutes"


def _reasons(
    c: Dict[str, Optional[float]], *, dwell_sec: float, dwell_rare_sec: Optional[float],
    prior_sightings: int, basis: Optional[str],
) -> List[str]:
    out: List[str] = []
    if c["unknown"]:
        out.append("I do not have a name for them")
    if c["unusual_time"] is not None and c["unusual_time"] >= 0.5:
        whose = "for them" if basis == "individual" else "for anyone of this kind here"
        out.append(f"this time of day is unusual {whose} ({c['unusual_time']:.2f})")
    if c["long_dwell"] is not None and dwell_rare_sec and dwell_sec >= dwell_rare_sec:
        out.append(
            f"they stayed {_fmt_minutes(dwell_sec)} where {_fmt_minutes(dwell_rare_sec)} is already rare"
        )
    if c["few_prior_sightings"] >= 0.5:
        out.append(
            "I have never seen them before" if prior_sightings == 0
            else f"I have seen them only {prior_sightings} time{'s' if prior_sightings != 1 else ''} before"
        )
    return out


def attention_narrative(
    *, kind: str, zone: Optional[str], stream_id: str, local_time: datetime,
    dwell_sec: float, result: AttentionResult,
) -> str:
    where = f"the {zone}" if zone else f"the {stream_id} camera"
    reasons = "; ".join(result.reasons) if result.reasons else "several small things added up"
    return (
        f"At {local_time.strftime('%H:%M')} a {kind} at {where} caught my attention "
        f"(score {result.score:.2f}): {reasons}."
    )


def hour_history(pairs: Iterable[tuple[int, str]], basis: str) -> HourHistory:
    hours, days = [], []
    for h, d in pairs:
        hours.append(int(h))
        days.append(str(d))
    return HourHistory(hours=hours, days=days, basis=basis)
