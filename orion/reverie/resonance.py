"""Phase H — resonance detector (the automated ouroboros tripwire).

A reverie chain habituates a resolved theme into a refractory cooldown so a
discharged loop cannot immediately re-ignite. Resonance is the failure of that
damping: a theme that keeps recurring *inside* its own refractory window — a
runaway loop feeding itself. This detector is the automated guard that watches
for it.

Pure and deterministic (§4): given a sequence of (theme_key, timestamp) chain
events and the refractory bound, the alert is a pure function — no LLM, no I/O.
That is exactly what makes it a trustworthy tripwire and what licenses (or
withholds licence for) turning the compaction applier's hot gate on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from orion.schemas.reverie import MAX_RESONANCE_SAMPLES, ResonanceAlertV1

# A theme must breach its refractory bound at least this many times to trip.
DEFAULT_MIN_VIOLATIONS = 2


@dataclass(frozen=True)
class ThemeEvent:
    """One chain occurrence: a theme fired at a time."""

    theme_key: str
    at: datetime


def _sorted_times(events: list[ThemeEvent], theme_key: str) -> list[datetime]:
    times = [e.at for e in events if e.theme_key == theme_key and e.theme_key not in ("", "unknown")]
    return sorted(times)


def detect_resonance(
    events: list[ThemeEvent],
    *,
    refractory_sec: float,
    min_violations: int = DEFAULT_MIN_VIOLATIONS,
) -> ResonanceAlertV1 | None:
    """Return the most severe resonance alert, or None if every theme is damped.

    A *violation* is a consecutive recurrence of the same theme with a gap
    shorter than `refractory_sec` — i.e. the theme re-ignited before its
    habituation cooldown expired. A theme with `>= min_violations` such gaps is a
    runaway loop. The returned alert is for the theme with the most violations
    (ties broken by the smallest gap — the tightest loop).
    """
    if refractory_sec <= 0 or not events:
        return None
    min_violations = max(1, int(min_violations))

    best: ResonanceAlertV1 | None = None
    best_key: tuple[int, float, str] | None = None
    # Iterate in sorted theme order so a full tie resolves deterministically
    # (independent of set-iteration / PYTHONHASHSEED).
    for theme_key in sorted({e.theme_key for e in events}):
        if theme_key in ("", "unknown"):
            continue
        times = _sorted_times(events, theme_key)
        if len(times) < 2:
            continue
        breaching_gaps: list[float] = []
        for prev, cur in zip(times, times[1:]):
            gap = (cur - prev).total_seconds()
            if gap < refractory_sec:
                breaching_gaps.append(gap)
        if len(breaching_gaps) < min_violations:
            continue
        # Deterministic id keyed on theme + the refractory window the last
        # occurrence falls in, so a persisting runaway dedups (ON CONFLICT) into
        # one row per window instead of flooding the observability surface.
        bucket = int(times[-1].timestamp() // refractory_sec)
        # Total order: more violations, then tighter loop, then theme name.
        rank = (len(breaching_gaps), -min(breaching_gaps), theme_key)
        if best_key is None or rank > best_key:
            best_key = rank
            best = ResonanceAlertV1(
                alert_id=f"resonance:{theme_key}:{bucket}",
                theme_key=theme_key,
                violation_count=len(breaching_gaps),
                refractory_sec=float(refractory_sec),
                min_gap_sec=min(breaching_gaps),
                occurrences=len(times),
                sample_ats=times[:MAX_RESONANCE_SAMPLES],
            )
    return best
