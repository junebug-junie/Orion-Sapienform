"""Blind-detector watch: frames keep arriving, but the detector stops seeing.

The incident (2026-10-22 .. 10-23): the walkway GroundingDINO detector kept
receiving frames, and the vision window kept writing one
``vision_scene_inventory`` row per window, but almost every window came back
empty. Walkway windows with at least one detection fell from ~0.5-1.0 of
daytime windows to 0.01-0.2. Mean detection scores had dropped from ~0.38 to
~0.36, under the 0.35 box threshold. Nothing alarmed: the camera-input rung
(``camera_input:<stream>``) only asks "did an embedding arrive", and they did.

What this measures. For one stream, over a trailing span (default 3 h):

    share = windows with >= 1 detection / windows

``vision_scene_inventory`` gets one row per vision-window flush, and a window
is only flushed when detector replies arrived (``frame_count`` = replies), so
the denominator is "the detector ran" and the numerator is "it saw something".

Night is legitimately empty on the walkway, so the share is compared with the
same clock span on each of the previous ``baseline_days`` days; the reference
is their median. Known limit: a blind spell that lasts 4+ days becomes its own
baseline and stops reading red. The onset still cards. The upper quartile
was tried as the reference (2026-10-29 replay) and raised 48 false red ticks
on normal days 10-18..10-25, against 0 with the median.

Outcomes (``red`` = carded; ``green`` = may re-arm a delivered card):

- ``collapsed`` (red): the reference says this span normally sees things
  (>= ``floor``) and the current share is below ``ratio`` x the reference.
- ``silent`` (red): the detector produced (almost) no windows while the
  baseline had them AND the stream's camera-input rung is fresh: embeddings
  arrive, detections do not. Nothing else covers this case.
- ``ok`` (green): current share >= ``rearm_ratio`` x the reference.
- ``low``: between ``ratio`` and ``rearm_ratio`` x the reference. Neither red
  nor green (hysteresis: a share hovering at the line does not re-card).
- ``quiet``: the reference is under ``floor`` (a normal night). Calm, but not
  evidence that a blind detector recovered, so not green either.
- ``unknown``: too few windows and the camera itself is down or its state is
  not known (the camera-input rung owns that), or too few baseline days.
  Absence never reads as calm.

Pure; Postgres IO lives in ``scripts/check_substrate_ladder_liveness.py``.
Stdlib only.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, Sequence

#: Streams the watch covers: the cameras that run around the clock (same list
#: as the camera-input rungs). The Ring doorbell is motion-triggered and has
#: ~150 windows a day; there is no steady rate to collapse.
DETECTION_RATE_STREAMS: tuple[str, ...] = ("cam0", "walkway")

RED_STATUSES = frozenset({"collapsed", "silent"})


@dataclass(frozen=True)
class DetectionRateConfig:
    span: timedelta = timedelta(hours=3)
    baseline_days: int = 7
    # A baseline day needs this many windows in the span to count. Normal is
    # ~380-540 windows per hour on both cameras.
    min_baseline_windows: int = 100
    min_baseline_days: int = 3
    # Fewer windows than this right now: the detector is not running steadily
    # enough to judge the share.
    min_current_windows: int = 100
    # Below this reference share the span is "normally quiet" and a collapse
    # cannot be told apart from a quiet night.
    floor: float = 0.2
    # Red when the current share is under ratio x the reference.
    ratio: float = 0.25
    # Green (re-arms a delivered card) only at or above this x the reference.
    rearm_ratio: float = 0.5


DEFAULT_CONFIG = DetectionRateConfig()


@dataclass(frozen=True)
class SpanCount:
    windows: int
    detected: int

    @property
    def share(self) -> Optional[float]:
        return self.detected / self.windows if self.windows > 0 else None


@dataclass(frozen=True)
class DetectionRateResult:
    stream: str
    status: str  # collapsed | silent | ok | low | quiet | unknown
    current: SpanCount
    baseline_share: Optional[float]
    baseline_days_used: int
    reason: str

    @property
    def name(self) -> str:
        return f"detection_rate:{self.stream}"

    @property
    def red(self) -> bool:
        return self.status in RED_STATUSES

    @property
    def green(self) -> bool:
        return self.status == "ok"

    def summary(self) -> str:
        cur = self.current.share
        cur_s = f"{cur:.2f}" if cur is not None else "?"
        base_s = f"{self.baseline_share:.2f}" if self.baseline_share is not None else "?"
        return (
            f"{self.name}: {self.status} -- {self.current.detected}/{self.current.windows} windows saw "
            f"something (share {cur_s}) vs same-hours reference {base_s} over {self.baseline_days_used} "
            f"day(s); {self.reason}"
        )


def evaluate_detection_rate(
    stream: str,
    current: SpanCount,
    baseline: Sequence[SpanCount],
    cfg: DetectionRateConfig = DEFAULT_CONFIG,
    *,
    camera_fresh: Optional[bool] = None,
) -> DetectionRateResult:
    """``current``: the trailing span now. ``baseline``: the same clock span on
    each previous day (any order; days with too few windows are skipped).
    ``camera_fresh``: the stream's camera-input rung (None = not known)."""
    usable = [b.share for b in baseline if b.windows >= cfg.min_baseline_windows and b.share is not None]
    base = statistics.median(usable) if usable else None
    n = len(usable)
    if current.windows < cfg.min_current_windows:
        if camera_fresh is True and n >= cfg.min_baseline_days:
            return DetectionRateResult(
                stream, "silent", current, base, n,
                f"only {current.windows} detector windows in the span while the camera is delivering "
                f"(usually >= {cfg.min_baseline_windows}): the detector or the vision window has stopped",
            )
        return DetectionRateResult(
            stream, "unknown", current, base, n,
            f"only {current.windows} detector windows in the span (need {cfg.min_current_windows}); "
            "camera down or its state unknown (see camera_input)",
        )
    if n < cfg.min_baseline_days:
        return DetectionRateResult(
            stream, "unknown", current, base, n,
            f"only {n} baseline day(s) with data (need {cfg.min_baseline_days})",
        )
    assert base is not None and current.share is not None
    if base < cfg.floor:
        return DetectionRateResult(stream, "quiet", current, base, n, f"quiet hours (reference under {cfg.floor:.2f})")
    if current.share < cfg.ratio * base:
        return DetectionRateResult(
            stream, "collapsed", current, base, n,
            f"under {cfg.ratio:.2f} x the usual share while the detector keeps running: it has gone "
            "blind (scores under the box threshold? compare the hourly frame samples)",
        )
    if current.share < cfg.rearm_ratio * base:
        return DetectionRateResult(stream, "low", current, base, n,
                                   f"between {cfg.ratio:.2f} and {cfg.rearm_ratio:.2f} x the usual share")
    return DetectionRateResult(stream, "ok", current, base, n, "within range")


#: Parameters: baseline_days, stream_id, span end (timestamptz), span seconds,
#: span end. Row per day offset d (0 = now): windows, windows with a detection.
#: Day offsets are exact 86400 s (not calendar days), so a DST change in the
#: session time zone cannot shift the baseline. Each offset is an index range
#: scan on (stream_id, observed_at).
DETECTION_RATE_SQL = (
    "SELECT d, count(v.window_id), count(v.window_id) FILTER (WHERE v.counts <> '{}'::jsonb) "
    "FROM generate_series(0, %s) AS d "
    "LEFT JOIN vision_scene_inventory v ON v.stream_id = %s "
    "AND v.observed_at > %s::timestamptz - make_interval(secs => d * 86400 + %s) "
    "AND v.observed_at <= %s::timestamptz - make_interval(secs => d * 86400) "
    "GROUP BY d ORDER BY d"
)


def sql_params(stream: str, end, cfg: DetectionRateConfig = DEFAULT_CONFIG) -> list:
    return [cfg.baseline_days, stream, end, cfg.span.total_seconds(), end]


def split_rows(rows: Sequence[Sequence[int]]) -> tuple[SpanCount, list[SpanCount]]:
    """``rows`` from DETECTION_RATE_SQL -> (current, baseline days)."""
    by_d = {int(r[0]): SpanCount(int(r[1] or 0), int(r[2] or 0)) for r in rows}
    current = by_d.get(0, SpanCount(0, 0))
    return current, [c for d, c in sorted(by_d.items()) if d > 0]
