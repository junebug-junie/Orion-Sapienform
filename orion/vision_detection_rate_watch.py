"""Blind-detector watch: frames keep arriving, but the detector stops seeing.

The incident (2026-10-22 .. 10-23): the walkway GroundingDINO detector kept
receiving frames, and the vision window kept writing one
``vision_scene_inventory`` row per window, but almost every window came back
empty. Walkway windows with at least one detection fell from ~0.5-1.0 of
daytime windows to 0.01-0.2. Mean detection scores had dropped from ~0.38 to
~0.36, under the 0.35 box threshold. Nothing alarmed: the camera-input rung
(``camera_input:<stream>``) only asks "did a frame arrive", and frames did.

What this measures. For one stream, over a trailing span (default 3 h):

    share = windows with >= 1 detection / windows

``vision_scene_inventory`` gets one row per vision-window flush, and a window
is only flushed when detector replies arrived (``frame_count`` = replies), so
the denominator is "the detector ran" and the numerator is "it saw something".

Night is legitimately empty on the walkway, so the share is compared with the
same clock span on each of the previous ``baseline_days`` days (median).
Three outcomes:

- ``collapsed`` (red): the baseline says this span normally sees things
  (median share >= ``floor``) and the current share is below ``ratio`` x that.
- ``ok`` (green): otherwise, including a quiet span whose baseline is also
  quiet. This is how a normal night reads calm.
- ``unknown``: too few windows now (the detector or the camera is not
  delivering: the camera-input rung owns that), or too few baseline days with
  data. Absence never reads as calm.

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


@dataclass(frozen=True)
class DetectionRateConfig:
    span: timedelta = timedelta(hours=3)
    baseline_days: int = 7
    # A baseline day needs this many windows in the span to count. Normal is
    # ~380-540 windows per hour on both cameras.
    min_baseline_windows: int = 100
    min_baseline_days: int = 3
    # Fewer windows than this right now: the detector is not running steadily
    # enough to judge. Reads unknown, never calm.
    min_current_windows: int = 100
    # Below this median baseline share the span is "normally quiet" and a
    # collapse cannot be told apart from a quiet night: reads ok.
    floor: float = 0.2
    # Red when the current share is under ratio x the baseline median.
    ratio: float = 0.25


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
    status: str  # "ok" | "collapsed" | "unknown"
    current: SpanCount
    baseline_share: Optional[float]
    baseline_days_used: int
    reason: str

    @property
    def name(self) -> str:
        return f"detection_rate:{self.stream}"

    @property
    def red(self) -> bool:
        return self.status == "collapsed"

    def summary(self) -> str:
        cur = self.current.share
        cur_s = f"{cur:.2f}" if cur is not None else "?"
        base_s = f"{self.baseline_share:.2f}" if self.baseline_share is not None else "?"
        return (
            f"{self.name}: {self.status} -- {self.current.detected}/{self.current.windows} windows saw "
            f"something (share {cur_s}) vs same-hours median {base_s} over {self.baseline_days_used} "
            f"day(s); {self.reason}"
        )


def evaluate_detection_rate(
    stream: str,
    current: SpanCount,
    baseline: Sequence[SpanCount],
    cfg: DetectionRateConfig = DEFAULT_CONFIG,
) -> DetectionRateResult:
    """``current``: the trailing span now. ``baseline``: the same clock span on
    each previous day (any order; days with too few windows are skipped)."""
    usable = [b.share for b in baseline if b.windows >= cfg.min_baseline_windows and b.share is not None]
    base = statistics.median(usable) if usable else None
    n = len(usable)
    if current.windows < cfg.min_current_windows:
        return DetectionRateResult(
            stream, "unknown", current, base, n,
            f"only {current.windows} detector windows in the span (need {cfg.min_current_windows}); "
            "camera or detector not delivering",
        )
    if n < cfg.min_baseline_days:
        return DetectionRateResult(
            stream, "unknown", current, base, n,
            f"only {n} baseline day(s) with data (need {cfg.min_baseline_days})",
        )
    assert base is not None and current.share is not None
    if base < cfg.floor:
        return DetectionRateResult(stream, "ok", current, base, n, f"quiet hours (baseline under {cfg.floor:.2f})")
    if current.share < cfg.ratio * base:
        return DetectionRateResult(
            stream, "collapsed", current, base, n,
            f"under {cfg.ratio:.2f} x the usual share while the detector keeps running: it has gone "
            "blind (scores under the box threshold? compare the hourly frame samples)",
        )
    return DetectionRateResult(stream, "ok", current, base, n, "within range")


#: Parameters: span end (timestamptz), baseline_days, stream_id, span seconds.
#: Row per day offset d (0 = now): windows, windows with any detection.
#: Each offset is an index range scan on (stream_id, observed_at).
DETECTION_RATE_SQL = (
    "SELECT d, count(v.window_id), count(v.window_id) FILTER (WHERE v.counts <> '{}'::jsonb) "
    "FROM generate_series(0, %s) AS d "
    "LEFT JOIN vision_scene_inventory v ON v.stream_id = %s "
    "AND v.observed_at > %s::timestamptz - make_interval(days => d) - make_interval(secs => %s) "
    "AND v.observed_at <= %s::timestamptz - make_interval(days => d) "
    "GROUP BY d ORDER BY d"
)


def sql_params(stream: str, end, cfg: DetectionRateConfig = DEFAULT_CONFIG) -> list:
    return [cfg.baseline_days, stream, end, cfg.span.total_seconds(), end]


def split_rows(rows: Sequence[Sequence[int]]) -> tuple[SpanCount, list[SpanCount]]:
    """``rows`` from DETECTION_RATE_SQL -> (current, baseline days)."""
    by_d = {int(r[0]): SpanCount(int(r[1] or 0), int(r[2] or 0)) for r in rows}
    current = by_d.get(0, SpanCount(0, 0))
    return current, [c for d, c in sorted(by_d.items()) if d > 0]
