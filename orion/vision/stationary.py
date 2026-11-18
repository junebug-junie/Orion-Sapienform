"""Stationary fixtures: a "person" box that never moves is a prop, not a visitor.

Live 2026-10-15 (walkway camera): a Halloween skeleton at the patio table was
detected as ``person`` (score ~0.37) in the same box, ~[2060, 877, 2274, 1227]
on a 2560x1920 frame, 40 times in 13 minutes, all night. Once the patio is
fingerprinted it would become an individual seen every night: an ask
("who is this?"), attention events, a rhythm subject, and a permanent
"someone is on the patio".

The rule is geometry, not words: a detection whose box keeps overlapping the
box that started its run (IoU >= ``iou_threshold``), with no gap longer than
``max_gap_sec`` between observations, for at least ``min_sec`` is stationary.
The first box that moves away starts a new run, so a real person who sat
still for a while stops being stationary the moment they get up.

Pure functions over epoch seconds; callers keep the run state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_MIN_SEC = 1800.0


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class StillRun:
    """One run of a box staying put: the anchor box, when the run started,
    and the latest observation that kept it going."""

    box: Tuple[float, ...]
    since: float
    last: float

    def held_sec(self) -> float:
        return max(0.0, self.last - self.since)


def advance(
    run: Optional[StillRun],
    box: Sequence[float],
    t: float,
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    max_gap_sec: Optional[float] = None,
) -> StillRun:
    """The run after observing ``box`` at ``t``. Out-of-order (older) boxes
    leave the run unchanged; a moved box or a gap restarts it."""
    b = tuple(float(v) for v in box)
    if run is None:
        return StillRun(b, t, t)
    if t < run.last:
        return run
    gap_broken = max_gap_sec is not None and (t - run.last) > max_gap_sec
    if gap_broken or box_iou(b, run.box) < iou_threshold:
        return StillRun(b, t, t)
    return StillRun(run.box, run.since, t)


def is_stationary(run: Optional[StillRun], min_sec: float = DEFAULT_MIN_SEC) -> bool:
    return run is not None and min_sec > 0 and run.held_sec() >= min_sec
