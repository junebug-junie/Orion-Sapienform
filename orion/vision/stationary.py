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
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

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
    restart: bool = False,
) -> StillRun:
    """The run after observing ``box`` at ``t``. Out-of-order (older) boxes
    leave the run unchanged; a moved box, a gap or ``restart`` (a face on a
    room camera) restarts it."""
    b = tuple(float(v) for v in box)
    if run is None:
        return StillRun(b, t, t)
    if t < run.last:
        return run
    gap_broken = max_gap_sec is not None and (t - run.last) > max_gap_sec
    if restart or gap_broken or box_iou(b, run.box) < iou_threshold:
        return StillRun(b, t, t)
    return StillRun(run.box, run.since, t)


def is_stationary(run: Optional[StillRun], min_sec: float = DEFAULT_MIN_SEC) -> bool:
    return run is not None and min_sec > 0 and run.held_sec() >= min_sec


@dataclass(frozen=True)
class StreamRule:
    """How the stationary rule treats one camera.

    An outdoor camera (``room=False``) keeps a marked prop across gaps: the
    patio skeleton is still there after a night the detector missed it.

    A room camera (``room=True``, the office since 2026-11-24) has a longer
    hold, a gap always ends the run (even a marked one), and a box with a face
    starts a new run. Live 11-15..16 a pile of boxes on the office floor read
    as ``person`` in one box for 4-5 h; a person at the desk sits still too,
    so a sticky prop would hide Juniper's seat on every later visit (review of
    PR #3616, issue #3614).
    """

    min_sec: float
    room: bool = False

    def gap_sec(self, run: Optional[StillRun], max_gap_sec: float) -> Optional[float]:
        """The gap that breaks ``run``: none for a marked outdoor prop."""
        return None if (not self.room and is_stationary(run, self.min_sec)) else max_gap_sec


def _names(spec: Union[str, Iterable[str], None]) -> Tuple[str, ...]:
    items = str(spec).split(",") if isinstance(spec, str) else list(spec or ())
    return tuple(dict.fromkeys(str(x).strip().lower() for x in items if str(x).strip()))


def stream_rules(
    streams: Union[str, Iterable[str], None],
    min_sec: float,
    room_streams: Union[str, Iterable[str], None] = (),
    room_min_sec: Optional[float] = None,
) -> Dict[str, StreamRule]:
    """Stream id (lower case) -> rule. ``room_streams`` win over ``streams``.
    A hold <= 0 turns the rule off for that stream (it is left out)."""
    out = {s: StreamRule(float(min_sec)) for s in _names(streams)}
    room_hold = float(min_sec if room_min_sec is None else room_min_sec)
    out.update({s: StreamRule(room_hold, room=True) for s in _names(room_streams)})
    return {s: r for s, r in out.items() if r.min_sec > 0}
