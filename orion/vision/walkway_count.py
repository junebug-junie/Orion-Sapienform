"""How many people passed the walkway camera: the candidate estimators (pure).

Spec PR #2763, Item 2 (people are over-counted), patch B, and decision 6.

Every estimator here reads the same detector boxes. They are not independent signals; they
are different readings of one sensor (spec, metric gate point 2):

- ``clusters``: person individuals sighted in the hour (the rollup's passings, the HIGH end of
  the range ``orion/vision/walkway_rollup.py`` prints). Over-counts: the matcher makes a new
  cluster for most single detections.
- ``most_at_once``: the most people in one vision window (the LOW end of that range). A floor.
- ``stories``: multi-frame motion stories, plus clear glimpses that no other story already
  covers (``story_count``). Theory anchor: a passage is one continuous track through the
  scene, and ``services/orion-sql-writer/app/vision_motion.py`` already builds tracks.
- ``stories_in_range``: ``stories`` held inside [most_at_once, clusters] (``within_range``).
  This is the only form a story-based count may take in Orion-facing text.
- ``run_peaks``: presence runs (frames with a person, no gap over ``RUN_GAP_SEC``), each
  counted as its peak. Theory anchor: contiguous occupancy.

Decision 6 (Juniper, 2026-10-23): a one-frame glimpse story counts as a person only if it is
clearly a person. ``clear_glimpse`` is that rule. It is used only by the ``stories``
estimator and its eval (``services/orion-sql-writer/evals/run_walkway_count_estimator_eval.py``).
It does not move the Orion-facing range: the low stays the window count (a frame fact) and
the high stays the clusters (every glimpse detection is also a sighting, so it is already in
the clusters). Nothing here is a schema column.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Optional, Sequence

# --- decision 6: what makes a glimpse "clearly a person" ----------------------------------------
#
# Anchored on the stories Juniper checked by hand in the Hub (vision_motion_story_verdict,
# walkway persons, read 2026-10-23): 40 stories marked "right" and 2 marked "wrong_other".
# - Score: the best detection of a "right" story is >= 0.437 in 75 % of them (p25 = 0.437).
#   The two "wrong" stories peak at 0.363 / 0.397. Rounded up: 0.44.
# - Box height: 95 % of detections in "right" stories are >= 120 px tall (p5 = 121 px of a
#   1920 px frame). The "wrong" stories' boxes are 55-77 px (p5-p50). 120 / 1920 = 0.0625.
# - Not edge-clipped and inside a configured zone: a clipped box is part of a body, and a box
#   outside every zone is not on the walkway.
# - Not a prop and not stationary: the motion reducer gives a prop story ``drop_reason=prop``
#   before it can be a glimpse; the individual check is a second guard.
CLEAR_GLIMPSE_MIN_SCORE = 0.44
CLEAR_GLIMPSE_MIN_BOX_H_FRAC = 0.0625
DEFAULT_FRAME_H = 1920.0  # VISION_MOTION_FRAME_HEIGHT, used when a story did not record its frame

# Stories that followed one body over several frames (vision_motion.build_stories reasons).
# ``None`` is published. ``prop``, ``glimpse``, ``parked`` and ``companion`` are not here.
MULTI_FRAME_REASONS = frozenset({None, "standing", "edge_clipped", "out_of_scope", "too_long"})
GLIMPSE = "glimpse"

# A clear glimpse within this many seconds of another counted person story is taken as the
# same person (detector flicker beside a track). Same gap the motion reducer closes a story on
# (VISION_MOTION_STORY_CLOSE_GAP_SEC=11).
STORY_DEDUPE_SEC = 11.0

# Presence runs: frames with a person, no gap over this (the spec's definition).
RUN_GAP_SEC = 30.0

ESTIMATORS = ("clusters", "most_at_once", "stories", "stories_in_range", "stories_multi_only", "run_peaks", "runs")


def _f(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def clear_detection(d: Mapping[str, Any], *, frame_h: Any = None,
                    prop_ids: Iterable[str] = ()) -> bool:
    """One evidence detection of a motion story (``vision_motion_story.evidence[]``)."""
    if str(d.get("label") or "person").strip().lower() != "person":
        return False
    if d.get("clipped") is True or not d.get("zone"):
        return False
    iid = d.get("individual_id")
    if iid and str(iid) in set(prop_ids):
        return False
    score = _f(d.get("score"))
    box = d.get("box") or []
    if score is None or score < CLEAR_GLIMPSE_MIN_SCORE or len(box) != 4:
        return False
    y1, y2 = _f(box[1]), _f(box[3])
    fh = _f(frame_h) or DEFAULT_FRAME_H
    if y1 is None or y2 is None or fh <= 0:
        return False
    return (y2 - y1) / fh >= CLEAR_GLIMPSE_MIN_BOX_H_FRAC


def clear_glimpse(story: Mapping[str, Any], *, prop_ids: Iterable[str] = ()) -> bool:
    """Decision 6: a person glimpse story counts only when one of its detections is clear."""
    if str(story.get("kind") or "") != "person" or story.get("drop_reason") != GLIMPSE:
        return False
    if story.get("individual_id") and str(story["individual_id"]) in set(prop_ids):
        return False
    ev = story.get("evidence") or []
    return any(isinstance(d, Mapping) and clear_detection(d, frame_h=story.get("frame_h"), prop_ids=prop_ids)
               for d in ev)


def _span(s: Mapping[str, Any]) -> Optional[tuple[datetime, datetime]]:
    a, b = s.get("started_at"), s.get("ended_at") or s.get("started_at")
    if not isinstance(a, datetime) or not isinstance(b, datetime):
        return None
    return (a, b) if b >= a else (b, a)


def _near(a: tuple[datetime, datetime], b: tuple[datetime, datetime], gap: float) -> bool:
    g = timedelta(seconds=gap)
    return a[0] <= b[1] + g and b[0] <= a[1] + g


@dataclass(frozen=True)
class StoryCount:
    multi_frame: int
    clear_glimpses: int        # clear and not covered by another counted story
    covered_glimpses: int      # clear but within STORY_DEDUPE_SEC of a counted story
    unclear_glimpses: int

    @property
    def total(self) -> int:
        return self.multi_frame + self.clear_glimpses


def story_count(stories: Sequence[Mapping[str, Any]], *, prop_ids: Iterable[str] = (),
                dedupe_sec: float = STORY_DEDUPE_SEC) -> StoryCount:
    """Person stories of one stream in one window, counted as passages.

    Multi-frame stories each count once (a person standing 2 minutes is one ``standing``
    story). A glimpse counts only when it is clear (decision 6) AND no multi-frame story or
    already-counted glimpse lies within ``dedupe_sec`` of it.
    """
    people = [s for s in stories if str(s.get("kind") or "") == "person"]
    multi = [s for s in people if s.get("drop_reason") in MULTI_FRAME_REASONS]
    counted = [sp for sp in (_span(s) for s in multi) if sp]
    glimpses = sorted((s for s in people if s.get("drop_reason") == GLIMPSE),
                      key=lambda s: (_span(s) or (datetime.max, datetime.max))[0])
    clear = covered = unclear = 0
    for g in glimpses:
        if not clear_glimpse(g, prop_ids=prop_ids):
            unclear += 1
            continue
        sp = _span(g)
        if sp is None or any(_near(sp, c, dedupe_sec) for c in counted):
            covered += 1
            continue
        counted.append(sp)
        clear += 1
    return StoryCount(len(multi), clear, covered, unclear)


def presence_runs(windows: Iterable[tuple[datetime, int]], *, gap_sec: float = RUN_GAP_SEC
                  ) -> list[tuple[datetime, datetime, int]]:
    """(start, end, peak) per run of windows holding at least one person."""
    runs: list[list[Any]] = []
    for t, n in sorted((t, int(n)) for t, n in windows if int(n or 0) > 0):
        if runs and (t - runs[-1][1]).total_seconds() <= gap_sec:
            runs[-1][1] = t
            runs[-1][2] = max(runs[-1][2], n)
        else:
            runs.append([t, t, n])
    return [(a, b, p) for a, b, p in runs]


def within_range(estimate: int, low: int, high: int) -> int:
    """A best guess may never say fewer than were seen at once, nor more than the clusters."""
    low, high = max(0, int(low)), max(0, int(high))
    return min(max(int(estimate), low), max(high, low))


def estimate_all(*, clusters: int, most_at_once: int, stories: Sequence[Mapping[str, Any]],
                 windows: Iterable[tuple[datetime, int]], prop_ids: Iterable[str] = ()) -> dict[str, int]:
    """Every estimator for one stream-hour. ``clusters``/``most_at_once`` come from the rollup
    range (high/low); ``windows`` are (observed_at, people in that window)."""
    sc = story_count(stories, prop_ids=prop_ids)
    runs = presence_runs(windows)
    return {
        "clusters": max(0, int(clusters)),
        "most_at_once": max(0, int(most_at_once)),
        "stories": sc.total,
        "stories_in_range": within_range(sc.total, most_at_once, clusters),
        "stories_multi_only": sc.multi_frame,
        "run_peaks": sum(p for _, _, p in runs),
        "runs": len(runs),
    }
