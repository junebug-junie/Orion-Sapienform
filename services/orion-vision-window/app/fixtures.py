"""Stationary fixtures for presence: a "person" that never moves is a prop.

Live 2026-10-15 the walkway camera saw a Halloween skeleton at the patio table
as ``person`` in the same box all night, so this stream's presence row
("someone is at the walkway camera") read ``present`` with nobody there.

Per opted-in stream (``WINDOW_STATIONARY_STREAMS``) this keeps runs of
``person`` boxes (``orion.vision.stationary``): a box overlapping a run's
first box (IoU >= ``iou``), with gaps <= ``max_gap_sec``, extends it; held
``min_sec`` the run is a fixture, and stays one across gaps while its box does
not move. A window whose person boxes are ALL fixtures does not count as a
person for presence. A window with any moving person box does. A window with
no person box keeps the stream's last verdict (the belief tracker's own
hysteresis decides presence then).

Room cameras are not opted in: Juniper sitting at a desk for hours is not a prop.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from orion.vision.stationary import StillRun, advance, box_iou, is_stationary

# A fixture run nobody has seen for this long is forgotten. Runs live in
# memory only: after a window restart (or a day without the prop) the prop
# makes the stream "present" again for VISION_STATIONARY_MIN_SEC. The
# sql-writer reducer's stationary flag is persisted, so the two can briefly
# disagree; that is accepted for a presence signal.
FORGET_FIXTURE_SEC = 86400.0


class FixtureTracker:
    def __init__(
        self,
        *,
        streams: Iterable[str],
        min_sec: float,
        iou: float,
        max_gap_sec: float,
        label: str = "person",
    ) -> None:
        self.streams = {str(s).strip() for s in streams if str(s).strip()}
        self.min_sec = float(min_sec)
        self.iou = float(iou)
        self.max_gap_sec = float(max_gap_sec)
        self.label = label
        self._runs: Dict[str, List[StillRun]] = {}
        self._fixture_only: Dict[str, bool] = {}

    def enabled_for(self, stream_id: str) -> bool:
        return self.min_sec > 0 and stream_id in self.streams

    def observe_window(self, stream_id: str, frames: Sequence[tuple]) -> Optional[bool]:
        """``frames`` are ``(t, [box, ...])`` of this label, oldest first.
        Returns this window's verdict (True = only fixtures), None if the
        window had no box of the label."""
        if not self.enabled_for(stream_id):
            return None
        runs = self._runs.setdefault(stream_id, [])
        saw = False
        all_fixed = True
        for t, boxes in sorted(frames, key=lambda f: f[0]):
            runs[:] = [r for r in runs if (t - r.last) <= (FORGET_FIXTURE_SEC if self._fixed(r) else self.max_gap_sec)]
            free = list(range(len(runs)))
            for box in boxes:
                if len(box) != 4:
                    continue
                saw = True
                best, best_iou = None, self.iou
                for j in free:
                    v = box_iou(box, runs[j].box)
                    if v >= best_iou:
                        best, best_iou = j, v
                if best is None:
                    runs.append(StillRun(tuple(float(x) for x in box), t, t))
                    all_fixed = False
                    continue
                free.remove(best)
                old = runs[best]
                runs[best] = advance(old, box, t, iou_threshold=self.iou,
                                     max_gap_sec=None if self._fixed(old) else self.max_gap_sec)
                if not self._fixed(runs[best]):
                    all_fixed = False
        if not saw:
            return None
        self._fixture_only[stream_id] = all_fixed
        return all_fixed

    def _fixed(self, run: StillRun) -> bool:
        return is_stationary(run, self.min_sec)

    def is_fixture_box(self, stream_id: str, box) -> bool:
        """True when ``box`` overlaps a run that has already been held long
        enough to be a fixture (a prop), as of the last observed window."""
        if not self.enabled_for(stream_id) or len(box or []) != 4:
            return False
        return any(
            self._fixed(r) and box_iou(box, r.box) >= self.iou
            for r in self._runs.get(stream_id, [])
        )

    def fixture_only(self, stream_id: str) -> bool:
        return self.enabled_for(stream_id) and self._fixture_only.get(stream_id, False)

    def presence_labels(self, stream_id: str, believed: frozenset) -> frozenset:
        """Believed labels minus ``person`` when every person seen is a fixture."""
        if self.label in believed and self.fixture_only(stream_id):
            return frozenset(x for x in believed if x != self.label)
        return believed
