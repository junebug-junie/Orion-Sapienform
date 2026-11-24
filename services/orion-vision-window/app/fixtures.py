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

Room cameras (``VISION_STATIONARY_ROOM_STREAMS``, the office since
2026-11-24) are tracked differently: a longer hold, a gap always ends the run,
and a box with a face starts a new one (``orion.vision.stationary.StreamRule``).
Juniper sitting still at her desk must never leave her seat marked as a prop
for her next visit.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Union

from orion.vision.stationary import StillRun, StreamRule, advance, box_iou, is_stationary, stream_rules

# A fixture run nobody has seen for this long is forgotten. Runs live in
# memory only: after a window restart (or a day without the prop) the prop
# makes the stream "present" again for VISION_STATIONARY_MIN_SEC. The
# sql-writer reducer's stationary flag is persisted, so the two can briefly
# disagree; that is accepted for a presence signal.
FORGET_FIXTURE_SEC = 86400.0


def _key(stream_id: str) -> str:
    return str(stream_id or "").strip().lower()


class FixtureTracker:
    def __init__(
        self,
        *,
        streams: Union[str, Iterable[str]],
        min_sec: float,
        iou: float,
        max_gap_sec: float,
        label: str = "person",
        room_streams: Union[str, Iterable[str]] = (),
        room_min_sec: Optional[float] = None,
    ) -> None:
        self.rules: Dict[str, StreamRule] = stream_rules(streams, min_sec, room_streams, room_min_sec)
        self.streams = set(self.rules)
        self.iou = float(iou)
        self.max_gap_sec = float(max_gap_sec)
        self.label = label
        self._runs: Dict[str, List[StillRun]] = {}
        self._fixture_only: Dict[str, bool] = {}

    def enabled_for(self, stream_id: str) -> bool:
        return _key(stream_id) in self.rules

    def observe_window(self, stream_id: str, frames: Sequence[tuple]) -> Optional[bool]:
        """``frames`` are ``(t, [box, ...])`` or ``(t, [box, ...], [has_face, ...])``
        of this label, oldest first. Returns this window's verdict (True = only
        fixtures), None if the window had no box of the label."""
        if not self.enabled_for(stream_id):
            return None
        stream_id = _key(stream_id)
        rule = self.rules[stream_id]
        runs = self._runs.setdefault(stream_id, [])
        saw = False
        all_fixed = True
        for frame in sorted(frames, key=lambda f: f[0]):
            t, boxes = frame[0], frame[1]
            faces = list(frame[2]) if len(frame) > 2 and frame[2] is not None else []
            # A marked outdoor prop is remembered for a day; anything else
            # (and every run on a room camera) is forgotten after one gap.
            runs[:] = [r for r in runs if (t - r.last) <= (FORGET_FIXTURE_SEC if rule.gap_sec(r, self.max_gap_sec) is None
                                                          else self.max_gap_sec)]
            free = list(range(len(runs)))
            for i, box in enumerate(boxes):
                if len(box) != 4:
                    continue
                face = rule.room and i < len(faces) and bool(faces[i])
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
                                     max_gap_sec=rule.gap_sec(old, self.max_gap_sec), restart=face)
                if not self._fixed(stream_id, runs[best]):
                    all_fixed = False
        if not saw:
            return None
        self._fixture_only[stream_id] = all_fixed
        return all_fixed

    def _fixed(self, stream_id: str, run: StillRun) -> bool:
        rule = self.rules.get(stream_id)
        return rule is not None and is_stationary(run, rule.min_sec)

    def is_fixture_box(self, stream_id: str, box) -> bool:
        """True when ``box`` overlaps a run that has already been held long
        enough to be a fixture (a prop), as of the last observed window."""
        if not self.enabled_for(stream_id) or len(box or []) != 4:
            return False
        key = _key(stream_id)
        return any(
            self._fixed(key, r) and box_iou(box, r.box) >= self.iou
            for r in self._runs.get(key, [])
        )

    def fixture_only(self, stream_id: str) -> bool:
        return self.enabled_for(stream_id) and self._fixture_only.get(_key(stream_id), False)

    def presence_labels(self, stream_id: str, believed: frozenset) -> frozenset:
        """Believed labels minus ``person`` when every person seen is a fixture."""
        if self.label in believed and self.fixture_only(stream_id):
            return frozenset(x for x in believed if x != self.label)
        return believed
