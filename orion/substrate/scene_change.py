"""Camera scene change: a two-stage rank of how much a camera's picture changed. Pure, stdlib only.

Design: docs/superpowers/specs/2026-11-24-pe-retire-perception-and-alarms-design.md rev 4
(PR #3827), item 3. Replaces the clipped EWMA z-score that ``perception_prediction_error()``
returns (``clip(z, 0, 3) / 3`` with a 2e-6 variance floor that set the scale instead of the data).
Not a prediction error: nothing here forecasts anything. It answers "how unusual is the change in
this camera's picture over the last 5 minutes, relative to the last 2 hours of the same camera,
ranked against its own last 24 hours".

Input: the raw per-frame magnitude ``1 - cos(frame, stream EWMA)`` (``PerceptionPredictionErrorResult
.raw_surprise``), unchanged.

Stage 1, per frame, per stream. p1 = (1 + #{past >= x}) / (n + 1) over the same stream's frames in
the last ``STAGE1_WINDOW_SEC`` (2 h of time, so a frame-rate change does not move the window).
A stream is ready once it holds ``STAGE1_MIN_FRAMES`` frames spanning ``STAGE1_MIN_SPAN_SEC``.

Stage 2, per node, every tick. m = the smallest p1 over the node's frames in the trailing
``STAGE2_INTERVAL_SEC`` (5 min). p2 = (1 + #{past m <= m}) / (n + 1) against the node's own m
values from the last 24 h, sampled once every 5 min (at most ``STAGE2_MAX_VALUES`` = 288).
Value = excess = max(0, -ln p2 - 1), in nats.

Why the trailing 5 min: the history sampler and the attention contest read the node about every
4-5 min. A value that covered only the last 30 s missed 86-92% of short changes in replay.

Calm point by theory: under exchangeable 5-min intervals p2 is uniform, so -ln p2 ~ Exp(1);
excess is exactly 0 with probability 1 - 1/e = 0.632 and crosses 2.0 nats with probability
e^-3 = 0.050. The cap is ln(289) - 1 = 4.67 nats (``SCENE_CHANGE_CAP_NATS``).

Cold start: until stage 1 is ready and stage 2 holds ``STAGE2_MIN_VALUES`` (48 = 4 h) values the
reading is missing (``warming``), never a p = 1 "calm". While stage 2 holds less than 24 h of
values the reading's scale is ``v2_partial`` (read by the contests; excluded from calm checks and
win share); after that ``v2``.
"""
from __future__ import annotations

import bisect
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from orion.prediction.signal_history_sql import SCALE_V2, SCALE_V2_PARTIAL

STAGE1_WINDOW_SEC = 2 * 3600.0
STAGE1_MIN_FRAMES = 200
STAGE1_MIN_SPAN_SEC = 30 * 60.0
STAGE2_INTERVAL_SEC = 5 * 60.0
STAGE2_WINDOW_SEC = 24 * 3600.0
STAGE2_MAX_VALUES = 288
STAGE2_MIN_VALUES = 48
# The stage-2 window "holds 24 h" once its oldest value is this old (one 5-min sample of slack,
# so the 288th sample counts).
STAGE2_FULL_SPAN_SEC = STAGE2_WINDOW_SEC - STAGE2_INTERVAL_SEC
# Largest reachable value: p2 = 1 / (288 + 1).
SCENE_CHANGE_CAP_NATS = math.log(STAGE2_MAX_VALUES + 1) - 1.0

MISSING_WARMING = "warming"
MISSING_NO_FRAME = "no_scored_frame_since_last_tick"


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def excess_nats(p: float) -> float:
    """max(0, -ln p - 1): 0 for p >= 1/e, positive for rarer p."""
    if p <= 0.0:
        raise ValueError("p must be > 0")
    return max(0.0, -math.log(p) - 1.0)


class Stage1Window:
    """One stream's raw frame magnitudes over the last 2 h, kept sorted for ranking."""

    def __init__(self) -> None:
        self._frames: deque[tuple[datetime, float]] = deque()
        self._sorted: list[float] = []

    def __len__(self) -> int:
        return len(self._frames)

    def _prune(self, now: datetime) -> None:
        horizon = now - timedelta(seconds=STAGE1_WINDOW_SEC)
        while self._frames and self._frames[0][0] < horizon:
            _, old = self._frames.popleft()
            i = bisect.bisect_left(self._sorted, old)
            if i < len(self._sorted) and self._sorted[i] == old:
                del self._sorted[i]

    def ready(self, now: datetime) -> bool:
        if len(self._frames) < STAGE1_MIN_FRAMES:
            return False
        return (now - self._frames[0][0]).total_seconds() >= STAGE1_MIN_SPAN_SEC

    def observe(self, ts: datetime, raw: float) -> float | None:
        """Rank ``raw`` against the window (before adding it); None while not ready."""
        ts = _aware(ts)
        raw = float(raw)
        self._prune(ts)
        p: float | None = None
        if self.ready(ts):
            n = len(self._sorted)
            at_or_above = n - bisect.bisect_left(self._sorted, raw)
            p = (1.0 + at_or_above) / (n + 1.0)
        self._frames.append((ts, raw))
        bisect.insort(self._sorted, raw)
        return p


@dataclass(frozen=True)
class SceneChangeReading:
    """One tick's stage-2 result. ``value`` is None exactly when ``missing_reason`` is set."""

    value: float | None
    missing_reason: str | None
    scale: str | None
    m: float | None = None
    p2: float | None = None
    n_stage2: int = 0
    n_recent: int = 0


@dataclass
class SceneChangeScorer:
    """Per camera node: a stage-1 window per stream and the node's stage-2 history.

    Not thread-safe; the caller holds a lock (the frame listener and the tick run on different
    threads)."""

    streams: dict[str, Stage1Window] = field(default_factory=dict)
    recent: deque = field(default_factory=deque)  # (ts, p1) over the trailing 5 min
    stage2: deque = field(default_factory=deque)  # (ts, m), last 24 h, at most 288
    last_stage2_at: datetime | None = None
    restored: bool = False

    def observe(self, stream_id: str, ts: datetime, raw: float) -> float | None:
        """Fold one frame; returns its stage-1 p (None while that stream is warming)."""
        window = self.streams.get(stream_id)
        if window is None:
            window = self.streams[stream_id] = Stage1Window()
        ts = _aware(ts)
        p = window.observe(ts, raw)
        if p is not None:
            self.recent.append((ts, p))
        return p

    def stage1_ready(self, now: datetime) -> bool:
        return any(w.ready(now) for w in self.streams.values())

    def _prune(self, now: datetime) -> None:
        recent_horizon = now - timedelta(seconds=STAGE2_INTERVAL_SEC)
        while self.recent and self.recent[0][0] < recent_horizon:
            self.recent.popleft()
        stage2_horizon = now - timedelta(seconds=STAGE2_WINDOW_SEC)
        while self.stage2 and self.stage2[0][0] < stage2_horizon:
            self.stage2.popleft()
        while len(self.stage2) > STAGE2_MAX_VALUES:
            self.stage2.popleft()

    def stage2_scale(self, now: datetime) -> str:
        if self.stage2 and (now - self.stage2[0][0]).total_seconds() >= STAGE2_FULL_SPAN_SEC:
            return SCALE_V2
        return SCALE_V2_PARTIAL

    def tick(self, now: datetime) -> tuple[SceneChangeReading, bool]:
        """(reading, sampled): ``sampled`` is True when this tick added a stage-2 value (the
        caller persists the stage-2 history then)."""
        now = _aware(now)
        self._prune(now)
        if not self.recent:
            # Warming until both stages could score; after that an empty interval is "no frame".
            warm = self.stage1_ready(now) and len(self.stage2) >= STAGE2_MIN_VALUES
            reason = MISSING_NO_FRAME if warm else MISSING_WARMING
            return SceneChangeReading(None, reason, None, n_stage2=len(self.stage2)), False
        m = min(p for _, p in self.recent)
        n2 = len(self.stage2)
        p2: float | None = None
        value: float | None = None
        if n2 >= STAGE2_MIN_VALUES:
            at_or_below = sum(1 for _, past in self.stage2 if past <= m)
            p2 = (1.0 + at_or_below) / (n2 + 1.0)
            value = excess_nats(p2)
        scale = self.stage2_scale(now)
        sampled = False
        if self.last_stage2_at is None or (now - self.last_stage2_at).total_seconds() >= STAGE2_INTERVAL_SEC:
            # Ranked above before it joins the window: never against itself.
            self.stage2.append((now, m))
            self.last_stage2_at = now
            self._prune(now)
            sampled = True
        if value is None:
            return SceneChangeReading(
                None, MISSING_WARMING, None, m=m, n_stage2=n2, n_recent=len(self.recent)
            ), sampled
        return SceneChangeReading(
            value, None, scale, m=m, p2=p2, n_stage2=n2, n_recent=len(self.recent)
        ), sampled

    # --- stage-2 persistence (substrate_perception_embedding_baseline, one row per node) -------

    def stage2_to_json(self) -> dict[str, Any]:
        return {
            "version": 1,
            "values": [[ts.isoformat(), m] for ts, m in self.stage2],
            "last_sample_at": self.last_stage2_at.isoformat() if self.last_stage2_at else None,
        }

    def restore_stage2(self, payload: Any, *, now: datetime) -> int:
        """Load saved stage-2 values (dropping any older than 24 h). Returns how many loaded.
        A malformed payload loads nothing (a cold stage 2 is an honest state)."""
        self.restored = True
        if not isinstance(payload, dict) or payload.get("version") != 1:
            return 0
        values: list[tuple[datetime, float]] = []
        for item in payload.get("values") or []:
            try:
                ts = _aware(datetime.fromisoformat(str(item[0])))
                m = float(item[1])
            except (TypeError, ValueError, IndexError):
                continue
            if not (0.0 < m <= 1.0) or ts > now:
                continue
            values.append((ts, m))
        values.sort(key=lambda v: v[0])
        self.stage2 = deque(values)
        last = payload.get("last_sample_at")
        try:
            self.last_stage2_at = _aware(datetime.fromisoformat(str(last))) if last else None
        except ValueError:
            self.last_stage2_at = None
        if self.last_stage2_at is None and values:
            self.last_stage2_at = values[-1][0]
        self._prune(_aware(now))
        return len(self.stage2)


def stage2_store_key(node_id: str) -> str:
    """The stream_id the stage-2 history is saved under in substrate_perception_embedding_baseline.
    Never a camera name (camera names carry no '|')."""
    return f"scene_change_stage2|{node_id}"
