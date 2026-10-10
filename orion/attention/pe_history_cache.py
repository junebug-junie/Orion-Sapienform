"""In-memory 7-day window over ``substrate_node_prediction_error_history``.

orion-substrate-runtime writes one row per real ``observed_at`` advance of a
``node:substrate.*`` node (manual_migration_node_prediction_error_history_v1.sql)
and computes magnitudes for the broadcast. orion-attention-runtime runs the
field contest in another process and has no magnitudes, so it reads the same
table (spec: "the runtime can compute them itself"). Seeded once with the
7-day window, then topped up incrementally on ``observed_at`` (indexed), with
a 30-minute overlap so a row committed late is not missed (live max lag
between observed_at and recorded_at was 355 s on 2026-10-10, so 5x margin;
recorded_at has no index, and a full scan every ~2 s tick is not worth the
last bit of certainty). The (node_id, observed_at) primary key makes the
overlap a dedupe, not a double count. The newest row per node is that node's
current reading.

Magnitudes are memoized per node on (newest reading, window size, minute):
the field contest ticks every ~2 s, the readings move every ~35 s, so only
the reading's age is refreshed in between (review 2026-10-10: 9 nodes x 15k
rows was ~120 ms per tick).
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

from orion.schemas.attention_frame import PredictionErrorMagnitudeV1
from orion.substrate.prediction_error_magnitude import (
    WINDOW_7D,
    compute_prediction_error_magnitude,
)

Row = tuple[str, datetime, float]
FetchSince = Callable[[datetime], Iterable[Row]]

# substrate-runtime records readings on its ~35 s broadcast tick, so a row's
# observed_at can trail its commit (live max 355 s).
_OVERLAP = timedelta(minutes=30)


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


class NodePeHistoryCache:
    def __init__(self) -> None:
        self._windows: dict[str, deque[tuple[datetime, float]]] = {}
        self._seen: set[tuple[str, datetime]] = set()
        self._max_observed: datetime | None = None
        self._memo: dict[str, tuple[tuple, PredictionErrorMagnitudeV1, float | None]] = {}

    @property
    def seeded(self) -> bool:
        return self._max_observed is not None

    def refresh(self, fetch_since: FetchSince, *, now: datetime) -> int:
        """Pull new rows; trim to 7 days. Returns rows added. Raises on a
        read error (the caller decides: no magnitudes this tick)."""
        now = _aware(now)
        since = (
            now - WINDOW_7D
            if self._max_observed is None
            else min(self._max_observed, now) - _OVERLAP
        )
        added = 0
        fresh: list[Row] = []
        for node_id, observed_at, value in fetch_since(since):
            observed_at = _aware(observed_at)
            key = (str(node_id), observed_at)
            if key in self._seen:
                continue
            self._seen.add(key)
            fresh.append((str(node_id), observed_at, float(value)))
        fresh.sort(key=lambda r: r[1])
        for node_id, observed_at, value in fresh:
            window = self._windows.setdefault(node_id, deque())
            if window and observed_at < window[-1][0]:
                # Late row older than the tail: insert in order (rare).
                items = sorted([*window, (observed_at, value)], key=lambda x: x[0])
                window.clear()
                window.extend(items)
            else:
                window.append((observed_at, value))
            added += 1
            if self._max_observed is None or observed_at > self._max_observed:
                self._max_observed = observed_at
        if self._max_observed is None:
            # Seeded with nothing: still mark seeded so we go incremental.
            self._max_observed = now - WINDOW_7D
        horizon = now - WINDOW_7D
        for node_id, window in self._windows.items():
            while window and window[0][0] < horizon:
                ts, _ = window.popleft()
                self._seen.discard((node_id, ts))
        return added

    def current(self) -> dict[str, tuple[float, datetime]]:
        return {
            node_id: (window[-1][1], window[-1][0])
            for node_id, window in self._windows.items()
            if window
        }

    def magnitudes(
        self, *, now: datetime
    ) -> dict[str, tuple[PredictionErrorMagnitudeV1, datetime, float | None]]:
        """node_id -> (magnitude, observed_at, mid-rank percentile)."""
        from orion.attention.world_first import midrank_percentile

        now = _aware(now)
        bucket = now.replace(second=0, microsecond=0)
        out: dict[str, tuple[PredictionErrorMagnitudeV1, datetime, float | None]] = {}
        for node_id, (value, observed_at) in self.current().items():
            window = self._windows[node_id]
            key = (observed_at, len(window), bucket)
            memo = self._memo.get(node_id)
            if memo is not None and memo[0] == key:
                age = max(0.0, (now - observed_at).total_seconds())
                out[node_id] = (memo[1].model_copy(update={"age_sec": round(age, 3)}), observed_at, memo[2])
                continue
            history = list(window)
            mag = compute_prediction_error_magnitude(
                value=value, observed_at=observed_at, history=history, now=now
            )
            mid = midrank_percentile(value, [v for ts, v in history if ts >= now - WINDOW_7D])
            self._memo[node_id] = (key, mag, mid)
            out[node_id] = (mag, observed_at, mid)
        return out
