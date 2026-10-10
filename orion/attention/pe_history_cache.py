"""In-memory 7-day window over ``substrate_node_prediction_error_history``.

orion-substrate-runtime writes one row per real ``observed_at`` advance of a
``node:substrate.*`` node (manual_migration_node_prediction_error_history_v1.sql)
and computes magnitudes for the broadcast. orion-attention-runtime runs the
field contest in another process and has no magnitudes, so it reads the same
table (spec: "the runtime can compute them itself"). Seeded once with the
7-day window, then topped up incrementally on ``observed_at`` (indexed), with
an overlap so a row committed slightly late is not missed; the
(node_id, observed_at) primary key makes the overlap a dedupe, not a double
count. The newest row per node is that node's current reading.
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
# observed_at can trail its commit by a tick or two.
_OVERLAP = timedelta(minutes=10)


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


class NodePeHistoryCache:
    def __init__(self) -> None:
        self._windows: dict[str, deque[tuple[datetime, float]]] = {}
        self._seen: set[tuple[str, datetime]] = set()
        self._max_observed: datetime | None = None

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
    ) -> dict[str, tuple[PredictionErrorMagnitudeV1, datetime]]:
        out: dict[str, tuple[PredictionErrorMagnitudeV1, datetime]] = {}
        for node_id, (value, observed_at) in self.current().items():
            out[node_id] = (
                compute_prediction_error_magnitude(
                    value=value,
                    observed_at=observed_at,
                    history=list(self._windows[node_id]),
                    now=now,
                ),
                observed_at,
            )
        return out
