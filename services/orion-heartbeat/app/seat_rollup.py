"""Five-minute per-seat rollup of the heartbeat's routed atoms (issue #3886).

Pure in-memory accumulator, no I/O. Same shape as orion-bus-mirror's app/rate_rollup.py (#3282).

What it sums: for each seat and each 5-minute bucket, the ``route_atom`` level of every atom the
heartbeat routes to that seat, the number of those atoms, and the number of atoms from that seat's
services that the input rule skipped. Atoms are bucketed by the grammar event's ``emitted_at``,
the same column scripts/demon_network_replay.py buckets ``grammar_events`` by, so the two agree.
Levels are summed with ``math.fsum``: the sum is exactly rounded, so it does not depend on the
order the atoms arrived in.

What counts as missing (absence must read as unknown, never as zero):

- A bucket the heartbeat never heard a message in has no rows at all.
- The bucket the process started in is ``partial``.
- Every bucket between losing the orion:grammar:event subscription and getting it back (the
  chassis restarts ``_run`` after a bus drop) is ``partial`` (``mark_blind``).
- A silence on orion:grammar:event longer than ``silence_gap_sec`` marks every bucket from the
  last message to the next one ``partial``. A bucket flushed while that silence is still going on
  (nothing heard within ``silence_gap_sec`` of its end) is ``partial`` too. The channel carries bus atoms every ~11 s and
  biometrics atoms every ~16 s, so a long silence means the heartbeat was not receiving.
- The open bucket at shutdown is never published.

A seat with no atoms in a complete bucket gets a row with level 0 and count 0: the heartbeat was
listening and that seat was quiet.

An atom stamped in a bucket that was already published (or before the process started) cannot be
added without rewriting a stored row, so it is dropped and counted as ``late`` (also per bucket in
``late_by_bucket``, so a stored row that disagrees with grammar_events can be traced). An atom stamped more
than one bucket in the future (clock skew) is dropped and counted as ``future``.

A repeated ``event_id`` is counted once, in the bucket of its first copy, also when the copies
fall in different buckets (issue #3900). The ledger does the same: orion/grammar/ledger.py
``_bulk_insert_events`` inserts with ``on_conflict_do_nothing(event_id)``, so the first write wins.
orion-harness-governor sends ``exec_result_assembled`` twice with one event_id: at motor close and
again at finalize, seconds to minutes later, with the finalize time as emitted_at
(orion/harness/grammar_emit.py ``build_harness_grammar_finalize_events``). When the two copies
straddle a bucket boundary, a per-bucket dedup counted the second copy in the later bucket. Event
ids are kept for ``DEDUP_RETENTION_SEC`` after their bucket closes.

Arrival lag (issue #3901): for every first copy with a real emitted_at, the receive time minus
emitted_at is kept per seat (the last ``LAG_WINDOW`` atoms) for p50/p95/max on /health. A row's
``late_count`` is the number of its routed atoms that arrived after the bucket ended (inside the
flush grace). Atoms that arrive after the row was published are dropped and counted on /health
(``late``, ``late_by_bucket``, ``late_by_seat``), since a stored row is never rewritten.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional, Sequence

from orion.schemas.heartbeat_seat_rollup import (
    HEARTBEAT_SEAT_ROLLUP_BUCKET_SEC,
    HeartbeatSeatRollupV1,
)

# Backstop for a subscription that hangs without an error (a resubscribe is caught by mark_blind).
# The channel's 99.99th-percentile gap is ~11 s, but bus and biometrics pause together for 30-60 s
# a few times a day (46 s on 2026-11-25 02:28:48). The ledger shows those pauses too, so they are
# real data: 30 s would mark them missing and cut the replay's scored run. 60 s does not.
DEFAULT_SILENCE_GAP_SEC = 60.0
# Buckets kept in late_by_bucket (4 h).
LATE_BY_BUCKET_MAX = 48
# Event ids are remembered this long after their bucket closes. The harness finalize copy comes
# 20 s to ~2.5 min after the motor copy (11-25 harness-governor logs), so 1 h is wide.
DEDUP_RETENTION_SEC = 3600
# Hard cap on remembered event ids (about 7,000 routed-service atoms an hour on 11-25).
DEDUP_MAX_IDS = 200_000
# Arrival lags kept per seat for the /health percentiles.
LAG_WINDOW = 4096

Outcome = Literal["counted", "duplicate", "late", "future"]


def bucket_start_epoch(epoch: float, bucket_sec: int = HEARTBEAT_SEAT_ROLLUP_BUCKET_SEC) -> int:
    return int(math.floor(epoch / bucket_sec)) * bucket_sec


@dataclass
class _Bucket:
    n_seats: int
    levels: list[list[float]] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)
    late: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.levels = [[] for _ in range(self.n_seats)]
        self.counts = [0] * self.n_seats
        self.skipped = [0] * self.n_seats
        self.late = [0] * self.n_seats


def _nearest_rank(sorted_values: list[float], q: float) -> float:
    return sorted_values[max(0, math.ceil(q * len(sorted_values)) - 1)]


class SeatRollupAccumulator:
    def __init__(
        self,
        *,
        node: str,
        seats: Sequence[str],
        started_at_epoch: float,
        routing_version: str,
        bucket_sec: int = HEARTBEAT_SEAT_ROLLUP_BUCKET_SEC,
        silence_gap_sec: float = DEFAULT_SILENCE_GAP_SEC,
    ) -> None:
        self._node = node
        self._routing_version = routing_version
        self._seats = tuple(seats)
        self._bucket_sec = bucket_sec
        self._silence_gap_sec = silence_gap_sec
        self._buckets: dict[int, _Bucket] = {}
        start = bucket_start_epoch(started_at_epoch, bucket_sec)
        # Buckets before this start are closed: already published, or before the process started.
        self._open_from = start
        self._partial_starts: set[int] = set()
        if started_at_epoch != start:
            self._partial_starts.add(start)
        self._last_receipt_epoch = float(started_at_epoch)
        self.late = 0
        self.late_by_bucket: dict[int, int] = {}
        self.late_by_seat = [0] * len(self._seats)
        self.future = 0
        self.duplicates = 0
        # Duplicates whose first copy was in another bucket (the #3900 case).
        self.duplicates_cross_bucket = 0
        # event_id -> bucket start of its first copy, in arrival order.
        self._seen: dict[str, int] = {}
        self._lags: list[deque[float]] = [deque(maxlen=LAG_WINDOW) for _ in self._seats]

    def __len__(self) -> int:
        return len(self._buckets)

    def _bucket(self, start: int) -> _Bucket:
        bucket = self._buckets.get(start)
        if bucket is None:
            bucket = self._buckets[start] = _Bucket(n_seats=len(self._seats))
        return bucket

    def mark_blind(self, from_epoch: float, to_epoch: float) -> None:
        """The heartbeat was not subscribed from ``from_epoch`` to ``to_epoch``: every bucket that
        span touches is partial. Buckets already published are not touched."""
        first = max(bucket_start_epoch(from_epoch, self._bucket_sec), self._open_from)
        end = bucket_start_epoch(to_epoch, self._bucket_sec)
        for start in range(first, end + 1, self._bucket_sec):
            self._partial_starts.add(start)

    def observe_receipt(self, now_epoch: float) -> None:
        """Any message received on orion:grammar:event. Opens the bucket and detects blind gaps."""
        last = self._last_receipt_epoch
        if now_epoch - last > self._silence_gap_sec:
            self.mark_blind(last, now_epoch)
        if now_epoch > last:
            self._last_receipt_epoch = now_epoch
        start = bucket_start_epoch(now_epoch, self._bucket_sec)
        if start >= self._open_from:
            self._bucket(start)

    def _place(
        self, site_index: int, emitted_epoch: float, now_epoch: float, event_id: Optional[str], record_lag: bool
    ) -> tuple[Outcome, Optional[_Bucket]]:
        start = bucket_start_epoch(emitted_epoch, self._bucket_sec)
        if event_id:
            first = self._seen.get(event_id)
            if first is not None:
                self.duplicates += 1
                if first != start:
                    self.duplicates_cross_bucket += 1
                return "duplicate", None
            # Remembered also when this copy is late or future: the ledger keeps this copy, so a
            # later copy must not be counted either.
            self._seen[event_id] = start
            if len(self._seen) > DEDUP_MAX_IDS:
                del self._seen[next(iter(self._seen))]
        if record_lag:
            self._lags[site_index].append(now_epoch - emitted_epoch)
        if start < self._open_from:
            self.late += 1
            self.late_by_bucket[start] = self.late_by_bucket.get(start, 0) + 1
            while len(self.late_by_bucket) > LATE_BY_BUCKET_MAX:
                self.late_by_bucket.pop(min(self.late_by_bucket))
            self.late_by_seat[site_index] += 1
            return "late", None
        if start > bucket_start_epoch(now_epoch, self._bucket_sec) + self._bucket_sec:
            self.future += 1
            return "future", None
        return "counted", self._bucket(start)

    def observe_atom(
        self,
        *,
        site_index: int,
        level: float,
        emitted_epoch: float,
        now_epoch: float,
        event_id: Optional[str] = None,
        record_lag: bool = True,
    ) -> Outcome:
        """``record_lag=False`` when emitted_epoch is not the atom's real emitted_at (fallback)."""
        outcome, bucket = self._place(site_index, emitted_epoch, now_epoch, event_id, record_lag)
        if bucket is not None:
            bucket.levels[site_index].append(float(level))
            bucket.counts[site_index] += 1
            if now_epoch >= bucket_start_epoch(emitted_epoch, self._bucket_sec) + self._bucket_sec:
                bucket.late[site_index] += 1
        return outcome

    def observe_skip(
        self,
        *,
        site_index: int,
        emitted_epoch: float,
        now_epoch: float,
        event_id: Optional[str] = None,
        record_lag: bool = True,
    ) -> Outcome:
        outcome, bucket = self._place(site_index, emitted_epoch, now_epoch, event_id, record_lag)
        if bucket is not None:
            bucket.skipped[site_index] += 1
        return outcome

    def flush_closed(self, now_epoch: float, *, grace_sec: float = 0.0) -> list[HeartbeatSeatRollupV1]:
        """Pop every bucket whose end + ``grace_sec`` is at or before ``now_epoch``, oldest first,
        as one event per seat. The grace lets an atom stamped just before the boundary arrive."""
        out: list[HeartbeatSeatRollupV1] = []
        for start in sorted(self._buckets):
            end = start + self._bucket_sec
            if end + grace_sec > now_epoch:
                break
            bucket = self._buckets.pop(start)
            # Nothing heard for longer than the gap before the bucket ended (the subscription
            # went quiet and no later message has arrived to close the gap): the tail is blind.
            blind_tail = self._last_receipt_epoch < end - self._silence_gap_sec
            out.extend(self._to_events(start, bucket, partial=blind_tail or start in self._partial_starts))
            self._open_from = max(self._open_from, end)
        self._partial_starts = {s for s in self._partial_starts if s >= self._open_from}
        if out:
            keep_from = self._open_from - DEDUP_RETENTION_SEC
            self._seen = {k: v for k, v in self._seen.items() if v >= keep_from}
        return out

    def lag_stats(self) -> dict[str, dict[str, Optional[float]]]:
        """Arrival lag (receive time minus emitted_at, seconds) per seat over the last LAG_WINDOW
        first copies. A seat with no samples reads null, not 0."""
        out: dict[str, dict[str, Optional[float]]] = {}
        for seat, lags in zip(self._seats, self._lags):
            values = sorted(lags)
            if not values:
                out[seat] = {"n": 0, "p50_sec": None, "p95_sec": None, "max_sec": None}
                continue
            out[seat] = {
                "n": len(values),
                "p50_sec": round(_nearest_rank(values, 0.50), 3),
                "p95_sec": round(_nearest_rank(values, 0.95), 3),
                "max_sec": round(values[-1], 3),
            }
        return out

    def _to_events(self, start: int, bucket: _Bucket, *, partial: bool) -> list[HeartbeatSeatRollupV1]:
        bucket_start = datetime.fromtimestamp(start, tz=timezone.utc)
        bucket_end = datetime.fromtimestamp(start + self._bucket_sec, tz=timezone.utc)
        return [
            HeartbeatSeatRollupV1(
                rollup_id=f"{self._node}:{seat}:{bucket_start.isoformat()}",
                node=self._node,
                seat=seat,
                site_index=i,
                bucket_start=bucket_start,
                bucket_end=bucket_end,
                level_sum=math.fsum(bucket.levels[i]),
                atom_count=bucket.counts[i],
                skipped_count=bucket.skipped[i],
                late_count=bucket.late[i],
                partial=partial,
                routing_version=self._routing_version,
            )
            for i, seat in enumerate(self._seats)
        ]


def parse_emitted_epoch(value: object) -> Optional[float]:
    """GrammarEventV1.emitted_at as it arrives on the bus (ISO string), as epoch seconds. A naive
    timestamp is read as UTC. None when missing or unparseable."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()
