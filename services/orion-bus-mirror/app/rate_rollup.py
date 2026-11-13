"""Five-minute bus traffic rollup (Patch 1 of the causal geometry redesign).

Pure in-memory accumulator, no I/O. The mirror feeds it every message it receives and
every cross-organ hop the ChainTracker reports. Buckets are wall-clock aligned:
bucket_start = floor(epoch / 300) * 300. `flush_closed(now)` returns every bucket that
ended at or before `now`, as `BusTrafficRollupV1` events, and forgets them.

Why it exists: the raw SQLite log keeps 24 h and the FalkorDB graph keeps only running
averages, so no bus history survived past a day. One 5-minute row per bucket is the
history the Patch 2 replay eval reads (docs/superpowers/specs/2026-11-11-causal-geometry-redesign.md).

Message counts and hop counts use the time the mirror received the message, not the
producer's timestamp, so a bucket is closed once wall clock passes its end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from orion.bus.census import normalize_channel_name
from orion.schemas.bus_traffic import (
    BUS_TRAFFIC_ROLLUP_BUCKET_SEC,
    BusTrafficHopV1,
    BusTrafficRollupV1,
)

# Channels not in the catalog keep their literal name. A producer minting one
# uncataloged channel per request would otherwise grow one bucket without bound,
# so distinct channel keys per bucket are capped; the rest are counted under this key.
OVERFLOW_CHANNEL_KEY = "__overflow__"
DEFAULT_MAX_CHANNELS_PER_BUCKET = 2000
# Same reason for organ pairs: organ ids come from envelope.source.name.
OVERFLOW_ORGAN = "__overflow__"
DEFAULT_MAX_HOPS_PER_BUCKET = 2000
# The mirror hears its own 10 s heartbeat plus every other service's, so a healthy
# subscription is never silent this long. A longer silence means the mirror was not
# receiving (Redis drop + resubscribe backoff, slow first connect), and the buckets on
# both sides of the gap are flagged `partial` instead of reading as a traffic dip.
DEFAULT_SILENCE_GAP_SEC = 30.0


def bucket_start_epoch(epoch: float, bucket_sec: int = BUS_TRAFFIC_ROLLUP_BUCKET_SEC) -> int:
    return int(epoch // bucket_sec) * bucket_sec


@dataclass
class _HopAgg:
    count: int = 0
    latency_sum_sec: float = 0.0


@dataclass
class _Bucket:
    message_count: int = 0
    decode_failures: int = 0
    channel_counts: dict[str, int] = field(default_factory=dict)
    hops: dict[tuple[str, str], _HopAgg] = field(default_factory=dict)


class TrafficRollupAccumulator:
    def __init__(
        self,
        *,
        node: str,
        catalog_names: set[str] | None = None,
        bucket_sec: int = BUS_TRAFFIC_ROLLUP_BUCKET_SEC,
        max_channels_per_bucket: int = DEFAULT_MAX_CHANNELS_PER_BUCKET,
        max_hops_per_bucket: int = DEFAULT_MAX_HOPS_PER_BUCKET,
        started_at_epoch: float | None = None,
        silence_gap_sec: float = DEFAULT_SILENCE_GAP_SEC,
    ) -> None:
        self._node = node
        self._catalog_names = catalog_names or set()
        self._bucket_sec = bucket_sec
        self._max_channels = max_channels_per_bucket
        self._max_hops = max_hops_per_bucket
        self._silence_gap_sec = silence_gap_sec
        self._buckets: dict[int, _Bucket] = {}
        # Bucket starts the mirror was not listening for in full (see DEFAULT_SILENCE_GAP_SEC).
        self._partial_starts: set[int] = set()
        # Everything before this epoch has been flushed. A message observed late (its
        # timestamp was taken before an await, and the flush ran in between) is counted in
        # the oldest still-open bucket instead of recreating an already-published one,
        # which the sql-writer would drop as a duplicate rollup_id.
        self._flushed_through: int | None = None
        # The process started part-way through a bucket: that bucket is partial.
        self._last_message_epoch: float | None = started_at_epoch
        if started_at_epoch is not None and started_at_epoch % bucket_sec != 0:
            self._partial_starts.add(bucket_start_epoch(started_at_epoch, bucket_sec))

    def __len__(self) -> int:
        return len(self._buckets)

    def _start_for(self, now_epoch: float) -> int:
        start = bucket_start_epoch(now_epoch, self._bucket_sec)
        if self._flushed_through is not None and start < self._flushed_through:
            start = self._flushed_through
        return start

    def _bucket(self, now_epoch: float) -> _Bucket:
        start = self._start_for(now_epoch)
        bucket = self._buckets.get(start)
        if bucket is None:
            bucket = self._buckets[start] = _Bucket()
        return bucket

    def observe_message(self, channel: str | None, now_epoch: float, *, decode_failed: bool = False) -> None:
        last = self._last_message_epoch
        if last is not None and now_epoch - last > self._silence_gap_sec:
            # Blind gap: flag the bucket the gap started in (if still open) and this one.
            # Buckets wholly inside the gap get no row at all.
            self._partial_starts.add(self._start_for(last))
            self._partial_starts.add(self._start_for(now_epoch))
        if last is None or now_epoch > last:
            self._last_message_epoch = now_epoch
        bucket = self._bucket(now_epoch)
        bucket.message_count += 1
        if decode_failed:
            bucket.decode_failures += 1
        name = str(channel) if channel else "__unknown__"
        if self._catalog_names:
            name = normalize_channel_name(name, self._catalog_names)
        if name not in bucket.channel_counts and len(bucket.channel_counts) >= self._max_channels:
            name = OVERFLOW_CHANNEL_KEY
        bucket.channel_counts[name] = bucket.channel_counts.get(name, 0) + 1

    def observe_hop(self, from_organ: str, to_organ: str, latency_sec: float, now_epoch: float) -> None:
        bucket = self._bucket(now_epoch)
        key = (from_organ, to_organ)
        if key not in bucket.hops and len(bucket.hops) >= self._max_hops:
            key = (OVERFLOW_ORGAN, OVERFLOW_ORGAN)
        agg = bucket.hops.get(key)
        if agg is None:
            agg = bucket.hops[key] = _HopAgg()
        agg.count += 1
        # A wall-clock step backwards can give a negative gap; clamp rather than poison the mean.
        agg.latency_sum_sec += max(float(latency_sec), 0.0)

    def flush_closed(self, now_epoch: float, *, grace_sec: float = 0.0) -> list[BusTrafficRollupV1]:
        """Pop and return every bucket whose end + `grace_sec` is at or before `now_epoch`,
        oldest first. The grace lets an in-flight message stamped just before the boundary
        land in its own bucket before that bucket is published."""
        out: list[BusTrafficRollupV1] = []
        for start in sorted(self._buckets):
            end = start + self._bucket_sec
            if end + grace_sec > now_epoch:
                break
            bucket = self._buckets.pop(start)
            out.append(self._to_event(start, bucket, partial=start in self._partial_starts))
            self._partial_starts.discard(start)
            self._flushed_through = max(self._flushed_through or end, end)
        # Partial marks for buckets that never got a message are dropped with them.
        if self._flushed_through is not None:
            self._partial_starts = {s for s in self._partial_starts if s >= self._flushed_through}
        return out

    def _to_event(self, start: int, bucket: _Bucket, *, partial: bool) -> BusTrafficRollupV1:
        bucket_start = datetime.fromtimestamp(start, tz=timezone.utc)
        bucket_end = datetime.fromtimestamp(start + self._bucket_sec, tz=timezone.utc)
        hops = [
            BusTrafficHopV1(
                from_organ=src,
                to_organ=dst,
                count=agg.count,
                latency_mean_sec=agg.latency_sum_sec / agg.count,
            )
            for (src, dst), agg in sorted(bucket.hops.items())
        ]
        return BusTrafficRollupV1(
            rollup_id=f"{self._node}:{bucket_start.isoformat()}",
            node=self._node,
            bucket_start=bucket_start,
            bucket_end=bucket_end,
            message_count=bucket.message_count,
            channel_counts=dict(sorted(bucket.channel_counts.items())),
            hops=hops,
            decode_failures=bucket.decode_failures,
            partial=partial,
        )
