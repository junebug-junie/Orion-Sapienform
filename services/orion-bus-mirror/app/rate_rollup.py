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
    ) -> None:
        self._node = node
        # The bucket the process started in saw only part of its 5 minutes. Emitting it
        # would store an undercount that reads as a real traffic dip, so it is dropped.
        self._first_full_bucket = (
            None
            if started_at_epoch is None
            else bucket_start_epoch(started_at_epoch, bucket_sec) + (
                0 if started_at_epoch % bucket_sec == 0 else bucket_sec
            )
        )
        self._catalog_names = catalog_names or set()
        self._bucket_sec = bucket_sec
        self._max_channels = max_channels_per_bucket
        self._max_hops = max_hops_per_bucket
        self._buckets: dict[int, _Bucket] = {}

    def __len__(self) -> int:
        return len(self._buckets)

    def _bucket(self, now_epoch: float) -> _Bucket:
        start = bucket_start_epoch(now_epoch, self._bucket_sec)
        bucket = self._buckets.get(start)
        if bucket is None:
            bucket = self._buckets[start] = _Bucket()
        return bucket

    def observe_message(self, channel: str | None, now_epoch: float, *, decode_failed: bool = False) -> None:
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

    def flush_closed(self, now_epoch: float) -> list[BusTrafficRollupV1]:
        """Pop and return every bucket whose end is at or before `now_epoch`, oldest first."""
        out: list[BusTrafficRollupV1] = []
        for start in sorted(self._buckets):
            if start + self._bucket_sec > now_epoch:
                break
            bucket = self._buckets.pop(start)
            if self._first_full_bucket is not None and start < self._first_full_bucket:
                continue  # partial startup bucket: dropped, see __init__
            out.append(self._to_event(start, bucket))
        return out

    def _to_event(self, start: int, bucket: _Bucket) -> BusTrafficRollupV1:
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
        )
