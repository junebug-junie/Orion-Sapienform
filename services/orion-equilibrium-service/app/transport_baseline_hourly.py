"""Hourly per-hop summaries of the transport baseline gate (durable readings).

The gate's per-window readings used to live only in ``transport_baseline_obs``
log lines, lost on every restart. This accumulator folds every ``FoldResult``
into one bucket per (service, instance, hop, UTC hour of the snapshot's
``window_end``) and turns closed buckets into ``TransportBaselineHourlyV1`` rows
that equilibrium publishes and sql-writer persists.

Pure: no clock reads, no I/O. The caller passes ``now`` (wall clock) to
``flush_due`` and publishes what it returns.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from orion.schemas.telemetry.transport_baseline_hourly import TransportBaselineHourlyV1

logger = logging.getLogger("orion.equilibrium.transport_baseline_hourly")

# Per-key samples kept per hour: 30 s windows -> 120 per hour. Bounded anyway so
# a burst of producers or a clock jump cannot grow a bucket without limit.
MAX_SAMPLES = 512
# Buckets held at once. ~64 keys live; a runaway key set is cut, not grown.
MAX_BUCKETS = 4096


def hour_start_of(ts: float) -> float:
    return math.floor(ts / 3600.0) * 3600.0


def _pct(values: list[float], q: float) -> float | None:
    """Nearest-rank-with-interpolation percentile; None on no data."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


@dataclass
class _Bucket:
    service: str
    instance: str | None
    key: str
    hour_start: float
    windows_seen: int = 0
    windows_evaluated: int = 0
    success_count: int = 0
    timeout_count: int = 0
    z: list[float] = field(default_factory=list)
    ratio: list[float] = field(default_factory=list)
    calls_sum: float = 0.0
    baseline_ms: float | None = None
    floor_ms_start: float | None = None
    floor_ms: float | None = None
    opened: dict[str, int] = field(default_factory=dict)
    open_now: tuple[str, ...] = ()
    would_emit: dict[str, int] = field(default_factory=dict)
    excluded: bool = False
    warm: bool = False
    warm_at_start: bool | None = None
    late: bool = False


class TransportBaselineHourly:
    def __init__(self, *, config_fingerprint: str, flush_grace_s: float = 90.0) -> None:
        self.config_fingerprint = config_fingerprint
        self.flush_grace_s = float(flush_grace_s)
        self._buckets: dict[tuple[str, str | None, str, float], _Bucket] = {}
        # Buckets already flushed (hour_end/shutdown): a later fold for one of
        # them becomes a separate "late" row instead of silently reopening it.
        self._closed: dict[tuple[str, str | None, str, float], float] = {}
        self.dropped_buckets = 0
        self._dropped_reported = 0

    def _bucket(self, service: str, instance: str | None, key: str, hour: float) -> _Bucket | None:
        bk = (service, instance, key, hour)
        b = self._buckets.get(bk)
        if b is None:
            if len(self._buckets) >= MAX_BUCKETS:
                self.dropped_buckets += 1
                return None
            b = _Bucket(service=service, instance=instance, key=key, hour_start=hour, late=bk in self._closed)
            self._buckets[bk] = b
        return b

    def observe(self, result: Any, *, window_end_ts: float) -> None:
        """Fold one gate ``FoldResult`` (observations + events) in."""
        hour = hour_start_of(window_end_ts)
        for ob in getattr(result, "observations", ()) or ():
            b = self._bucket(ob.service, ob.instance, ob.key, hour)
            if b is None:
                continue
            b.windows_seen += 1
            b.success_count += int(ob.success_count or 0)
            b.timeout_count += int(ob.timeout_count or 0)
            b.calls_sum += float(ob.calls_per_min or 0.0)
            if ob.evaluated:
                b.windows_evaluated += 1
                if ob.z is not None and len(b.z) < MAX_SAMPLES:
                    b.z.append(float(ob.z))
                if ob.saturation_ratio is not None and len(b.ratio) < MAX_SAMPLES:
                    b.ratio.append(float(ob.saturation_ratio))
            if ob.baseline_ms is not None:
                b.baseline_ms = float(ob.baseline_ms)
            if ob.floor_ms is not None:
                if b.floor_ms_start is None:
                    b.floor_ms_start = float(ob.floor_ms)
                b.floor_ms = float(ob.floor_ms)
            b.open_now = tuple(ob.open_conditions)
            b.excluded = bool(ob.excluded)
            b.warm = bool(ob.warm)
            if b.warm_at_start is None:
                b.warm_at_start = bool(ob.warm)
        for ev in getattr(result, "events", ()) or ():
            b = self._bucket(ev.service, ev.instance, ev.key, hour)
            if b is None:
                continue
            b.excluded = b.excluded or bool(ev.excluded)
            if ev.phase == "open":
                b.opened[ev.condition] = b.opened.get(ev.condition, 0) + 1
            if not ev.excluded:
                k = f"{ev.condition}:{ev.phase}"
                b.would_emit[k] = b.would_emit.get(k, 0) + 1

    def _row(self, b: _Bucket, *, reason: str, now: float, emit_effective: bool) -> TransportBaselineHourlyV1:
        return TransportBaselineHourlyV1(
            summary_id=uuid4().hex,
            service=b.service,
            instance=b.instance,
            key=b.key,
            hour_start=datetime.fromtimestamp(b.hour_start, tz=timezone.utc),
            flush_reason=("late" if b.late else reason),  # type: ignore[arg-type]
            flushed_at=datetime.fromtimestamp(now, tz=timezone.utc),
            windows_seen=b.windows_seen,
            windows_evaluated=b.windows_evaluated,
            success_count=b.success_count,
            timeout_count=b.timeout_count,
            z_p50=_pct(b.z, 0.5),
            z_p90=_pct(b.z, 0.9),
            saturation_ratio_p50=_pct(b.ratio, 0.5),
            baseline_ms=b.baseline_ms,
            floor_ms_start=b.floor_ms_start,
            floor_ms=b.floor_ms,
            calls_per_min_mean=(b.calls_sum / b.windows_seen) if b.windows_seen else 0.0,
            conditions_opened=dict(b.opened),
            open_at_hour_end=list(b.open_now),
            would_emit_by_condition=dict(b.would_emit),
            excluded=b.excluded,
            warm=b.warm,
            warm_at_start=bool(b.warm_at_start),
            emit_effective=emit_effective,
            config_fingerprint=self.config_fingerprint,
        )

    def flush_due(self, now: float, *, emit_effective: bool) -> list[TransportBaselineHourlyV1]:
        """Rows for every bucket whose hour ended more than ``flush_grace_s``
        ago (the grace lets the hour's last 30 s window land first)."""
        due = [k for k, b in self._buckets.items() if now >= b.hour_start + 3600.0 + self.flush_grace_s]
        rows = [
            self._row(self._buckets.pop(k), reason="hour_end", now=now, emit_effective=emit_effective)
            for k in sorted(due, key=lambda k: (k[3], k[0], k[1] or "", k[2]))
        ]
        for k in due:
            self._closed[k] = now
        # Remember closed hours for two days; older late folds are vanishingly rare.
        self._closed = {k: t for k, t in self._closed.items() if now - t < 2 * 86400.0}
        self._report_dropped()
        return rows

    def _report_dropped(self) -> None:
        if self.dropped_buckets > self._dropped_reported:
            logger.warning(
                "transport_baseline_hourly bucket cap %d reached: %d observation bucket(s) dropped so far",
                MAX_BUCKETS, self.dropped_buckets,
            )
            self._dropped_reported = self.dropped_buckets

    def flush_all(self, now: float, *, emit_effective: bool) -> list[TransportBaselineHourlyV1]:
        """Shutdown: every open bucket, partial hours included."""
        keys = sorted(self._buckets, key=lambda k: (k[3], k[0], k[1] or "", k[2]))
        self._report_dropped()
        return [
            self._row(self._buckets.pop(k), reason="shutdown", now=now, emit_effective=emit_effective)
            for k in keys
        ]

    @property
    def bucket_count(self) -> int:
        return len(self._buckets)
