"""In-process RPC health aggregator for OrionBusAsync.rpc_request().

Step 2 of docs/superpowers/specs/2026-07-23-transport-domain-rpc-health-redesign.md's
"Recommended next patch", after step 1 (worker-path logging fix + real baseline
measurement, PR #1290) and the "gated investigation" (PR #1299: confirmed no silent
uncaught-exception blind spot in `rpc_request()`'s outcome set, confirmed real
cross-service volume/variance, formally benchmarked the log line's overhead as
negligible -- +9.8us/call, 0.0135% of the fastest real observed RPC call).

**Scope of this patch, deliberately narrow:** this builds the in-memory accumulator
only -- `record_success()`/`record_timeout()` called synchronously inside
`rpc_request()`'s existing branches, `snapshot_and_reset()` exposed for a future caller
to drain. It does NOT add a periodic self-publish loop, a new schema/bus channel, or
any wiring into `orion-substrate-runtime`'s tick loop or any live consumer -- that
cross-process "how does one service's in-memory counters reach anything else" question
is a separate, still-open architectural decision (this aggregator is per-process, same
as every other piece of `OrionBusAsync` instance state -- `_pending_rpc`,
`_rpc_subscribed`, etc. -- and is NOT shared across `OrionBusAsync.fork()` children,
consistent with that existing pattern). Building that consumption path before this
piece is proven in production would be designing around an assumed shape, exactly what
the "measure before minting" discipline this whole redesign has followed exists to
avoid.

**Bounded by design, not an afterthought.** This codebase has hit the same
"unbounded evidence-list" bug class multiple times independently (`evidence_event_ids`,
execution-merge evidence, others) -- capped collections here from the start rather than
retrofitted after a live incident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# Same cap precedent as evidence_event_ids/execution-merge-evidence elsewhere in this
# codebase (see feedback_execution_merge_cap.md-class fixes) -- if nothing ever calls
# snapshot_and_reset(), memory must stay flat regardless of call volume, not grow
# unboundedly between drains.
MAX_SAMPLES_PER_BUCKET = 500
MAX_DISTINCT_CHANNELS = 100


@dataclass
class RpcHealthSnapshot:
    """One drained window's worth of real RPC call outcomes. Success latency and
    timeout elapsed-time are kept as separate fields, never blended -- a timeout's
    elapsed_ms is the caller's own configured timeout_sec ceiling, not real
    round-trip latency (the exact conflation bug found and fixed in
    measure_rpc_health_baseline.py during step 1)."""

    window_start: datetime
    window_end: datetime
    success_count: int
    timeout_count: int
    success_latency_ms_p50: Optional[float]
    success_latency_ms_p95: Optional[float]
    success_latency_ms_max: Optional[float]
    timeout_elapsed_ms_max: Optional[float]
    channel_counts: dict[str, int]
    truncated: bool


def _percentile(sorted_values: list[float], pct: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


@dataclass
class RpcHealthAggregator:
    """Cheap, in-memory, per-`OrionBusAsync`-instance accumulator. `record_success()`/
    `record_timeout()` are called synchronously from `rpc_request()`'s existing
    success/timeout branches -- no I/O, no bus publish, bounded list appends only.
    asyncio is single-threaded cooperative concurrency, so no lock is needed for these
    plain dict/list mutations (same assumption already relied on elsewhere in
    `OrionBusAsync`, e.g. `_pending_rpc`).
    """

    _window_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _success_latencies_ms: list[float] = field(default_factory=list)
    _timeout_elapsed_ms: list[float] = field(default_factory=list)
    _success_count: int = 0
    _timeout_count: int = 0
    _channel_counts: dict[str, int] = field(default_factory=dict)
    _truncated: bool = False

    def record_success(self, *, request_channel: str, latency_ms: float) -> None:
        self._success_count += 1
        self._bump_channel(request_channel)
        if len(self._success_latencies_ms) < MAX_SAMPLES_PER_BUCKET:
            self._success_latencies_ms.append(latency_ms)
        else:
            self._truncated = True

    def record_timeout(self, *, request_channel: str, elapsed_ms: float) -> None:
        self._timeout_count += 1
        self._bump_channel(request_channel)
        if len(self._timeout_elapsed_ms) < MAX_SAMPLES_PER_BUCKET:
            self._timeout_elapsed_ms.append(elapsed_ms)
        else:
            self._truncated = True

    def _bump_channel(self, request_channel: str) -> None:
        if not request_channel:
            return
        if request_channel in self._channel_counts:
            self._channel_counts[request_channel] += 1
        elif len(self._channel_counts) < MAX_DISTINCT_CHANNELS:
            self._channel_counts[request_channel] = 1
        else:
            self._truncated = True

    def snapshot_and_reset(self) -> RpcHealthSnapshot:
        """Atomically read the accumulated window and reset for the next one."""
        now = datetime.now(timezone.utc)
        sorted_success = sorted(self._success_latencies_ms)
        snapshot = RpcHealthSnapshot(
            window_start=self._window_start,
            window_end=now,
            success_count=self._success_count,
            timeout_count=self._timeout_count,
            success_latency_ms_p50=_percentile(sorted_success, 0.5),
            success_latency_ms_p95=_percentile(sorted_success, 0.95),
            success_latency_ms_max=max(self._success_latencies_ms) if self._success_latencies_ms else None,
            timeout_elapsed_ms_max=max(self._timeout_elapsed_ms) if self._timeout_elapsed_ms else None,
            channel_counts=dict(self._channel_counts),
            truncated=self._truncated,
        )
        self._window_start = now
        self._success_latencies_ms = []
        self._timeout_elapsed_ms = []
        self._success_count = 0
        self._timeout_count = 0
        self._channel_counts = {}
        self._truncated = False
        return snapshot
