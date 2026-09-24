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

**Per-hop breakdown (2026-09-24, A0 of
docs/superpowers/specs/2026-09-24-metacog-capture-and-transport-ewma-baseline-design.md).**
Next to the pooled fields, every window now also carries per-HOP sufficient statistics
(``channel_latency``: success/timeout counts, sum and sum-of-squares of ``ln(ms)``, max).
Hop key conventions -- producers MUST use these so one EWMA baseline per key means the
same thing everywhere:

- ``rpc_request()``: the bus request channel, e.g. ``orion:cortex:exec:request:chat``
- ``rpc_request(..., health_label="x")``: ``"<channel>#x"`` (``hop_key()``), used to split
  one channel's traffic by purpose -- e.g. metacog's own dispatch is labelled
  ``log_orion_metacognition`` so a transport gate can exclude it
- hand-rolled bus RPC: ``verb:<verb_name>`` (cortex-orch chat lane),
  ``governor:<mode>`` (hub -> harness-governor)
- HTTP: ``http:<host><path>`` (``orion.core.bus.http_health``)
- FCC motor subprocess wall time: ``fcc:<served_model>``

Hand-rolled paths call ``record_hop_success()``/``record_hop_timeout()``. Those record
into ``channel_latency`` ONLY -- the pooled fields keep their original, documented
meaning (``rpc_request()`` outcomes on this bus), so no existing consumer of the pooled
fields sees its input change under it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("orion.bus.rpc_health")

# Same cap precedent as evidence_event_ids/execution-merge-evidence elsewhere in this
# codebase (see feedback_execution_merge_cap.md-class fixes) -- if nothing ever calls
# snapshot_and_reset(), memory must stay flat regardless of call volume, not grow
# unboundedly between drains.
MAX_SAMPLES_PER_BUCKET = 500
MAX_DISTINCT_CHANNELS = 100
# Per-hop stats are O(1) per key (no sample list), so the bound is on key cardinality
# only. A key past the cap folds into OVERFLOW_HOP_KEY instead of being dropped, so
# window-level counts stay conserved and the overflow is itself visible.
MAX_DISTINCT_HOPS = 200
OVERFLOW_HOP_KEY = "_overflow"
# ln(0) is undefined; a sub-microsecond reading is clock noise, not a latency.
_MIN_LOG_MS = 1e-3


def hop_key(channel: str, health_label: Optional[str] = None) -> str:
    """Canonical hop key for an rpc_request() call. ``health_label`` splits one
    channel's traffic by purpose: ``hop_key("orion:x", "verb")`` -> ``"orion:x#verb"``."""
    label = (health_label or "").strip()
    return f"{channel}#{label}" if label else channel


@dataclass
class HopLatency:
    """Per-hop sufficient statistics for one window. Mirrors RpcChannelLatencyV1."""

    success_count: int = 0
    timeout_count: int = 0
    log_ms_sum: float = 0.0
    log_ms_sumsq: float = 0.0
    max_ms: Optional[float] = None

    def add_success(self, elapsed_ms: float) -> None:
        self.success_count += 1
        ms = float(elapsed_ms)
        if not math.isfinite(ms):
            return
        lm = math.log(max(ms, _MIN_LOG_MS))
        self.log_ms_sum += lm
        self.log_ms_sumsq += lm * lm
        if self.max_ms is None or ms > self.max_ms:
            self.max_ms = ms

    def add_timeout(self) -> None:
        self.timeout_count += 1

    def as_dict(self) -> dict:
        return {
            "success_count": self.success_count,
            "timeout_count": self.timeout_count,
            "log_ms_sum": self.log_ms_sum,
            "log_ms_sumsq": self.log_ms_sumsq,
            "max_ms": self.max_ms,
        }


@dataclass
class RpcHealthSnapshot:
    """One drained window's worth of real RPC call outcomes. Success latency and
    timeout elapsed-time are kept as separate fields, never blended -- a timeout's
    elapsed_ms is the caller's own configured timeout_sec ceiling, not real
    round-trip latency (the exact conflation bug found and fixed in
    measure_rpc_health_baseline.py during step 1).

    `truncated=True` means the window's real call *counts* (`success_count`/
    `timeout_count`) are still accurate, but the sample lists behind
    `success_latency_ms_*`/`timeout_elapsed_ms_max` are first-N-wins, not a rolling
    window: once `MAX_SAMPLES_PER_BUCKET` is hit mid-window, every later sample is
    dropped from the percentile/max computation, not the earliest ones. A future
    consumer draining a long-lived, high-volume window should not assume
    `success_latency_ms_max` is the true max of the whole window when `truncated` is
    set -- a later latency spike could be invisible. Not a concern yet (no periodic
    drain is wired), but a real caveat for whoever wires the first one."""

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
    channel_latency: dict[str, HopLatency] = field(default_factory=dict)


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
    _hops: dict[str, HopLatency] = field(default_factory=dict)

    def record_success(
        self, *, request_channel: str, latency_ms: float, health_label: Optional[str] = None
    ) -> None:
        """Never raises past this boundary: this call sits directly in
        `rpc_request()`'s success path, after the real result is already in hand --
        a bug in here must never mask or replace that real outcome."""
        try:
            self._success_count += 1
            self._bump_channel(request_channel)
            if len(self._success_latencies_ms) < MAX_SAMPLES_PER_BUCKET:
                self._success_latencies_ms.append(latency_ms)
            else:
                self._truncated = True
            if request_channel:
                self._hop(hop_key(request_channel, health_label)).add_success(latency_ms)
        except Exception:
            logger.warning("failed to record RPC success in health aggregator", exc_info=True)

    def record_timeout(
        self, *, request_channel: str, elapsed_ms: float, health_label: Optional[str] = None
    ) -> None:
        """Never raises past this boundary: this call sits directly in
        `rpc_request()`'s `except asyncio.TimeoutError` branch, immediately before
        `raise TimeoutError(...)` -- a bug in here must never substitute a different
        exception type for the real timeout the caller is about to see."""
        try:
            self._timeout_count += 1
            self._bump_channel(request_channel)
            if len(self._timeout_elapsed_ms) < MAX_SAMPLES_PER_BUCKET:
                self._timeout_elapsed_ms.append(elapsed_ms)
            else:
                self._truncated = True
            if request_channel:
                self._hop(hop_key(request_channel, health_label)).add_timeout()
        except Exception:
            logger.warning("failed to record RPC timeout in health aggregator", exc_info=True)

    def record_hop_success(self, hop: str, elapsed_ms: float) -> None:
        """Record a successful round trip on a hand-rolled hop (not rpc_request()).
        Lands in ``channel_latency`` only; pooled fields are untouched. Never raises."""
        try:
            if hop:
                self._hop(str(hop)).add_success(elapsed_ms)
        except Exception:
            logger.warning("failed to record hop success hop=%s", hop, exc_info=True)

    def record_hop_timeout(self, hop: str, elapsed_ms: Optional[float] = None) -> None:
        """Record a timeout on a hand-rolled hop. ``elapsed_ms`` is accepted for API
        symmetry and logging, but never enters the latency statistics (a timeout's
        elapsed time is the caller's ceiling, not a round trip). Never raises."""
        try:
            if hop:
                self._hop(str(hop)).add_timeout()
        except Exception:
            logger.warning("failed to record hop timeout hop=%s", hop, exc_info=True)

    def _hop(self, key: str) -> HopLatency:
        stats = self._hops.get(key)
        if stats is not None:
            return stats
        if len(self._hops) >= MAX_DISTINCT_HOPS:
            self._truncated = True
            key = OVERFLOW_HOP_KEY
            stats = self._hops.get(key)
            if stats is not None:
                return stats
        stats = HopLatency()
        self._hops[key] = stats
        return stats

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
            channel_latency=self._hops,
        )
        self._window_start = now
        self._success_latencies_ms = []
        self._timeout_elapsed_ms = []
        self._success_count = 0
        self._timeout_count = 0
        self._channel_counts = {}
        self._truncated = False
        self._hops = {}
        return snapshot
