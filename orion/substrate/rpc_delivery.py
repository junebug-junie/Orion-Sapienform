"""RPC delivery pressure -- fold ``RpcHealthSnapshotV1`` windows into one honest
"are my requests getting answered in time" reading for the field.

Every service's shared bus client counts, per request channel, how many
``rpc_request()`` calls got a reply and how many hit their deadline
(``orion/core/bus/rpc_health.py``), and publishes those counts every 30 s on
``orion:rpc_health:snapshot``. That is a real rate with a denominator. Until this
module it reached orion-signal-gateway and orion-equilibrium-service, but not the
field, so ``capability:transport`` ``reliability_pressure`` had no input that
could ever move (its only source, the bus-observer's own failure count, has been
exactly 0 for weeks).

What this computes
------------------
Over a rolling window (default 10 min of snapshot ``window_end`` time), per hop
key, summed across every producer that called it::

    hop_pressure = timeouts / max(successes + timeouts, min_denominator)

and the reading is the WORST hop. ``max`` over hops, not a mesh-wide pooled
ratio: pooling lets one high-volume hop dilute a dead one (a hop answering 0 of
6 calls disappears inside 500 healthy calls elsewhere), and the field's own
diffusion already combines inputs with ``max``. The worst hop is named in the
reading, so the number always says which target it is about.

The denominator floor is what keeps a 1-of-1 timeout from reading 1.0: with the
default 10, one timeout reads at most 0.1, and a hop only reaches 1.0 after 10
timeouts in a window with no successes. It is a chosen constant, not a
calibrated one; the eval shows the readings it produces on live data.

Scope: which hops count
-----------------------
Only bus RPC made through ``rpc_request()`` (hop keys starting ``orion:``, with
or without a ``#label``). Excluded, on purpose:

- ``http:`` hops (durable-runs polling llama.cpp slots, the cabinet API, ...):
  not the bus. Several of them poll every few seconds, so they would also
  swamp any ratio they joined.
- ``gpu_pool:<class>#gpu_pool_wait``: queue wait for a GPU is capacity, not
  delivery (same reason orion-equilibrium-service excludes it).
- ``fcc:`` (Claude motor wall time), ``verb:`` / ``governor:`` (hand-rolled
  long-running work): these time out because the work ran long, not because a
  message was not delivered.
- Labels in ``exclude_labels`` (``transport_baseline.is_excluded`` semantics,
  reused rather than re-implemented):
  ``log_orion_metacognition`` breaks metacog's self-loop, exactly as in the
  equilibrium transport gate; ``current_turn_probe`` is cortex-exec's
  fail-open, 3-second current-turn signal probe, whose deadline is set below
  normal LLM latency by design (it produced 247 of the 311
  ``LLMGatewayService`` RPC timeouts on 2026-09-22..25). Counting it would pin
  this reading at ~0.1 on a healthy mesh.

What this deliberately does NOT do
----------------------------------
- No baseline. ``orion/metacog/transport_baseline.py`` baselines *latency*, which
  has no natural zero, so it needs an EWMA and anti-normalization guards. A
  timeout ratio does have one: a working request path does not time out. A
  learned baseline here would teach the field that a steady 10% failure rate is
  "calm", which is the normalization the transport baseline exists to refuse.
- No reading when nothing was called. With zero bus RPC calls in the window
  there is nothing to measure, and "not measured" is never reported as 0.0.
- No wall-clock reads: ``now_ts`` is passed in, and every stored time comes
  from the snapshot's own ``window_end``.

Pure and deterministic apart from a lock that makes ``fold`` (bus listener) and
``reading`` (tick thread) safe to call from different threads.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from orion.metacog.transport_baseline import _parse_ts, is_excluded

RPC_DELIVERY_NODE_ID = "node:substrate.rpc_delivery"
RPC_DELIVERY_TARGET_KIND = "rpc_delivery"
RPC_DELIVERY_CHANNEL = "rpc_timeout_pressure"
RPC_DELIVERY_REDUCER_KEY = "rpc_delivery"

# rpc_request() hop keys are the bus request channel, and every bus channel
# in this repo is named "orion:...". See orion/core/bus/rpc_health.py's hop key
# conventions for the other prefixes (http:, verb:, governor:, fcc:, gpu_pool:).
BUS_RPC_HOP_PREFIX = "orion:"

DEFAULT_EXCLUDE_LABELS: tuple[str, ...] = (
    "log_orion_metacognition",
    "gpu_pool_wait",
    "current_turn_probe",
)


@dataclass(frozen=True)
class RpcDeliveryConfig:
    window_s: float = 600.0
    min_denominator: int = 10
    exclude_labels: tuple[str, ...] = DEFAULT_EXCLUDE_LABELS
    # Bounds. 14 producer instances publish every 30 s today, so a 10 min
    # window holds ~280 snapshots; the cap only matters if something floods.
    max_snapshots: int = 5000
    max_producers: int = 512


@dataclass(frozen=True)
class RpcDeliveryReading:
    pressure: float
    worst_hop: str | None
    worst_timeouts: int
    worst_calls: int
    measured_hops: int
    total_calls: int
    total_timeouts: int
    producers: int
    window_s: float
    min_denominator: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "pressure": self.pressure,
            "worst_hop": self.worst_hop,
            "worst_timeouts": self.worst_timeouts,
            "worst_calls": self.worst_calls,
            "measured_hops": self.measured_hops,
            "total_calls": self.total_calls,
            "total_timeouts": self.total_timeouts,
            "producers": self.producers,
            "window_s": self.window_s,
            "min_denominator": self.min_denominator,
        }


@dataclass
class _Window:
    end_ts: float
    producer: str
    hops: dict[str, tuple[int, int]] = field(default_factory=dict)


def _as_count(v: Any) -> int:
    if isinstance(v, bool):
        return 0
    try:
        return max(0, int(v or 0))
    except (TypeError, ValueError):
        return 0


def counted_hop(hop: str, exclude_labels: Iterable[str]) -> bool:
    """True when ``hop`` is bus RPC delivery this reading should count."""
    if not isinstance(hop, str) or not hop.startswith(BUS_RPC_HOP_PREFIX):
        return False
    return not is_excluded(hop, exclude_labels)


def hop_pressure(timeouts: int, successes: int, min_denominator: int) -> float:
    calls = timeouts + successes
    if calls <= 0:
        return 0.0
    return timeouts / float(max(calls, int(min_denominator), 1))


class RpcDeliveryWindow:
    """Rolling per-hop success/timeout counts across every snapshot producer."""

    def __init__(self, config: RpcDeliveryConfig | None = None) -> None:
        self.config = config or RpcDeliveryConfig()
        self._windows: deque[_Window] = deque()
        self._last_end: dict[str, float] = {}
        self._lock = threading.Lock()

    def fold(self, payload: Mapping[str, Any]) -> bool:
        """Add one snapshot payload (``RpcHealthSnapshotV1`` as a dict).

        Returns False when it was ignored: no ``channel_latency`` (an old
        producer -- the pooled fields cannot be split by hop, so nothing is
        guessed from them), no usable ``window_end``, or a ``window_end`` not
        newer than the last one folded for the same (service, instance) -- a
        redelivery or replay.
        """
        if not isinstance(payload, Mapping):
            return False
        latency = payload.get("channel_latency")
        if not isinstance(latency, Mapping):
            return False
        end_ts = _parse_ts(payload.get("window_end"))
        if end_ts is None:
            return False
        producer = f"{payload.get('service') or ''}|{payload.get('instance') or ''}"
        hops: dict[str, tuple[int, int]] = {}
        for hop, stats in latency.items():
            if not isinstance(stats, Mapping) or not counted_hop(hop, self.config.exclude_labels):
                continue
            s = _as_count(stats.get("success_count"))
            t = _as_count(stats.get("timeout_count"))
            if s or t:
                hops[hop] = (s, t)
        with self._lock:
            last = self._last_end.get(producer)
            if last is not None and end_ts <= last:
                return False
            if last is None and len(self._last_end) >= self.config.max_producers:
                return False
            self._last_end[producer] = end_ts
            # Snapshots with no counted traffic are still recorded as "this
            # producer was heard" so `producers` reflects who reported.
            self._windows.append(_Window(end_ts=end_ts, producer=producer, hops=hops))
            while len(self._windows) > self.config.max_snapshots:
                self._windows.popleft()
        return True

    def _evict(self, now_ts: float) -> None:
        cutoff = now_ts - float(self.config.window_s)
        while self._windows and self._windows[0].end_ts < cutoff:
            self._windows.popleft()
        # Arrival order is not strictly window_end order across producers, so
        # also drop stale entries that are not at the head.
        if any(w.end_ts < cutoff for w in self._windows):
            self._windows = deque(w for w in self._windows if w.end_ts >= cutoff)
        stale = [p for p, ts in self._last_end.items() if ts < cutoff]
        for p in stale:
            # Forget the replay guard only once its window is gone; a
            # producer that restarts later starts fresh.
            del self._last_end[p]

    def reading(self, now_ts: float) -> RpcDeliveryReading | None:
        """The current reading, or None when no counted bus RPC call happened
        inside the window (not measured, never a fabricated 0.0)."""
        with self._lock:
            self._evict(float(now_ts))
            totals: dict[str, list[int]] = {}
            producers: set[str] = set()
            for w in self._windows:
                if w.end_ts > now_ts:
                    continue
                producers.add(w.producer)
                for hop, (s, t) in w.hops.items():
                    acc = totals.setdefault(hop, [0, 0])
                    acc[0] += s
                    acc[1] += t
        total_calls = sum(s + t for s, t in totals.values())
        if total_calls <= 0:
            return None
        n0 = int(self.config.min_denominator)
        worst_hop: str | None = None
        worst = (-1.0, -1, "")
        for hop, (s, t) in totals.items():
            p = hop_pressure(t, s, n0)
            # Highest pressure wins; ties go to more timeouts, then name, so the
            # named hop is deterministic.
            key = (p, t, hop)
            if key > worst:
                worst = key
                worst_hop = hop
        if worst[0] <= 0.0:
            # Nothing timed out: there is no "worst" hop to name, and naming one
            # by tie-break would point an inspector at a healthy channel.
            worst_hop = None
        ws, wt = totals[worst_hop] if worst_hop else (0, 0)
        return RpcDeliveryReading(
            pressure=round(max(0.0, min(1.0, worst[0])), 4),
            worst_hop=worst_hop,
            worst_timeouts=wt,
            worst_calls=ws + wt,
            measured_hops=len(totals),
            total_calls=total_calls,
            total_timeouts=sum(t for _, t in totals.values()),
            producers=len(producers),
            window_s=float(self.config.window_s),
            min_denominator=n0,
        )


def parse_exclude_labels(raw: str | Iterable[str] | None) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_EXCLUDE_LABELS
    items = raw.split(",") if isinstance(raw, str) else list(raw)
    return tuple(x.strip() for x in items if str(x).strip())


def rpc_delivery_receipt(reading: RpcDeliveryReading, *, now: Any) -> Any:
    """One ``ReductionReceiptV1`` carrying one ``rpc_delivery`` state delta.

    orion-field-digester turns ``pressure_hints.rpc_timeout_pressure`` into a
    ``mode="replace"`` write on ``node:substrate.rpc_delivery``; the topology
    edge carries it to ``capability:transport`` ``reliability_pressure``. The
    rest of ``after`` is evidence for whoever inspects the receipt: which hop,
    how many calls, how many producers reported.
    """
    import uuid

    from orion.schemas.reduction_receipt import ReductionReceiptV1
    from orion.schemas.state_delta import StateDeltaV1

    stamp = now.isoformat()
    return ReductionReceiptV1(
        receipt_id=f"receipt:{RPC_DELIVERY_REDUCER_KEY}:{uuid.uuid4().hex[:8]}",
        state_deltas=[
            StateDeltaV1(
                delta_id=f"{RPC_DELIVERY_REDUCER_KEY}:{stamp}",
                target_projection=f"substrate.{RPC_DELIVERY_REDUCER_KEY}.projection",
                target_kind=RPC_DELIVERY_TARGET_KIND,
                target_id=RPC_DELIVERY_NODE_ID,
                operation="update",
                after={
                    "node_id": RPC_DELIVERY_NODE_ID,
                    "pressure_hints": {RPC_DELIVERY_CHANNEL: reading.pressure},
                    "reading": reading.as_dict(),
                },
                # Snapshots are pub/sub telemetry, not grammar events: there is
                # no event id to cite. The reading itself is the evidence.
                caused_by_event_ids=[],
                reducer_id=f"substrate.{RPC_DELIVERY_REDUCER_KEY}",
                explanation=(
                    f"worst hop {reading.worst_hop}: {reading.worst_timeouts} timeouts "
                    f"in {reading.worst_calls} calls over {int(reading.window_s)}s"
                    if reading.worst_hop
                    else f"no timeouts in {reading.total_calls} bus RPC calls over "
                    f"{int(reading.window_s)}s"
                ),
            )
        ],
        created_at=now,
    )
