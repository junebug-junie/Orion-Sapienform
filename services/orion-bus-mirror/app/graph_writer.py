"""Bus synaptic graph writer -- Phase 1 foundation.

Aggregates live bus traffic into bounded FalkorDB graph state instead of an
append-only raw log (the old MIRROR_SQLITE_PATH path, which grew to 98GB
unattended -- see services/orion-bus-mirror/README.md "Retention"). State size
here is bounded by mesh topology (organs x channels), not by message count, so
this can run against the full "orion:*" pattern indefinitely.

Two edge kinds, both derived from what a passive bus subscriber can actually
observe:

- ``PUBLISHES`` (Organ -> Channel): every message an organ publishes. This is
  the only thing a wiretap directly observes -- a CONSUMES edge would require
  each consumer to self-report (the RPC-health arc's pattern, PR #1313), not
  something inferable from eavesdropping alone. Deliberately not built here;
  see README "Known gaps" instead of faking it from static channels.yaml
  metadata (that would be config truth, not runtime truth).
- ``CAUSALLY_FOLLOWED_BY`` (Organ -> Organ): when a second envelope carrying a
  correlation_id already seen from a *different* organ arrives, that hop is
  recorded with its latency. This is a live version of verifying
  ``orion/signals/registry.py``'s ORGAN_REGISTRY causal_parent_organs edges,
  which its own comment admits are "first-pass structural approximations."

Both edge kinds carry a rolling EWMA baseline (of inter-arrival gap for
PUBLISHES, of hop latency for CAUSALLY_FOLLOWED_BY) plus a live z-score of the
most recent observation against that baseline -- the "connected components
get zscored on volume/latency" mechanism from the design brainstorm.

Deliberately out of scope for this module (Phase 2, not built here):
in-flight/still-open chain tracking, per-verb latency slicing from payload
content, anomaly propagation across edges, node centrality, chain-shape
clustering.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from orion.graph.falkor_client import FalkorGraphClient

__all__ = [
    "BusEventFact",
    "EwmaUpdate",
    "compute_ewma_update",
    "ChainTracker",
    "BusSynapticGraphWriter",
    "extract_bus_event_fact",
]

_MIN_VARIANCE = 1e-6


@dataclass(frozen=True)
class BusEventFact:
    """Extracted from one mirrored envelope -- everything the graph writer needs."""

    organ_id: str
    channel: str
    correlation_id: str | None
    observed_at_epoch: float


@dataclass(frozen=True)
class EwmaUpdate:
    ewma: float
    variance: float
    zscore: float | None  # None until a baseline exists (first observation on an edge)


def compute_ewma_update(
    *,
    prev_ewma: float,
    prev_variance: float,
    prev_count: int,
    value: float,
    alpha: float,
) -> EwmaUpdate:
    """Pure incremental EWMA mean/variance update, plus a z-score of ``value``
    against the *prior* baseline (computed before the baseline absorbs it).

    No z-score is returned on an edge's first observation (``prev_count == 0``)
    -- there is no baseline yet to be anomalous against. Returning 0.0 there
    would misrepresent "no data yet" as "measured, not anomalous" (this
    repo's own "no empty-shell cognition" rule).
    """
    if prev_count == 0:
        return EwmaUpdate(ewma=value, variance=0.0, zscore=None)

    variance_floor = max(prev_variance, _MIN_VARIANCE)
    zscore = (value - prev_ewma) / math.sqrt(variance_floor)
    new_ewma = alpha * value + (1 - alpha) * prev_ewma
    deviation = value - prev_ewma
    new_variance = alpha * (deviation * deviation) + (1 - alpha) * prev_variance
    return EwmaUpdate(ewma=new_ewma, variance=new_variance, zscore=zscore)


class ChainTracker:
    """Bounded correlation_id -> (organ_id, epoch) lookup for deriving
    CAUSALLY_FOLLOWED_BY edges. TTL-evicted so full-tilt "orion:*" traffic
    can't grow this without bound -- checked on every write, not on a
    separate timer, so there is no unbounded-growth window between sweeps.
    """

    def __init__(self, *, ttl_sec: float) -> None:
        self._ttl_sec = ttl_sec
        self._entries: dict[str, tuple[str, float]] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def _evict_stale(self, now_epoch: float) -> None:
        cutoff = now_epoch - self._ttl_sec
        stale = [cid for cid, (_, ts) in self._entries.items() if ts < cutoff]
        for cid in stale:
            del self._entries[cid]

    def observe(self, fact: BusEventFact) -> tuple[str, float] | None:
        """Records this fact's (organ_id, observed_at_epoch) under its
        correlation_id, evicting stale entries first. Returns the prior
        (organ_id, epoch) entry for this correlation_id if one existed and
        came from a *different* organ (a real cross-organ hop) -- None
        otherwise (no correlation_id, first sighting, or same-organ repeat).
        """
        self._evict_stale(fact.observed_at_epoch)
        if not fact.correlation_id:
            return None

        prior = self._entries.get(fact.correlation_id)
        self._entries[fact.correlation_id] = (fact.organ_id, fact.observed_at_epoch)

        if prior is None:
            return None
        prior_organ, _prior_epoch = prior
        if prior_organ == fact.organ_id:
            return None
        return prior


class BusSynapticGraphWriter:
    """Two-step read-then-write per edge: read prior EWMA state, compute the
    update in Python (testable, no Cypher math), write the result. Costs one
    extra round trip per edge versus doing the arithmetic inline in Cypher,
    traded deliberately for correctness that doesn't depend on a specific
    graph engine's WITH-clause evaluation-order semantics.

    Single-instance-only by design: the read-then-write pair has no CAS/version
    check, so concurrent writers racing on the same edge would corrupt EWMA
    state (lost updates). Safe as-is because ``mirror_bus()`` calls this
    sequentially within one process's event loop with no concurrent dispatch
    -- scaling to multiple instances would need a real fix first, not just a
    docker-compose replica bump.
    """

    def __init__(self, client: FalkorGraphClient, *, alpha: float) -> None:
        self._client = client
        self._alpha = alpha

    def record_publish(self, fact: BusEventFact) -> None:
        prior_rows = self._client.graph_query(
            """
            MATCH (:Organ {organ_id: $organ_id})-[e:PUBLISHES]->(:Channel {channel: $channel})
            RETURN e.gap_ewma_sec AS gap_ewma_sec, e.gap_var AS gap_var, e.count AS count,
                   e.last_seen_epoch AS last_seen_epoch
            """,
            {"organ_id": fact.organ_id, "channel": fact.channel},
        )
        row = prior_rows[0] if prior_rows else {}
        prior_ewma = float(row.get("gap_ewma_sec") or 0.0)
        prior_var = float(row.get("gap_var") or 0.0)
        prior_count = int(row.get("count") or 0)
        last_seen_epoch = (
            float(row["last_seen_epoch"])
            if row.get("last_seen_epoch") is not None
            else fact.observed_at_epoch
        )
        gap = max(fact.observed_at_epoch - last_seen_epoch, 0.0)
        update = compute_ewma_update(
            prev_ewma=prior_ewma,
            prev_variance=prior_var,
            prev_count=prior_count,
            value=gap,
            alpha=self._alpha,
        )
        self._client.graph_query(
            """
            MERGE (o:Organ {organ_id: $organ_id})
            MERGE (c:Channel {channel: $channel})
            MERGE (o)-[e:PUBLISHES]->(c)
            SET e.count = $count,
                e.last_seen_epoch = $now_epoch,
                e.gap_ewma_sec = $gap_ewma_sec,
                e.gap_var = $gap_var,
                e.gap_zscore = $gap_zscore
            """,
            {
                "organ_id": fact.organ_id,
                "channel": fact.channel,
                "count": prior_count + 1,
                "now_epoch": fact.observed_at_epoch,
                "gap_ewma_sec": update.ewma,
                "gap_var": update.variance,
                "gap_zscore": update.zscore,
            },
        )

    def record_causal_hop(
        self,
        *,
        prior_organ_id: str,
        prior_epoch: float,
        fact: BusEventFact,
    ) -> None:
        latency_sec = max(fact.observed_at_epoch - prior_epoch, 0.0)
        prior_rows = self._client.graph_query(
            """
            MATCH (:Organ {organ_id: $prior_organ_id})-[e:CAUSALLY_FOLLOWED_BY]->(:Organ {organ_id: $organ_id})
            RETURN e.latency_ewma_sec AS latency_ewma_sec, e.latency_var AS latency_var, e.count AS count
            """,
            {"prior_organ_id": prior_organ_id, "organ_id": fact.organ_id},
        )
        row = prior_rows[0] if prior_rows else {}
        prior_ewma = float(row.get("latency_ewma_sec") or 0.0)
        prior_var = float(row.get("latency_var") or 0.0)
        prior_count = int(row.get("count") or 0)

        update = compute_ewma_update(
            prev_ewma=prior_ewma,
            prev_variance=prior_var,
            prev_count=prior_count,
            value=latency_sec,
            alpha=self._alpha,
        )
        self._client.graph_query(
            """
            MERGE (a:Organ {organ_id: $prior_organ_id})
            MERGE (b:Organ {organ_id: $organ_id})
            MERGE (a)-[e:CAUSALLY_FOLLOWED_BY]->(b)
            SET e.count = $count,
                e.last_seen_epoch = $now_epoch,
                e.latency_ewma_sec = $latency_ewma_sec,
                e.latency_var = $latency_var,
                e.latency_zscore = $latency_zscore
            """,
            {
                "prior_organ_id": prior_organ_id,
                "organ_id": fact.organ_id,
                "count": prior_count + 1,
                "now_epoch": fact.observed_at_epoch,
                "latency_ewma_sec": update.ewma,
                "latency_var": update.variance,
                "latency_zscore": update.zscore,
            },
        )


def extract_bus_event_fact(envelope: Any, *, channel: str, now: float | None = None) -> BusEventFact | None:
    """``envelope`` is a decoded BaseEnvelope (or envelope-shaped object with
    ``source.name`` and ``correlation_id``). Returns None if it doesn't carry
    enough identity to place on the graph (no source name) -- fails open,
    consistent with the rest of this arc's fail-open-on-malformed-data rule.
    """
    source = getattr(envelope, "source", None)
    organ_id = getattr(source, "name", None) if source is not None else None
    if not organ_id:
        return None
    correlation_id = getattr(envelope, "correlation_id", None)
    return BusEventFact(
        organ_id=str(organ_id),
        channel=channel,
        correlation_id=str(correlation_id) if correlation_id else None,
        observed_at_epoch=now if now is not None else time.time(),
    )
