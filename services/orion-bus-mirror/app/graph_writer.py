"""Bus synaptic graph writer -- Phase 1 foundation + Phase 2 (in-flight
chains, per-verb latency).

Aggregates live bus traffic into bounded FalkorDB graph state instead of an
append-only raw log (the old MIRROR_SQLITE_PATH path, which grew to 98GB
unattended -- see services/orion-bus-mirror/README.md "Retention"). State size
here is bounded by mesh topology (organs x channels x verbs), not by message
count, so this can run against the full "orion:*" pattern indefinitely.

Phase 1 edge kinds, both derived from what a passive bus subscriber can
actually observe:

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

Phase 2 additions:

- **In-flight / long-lived chain tracking** (``ChainTracker.snapshot_open_chains``):
  a chain (correlation_id) is "in-flight" while it has had activity within the
  TTL window. There is no terminal marker in the envelope schema to know when
  a chain genuinely *completes* vs. goes quiet because it was abandoned --
  this is a real, named limitation (see design doc's "Real tensions" section),
  not solved here. What this *does* give: for every chain still being
  tracked, how long it has been running (first-seen to now) and how many hops
  it has had -- a live analog to thread-count/open-fd pressure, not a
  post-hoc latency stat.
- **Per-verb latency** (``EXECUTES_VERB`` edges, Organ -> Verb): mined from
  ``cognition.trace``-shaped payloads' ``steps[]`` array (real measured
  ``latency_ms`` per step, already present in the wire format -- see
  ``extract_verb_step_facts``). Only the "which organ, which verb, what's the
  latency baseline" slice is built here. The physical `node` a step ran on is
  captured as an edge property for manual inspection but does NOT partition
  the EWMA baseline (an organ running on multiple hosts would have its
  latencies blended) -- and preceding-verb / model_used / concurrent-load
  slicing (also named in the design doc) are not built at all. Named
  explicitly in README "Known gaps", not silently pretended solved.

Both edge kinds carry a rolling EWMA baseline plus a live z-score of the most
recent observation against that baseline -- the "connected components get
zscored on volume/latency" mechanism from the design brainstorm.

Deliberately out of scope for this module (Phase 3+, not built here):
anomaly propagation across edges, node centrality, chain-shape clustering,
per-node-partitioned verb latency, preceding-verb/model_used/concurrent-load
slicing.
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
    "OpenChainInfo",
    "OpenChainSummary",
    "summarize_open_chains",
    "VerbStepFact",
    "extract_verb_step_facts",
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


@dataclass
class _ChainEntry:
    started_at_epoch: float
    last_organ_id: str
    last_epoch: float
    hop_count: int


@dataclass(frozen=True)
class OpenChainInfo:
    correlation_id: str
    started_at_epoch: float
    last_seen_epoch: float
    duration_sec: float
    hop_count: int


@dataclass(frozen=True)
class OpenChainSummary:
    open_count: int
    long_running_count: int
    max_duration_sec: float


def summarize_open_chains(
    open_chains: list[OpenChainInfo], *, long_running_threshold_sec: float
) -> OpenChainSummary:
    """Pure aggregation over a ChainTracker snapshot -- separated from
    ChainTracker itself so the "what counts as long-running" policy is
    testable without touching TTL-eviction or dict internals.
    """
    if not open_chains:
        return OpenChainSummary(open_count=0, long_running_count=0, max_duration_sec=0.0)
    durations = [c.duration_sec for c in open_chains]
    long_running = sum(1 for d in durations if d >= long_running_threshold_sec)
    return OpenChainSummary(
        open_count=len(open_chains),
        long_running_count=long_running,
        max_duration_sec=max(durations),
    )


class ChainTracker:
    """Bounded correlation_id -> chain-state lookup for deriving
    CAUSALLY_FOLLOWED_BY edges and in-flight/long-running chain signals.
    TTL-evicted so full-tilt "orion:*" traffic can't grow this without bound
    -- checked on every write, not on a separate timer, so there is no
    unbounded-growth window between sweeps.

    "In-flight" here means "has had activity within the TTL window", not
    "known to still be running" -- there is no terminal marker in the
    envelope schema to distinguish a chain that finished from one that was
    abandoned. A chain simply stops being tracked once it goes quiet for
    longer than ttl_sec. See module docstring and README "Known gaps".
    """

    def __init__(self, *, ttl_sec: float) -> None:
        self._ttl_sec = ttl_sec
        self._entries: dict[str, _ChainEntry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def _evict_stale(self, now_epoch: float) -> None:
        cutoff = now_epoch - self._ttl_sec
        stale = [cid for cid, entry in self._entries.items() if entry.last_epoch < cutoff]
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

        existing = self._entries.get(fact.correlation_id)
        if existing is None:
            self._entries[fact.correlation_id] = _ChainEntry(
                started_at_epoch=fact.observed_at_epoch,
                last_organ_id=fact.organ_id,
                last_epoch=fact.observed_at_epoch,
                hop_count=1,
            )
            return None

        prior_organ_id, prior_epoch = existing.last_organ_id, existing.last_epoch
        existing.last_organ_id = fact.organ_id
        existing.last_epoch = fact.observed_at_epoch
        existing.hop_count += 1

        if prior_organ_id == fact.organ_id:
            return None
        return prior_organ_id, prior_epoch

    def snapshot_open_chains(self, now_epoch: float) -> list[OpenChainInfo]:
        """Read-only snapshot of every currently-tracked chain -- no bus/graph
        I/O, safe to call from a periodic tick independent of message
        processing. Does not evict (eviction only happens on ``observe()``,
        driven by real traffic) -- a chain quiet long enough to be stale but
        not yet observed-past is still reported here, which is correct: it
        really has been open at least that long as of ``now_epoch``.
        """
        return [
            OpenChainInfo(
                correlation_id=cid,
                started_at_epoch=entry.started_at_epoch,
                last_seen_epoch=entry.last_epoch,
                duration_sec=max(now_epoch - entry.started_at_epoch, 0.0),
                hop_count=entry.hop_count,
            )
            for cid, entry in self._entries.items()
        ]


@dataclass(frozen=True)
class VerbStepFact:
    """One executed step mined from a cognition-trace-shaped payload's
    ``steps[]`` array. ``node`` is captured for visibility but is NOT used to
    partition the EWMA baseline in this patch -- see module docstring.
    """

    organ_id: str
    verb_name: str
    latency_sec: float
    node: str | None
    observed_at_epoch: float


DEFAULT_MAX_VERB_STEPS_PER_ENVELOPE = 200


def extract_verb_step_facts(
    envelope: Any,
    *,
    now: float | None = None,
    max_steps: int = DEFAULT_MAX_VERB_STEPS_PER_ENVELOPE,
) -> list[VerbStepFact]:
    """Mines ``payload.steps[]`` for real measured per-step latency, if
    present. Returns an empty list (not an error) for any envelope whose
    payload isn't shaped this way -- most envelopes on the bus aren't
    cognition traces, and that's expected, not a malformed-data case.

    Two guards against untrusted payload content (found in review -- unlike
    Phase 1's facts, which only ever derive from trusted envelope metadata
    and wall-clock timestamps, this reads arbitrary nested payload fields):
    - ``steps`` is truncated to ``max_steps`` (real traces have a handful of
      steps; a buggy or adversarial producer sending thousands would
      otherwise stall the mirror's message loop for the full duration of
      writing that many edges sequentially -- the same O(N)-growth failure
      class this repo has hit before, just per-message instead of over time).
    - ``latency_ms`` must be finite (``math.isfinite``), not just numeric --
      a NaN/Infinity value would otherwise poison that edge's EWMA baseline
      permanently (verified live: one NaN observation makes every subsequent
      update on that edge NaN forever, with no self-healing).
    """
    source = getattr(envelope, "source", None)
    organ_id = getattr(source, "name", None) if source is not None else None
    if not organ_id:
        return []

    payload = getattr(envelope, "payload", None)
    if not isinstance(payload, dict):
        return []
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return []

    observed_at = now if now is not None else time.time()
    facts: list[VerbStepFact] = []
    for step in steps[:max_steps]:
        if not isinstance(step, dict):
            continue
        verb_name = step.get("verb_name")
        latency_ms = step.get("latency_ms")
        if not verb_name or not isinstance(latency_ms, (int, float)) or not math.isfinite(latency_ms):
            continue
        facts.append(
            VerbStepFact(
                organ_id=str(organ_id),
                verb_name=str(verb_name),
                latency_sec=max(float(latency_ms) / 1000.0, 0.0),
                node=str(step["node"]) if step.get("node") else None,
                observed_at_epoch=observed_at,
            )
        )
    return facts


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

    def record_verb_step(self, fact: VerbStepFact) -> None:
        prior_rows = self._client.graph_query(
            """
            MATCH (:Organ {organ_id: $organ_id})-[e:EXECUTES_VERB]->(:Verb {verb_name: $verb_name})
            RETURN e.latency_ewma_sec AS latency_ewma_sec, e.latency_var AS latency_var, e.count AS count
            """,
            {"organ_id": fact.organ_id, "verb_name": fact.verb_name},
        )
        row = prior_rows[0] if prior_rows else {}
        prior_ewma = float(row.get("latency_ewma_sec") or 0.0)
        prior_var = float(row.get("latency_var") or 0.0)
        prior_count = int(row.get("count") or 0)

        update = compute_ewma_update(
            prev_ewma=prior_ewma,
            prev_variance=prior_var,
            prev_count=prior_count,
            value=fact.latency_sec,
            alpha=self._alpha,
        )
        self._client.graph_query(
            """
            MERGE (o:Organ {organ_id: $organ_id})
            MERGE (v:Verb {verb_name: $verb_name})
            MERGE (o)-[e:EXECUTES_VERB]->(v)
            SET e.count = $count,
                e.last_seen_epoch = $now_epoch,
                e.last_node = $node,
                e.latency_ewma_sec = $latency_ewma_sec,
                e.latency_var = $latency_var,
                e.latency_zscore = $latency_zscore
            """,
            {
                "organ_id": fact.organ_id,
                "verb_name": fact.verb_name,
                "count": prior_count + 1,
                "now_epoch": fact.observed_at_epoch,
                "node": fact.node,
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
