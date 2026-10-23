"""Build the body atlas from what Orion's circulation actually is.

Two sources, kept apart by evidence label on every flow:

* ``declared``: ``orion/bus/channels.yaml`` producer -> channel -> consumer. Config truth. It
  also works on a past checkout (the catalog has producer lists from 2026-01).
* ``observed_*``: orion-bus-mirror's live synaptic graph in FalkorDB.
  ``observed_publish`` = a service seen publishing a channel, joined to that channel's declared
  consumers (the wiretap cannot see receivers, so the consumer side stays declared).
  ``observed_causal`` = ``CAUSALLY_FOLLOWED_BY``, one service's envelope followed by another's
  with the same correlation id. Runtime truth on both ends.

Neither source alone is the body: live 2026-10-23, 68 bus source ids match no catalog service.
33 are smokes/probes (transient); the other 35 are real body parts (security-watcher,
vision-retina-walkway, power-guard, orion-chat-memory, state-journaler, ...) that no catalog
producer/consumer list names.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from orion.body_atlas.model import (
    AtlasSourcesV1,
    AtlasStatsV1,
    BodyAtlasV1,
    FlowV1,
    OrganFlowV1,
    OrganV1,
    PartV1,
)
from orion.body_atlas.organs import louvain, modularity

CATALOG_RELPATH = Path("orion/bus/channels.yaml")
# A publisher not in the catalog is transient (smoke, probe, one-off script) when it is below
# this message count AND was not seen within the live window. Neither test alone separates them
# (live 2026-10-23): orion-hub-retro (461) and orion-situational (29) are real, low-traffic and
# live; boundary-prompt-eval (3,432, last seen 17 d ago) is an eval harness that clears the floor
# and stays a dormant part. Recent probes (2-18 messages) pass as parts and end up as weakly
# connected singletons, visible in the stats. Declared services are always parts.
DEFAULT_MIN_OBSERVED_COUNT = 1000
DEFAULT_LIVE_WINDOW_SEC = 24 * 3600
NAMESPACE_HINTS = 3


@dataclass
class ObservedBus:
    # (source_id, channel, count, last_seen_epoch)
    publishes: list[tuple[str, str, int, float]] = field(default_factory=list)
    # (source_id, target_id, count, last_seen_epoch)
    causal: list[tuple[str, str, int, float]] = field(default_factory=list)


_PUBLISH_CYPHER = (
    "MATCH (o:Organ)-[e:PUBLISHES]->(c:Channel) "
    "RETURN o.organ_id AS source, c.channel AS channel, e.count AS count, e.last_seen_epoch AS last_seen"
)
_CAUSAL_CYPHER = (
    "MATCH (a:Organ)-[e:CAUSALLY_FOLLOWED_BY]->(b:Organ) "
    "RETURN a.organ_id AS source, b.organ_id AS target, e.count AS count, e.last_seen_epoch AS last_seen"
)


def fetch_observed(client: Any) -> ObservedBus:
    """Read the mirror's graph with any ``graph_query`` client (use ``read_only=True``)."""

    def rows(cypher: str) -> list[dict]:
        return [r for r in (client.graph_query(cypher) or []) if isinstance(r, dict)]

    return ObservedBus(
        # values may arrive as strings (non-compact replies): int(float()) takes both
        publishes=[(str(r["source"]), str(r["channel"]), int(float(r["count"] or 0)), float(r["last_seen"] or 0))
                   for r in rows(_PUBLISH_CYPHER)],
        causal=[(str(r["source"]), str(r["target"]), int(float(r["count"] or 0)), float(r["last_seen"] or 0))
                for r in rows(_CAUSAL_CYPHER)],
    )


def load_catalog(root: Path) -> list[dict]:
    path = root / CATALOG_RELPATH
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [c for c in (data.get("channels") or []) if isinstance(c, dict) and c.get("name")]


def _services(entry: dict, key: str) -> list[str]:
    return [s for s in (entry.get(key) or []) if isinstance(s, str) and s and s != "*"]


def _namespace(channel: str) -> str | None:
    parts = channel.split(":")
    return parts[1] if len(parts) > 2 and parts[1] not in ("", "*") else None


def _iso(epoch: float | None) -> str | None:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat() if epoch else None


class _FlowAcc:
    def __init__(self) -> None:
        self.weight = 0.0
        self.evidence: set[str] = set()
        self.channels: set[str] = set()
        self.causal_count = 0


def build_atlas(
    catalog: Iterable[dict],
    observed: ObservedBus | None,
    *,
    now: datetime,
    catalog_path: str = str(CATALOG_RELPATH),
    live_graph: str | None = None,
    min_observed_count: int = DEFAULT_MIN_OBSERVED_COUNT,
    live_window_sec: int = DEFAULT_LIVE_WINDOW_SEC,
) -> BodyAtlasV1:
    catalog = list(catalog)
    by_channel = {c["name"]: c for c in catalog}
    declared: set[str] = set()
    for entry in catalog:
        declared.update(_services(entry, "producer_services"), _services(entry, "consumer_services"))

    def resolve(raw: str) -> str:
        # The bus stamps short names ("cortex-exec"); the catalog uses "orion-cortex-exec".
        if raw in declared:
            return raw
        if f"orion-{raw}" in declared:
            return f"orion-{raw}"
        return raw

    seen: dict[str, dict] = defaultdict(lambda: {"count": 0, "last": 0.0, "ids": set()})
    flows: dict[tuple[str, str], _FlowAcc] = defaultdict(_FlowAcc)

    def link(src: str, dst: str, weight: float, evidence: str, channel: str | None = None) -> None:
        if src == dst:
            return
        acc = flows[(src, dst)]
        acc.weight += weight
        acc.evidence.add(evidence)
        if channel:
            acc.channels.add(channel)

    # A channel is one hyperedge: it contributes one unit (declared) or log10(1 + messages)
    # (observed), split evenly over the producer -> consumer pairs it joins (clique expansion
    # normalized by edge size). Without this, broadcast channels glue the body into one blob:
    # live 2026-10-23, orion:system:health (every service's heartbeat) carried 29.5% of all flow
    # weight with pairwise weights and Q fell to 0.22.
    for entry in catalog:
        producers = _services(entry, "producer_services")
        consumers = _services(entry, "consumer_services")
        pairs = [(p, q) for p in producers for q in consumers if p != q]
        for p, q in pairs:
            link(p, q, 1.0 / len(pairs), "declared", entry["name"])

    if observed is not None:
        on_channel: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for raw, channel, count, last in observed.publishes:
            name = resolve(raw)
            rec = seen[name]
            rec["count"] += count
            rec["last"] = max(rec["last"], last)
            rec["ids"].add(raw)
            if channel in by_channel:
                on_channel[channel].append((name, count))
        for channel, pubs in on_channel.items():
            consumers = _services(by_channel[channel], "consumer_services")
            pairs = [(p, q, n) for p, n in pubs for q in consumers if p != q]
            if not pairs:
                continue
            total = sum(n for _, _, n in pairs)
            unit = math.log10(1 + sum(n for _, n in pubs))
            for p, q, n in pairs:
                # each pair's share of the channel's unit follows its publisher's traffic
                link(p, q, unit * n / total if total else unit / len(pairs), "observed_publish", channel)
        for raw_a, raw_b, count, last in observed.causal:
            a, b = resolve(raw_a), resolve(raw_b)
            for name, raw in ((a, raw_a), (b, raw_b)):
                seen[name]["ids"].add(raw)
                seen[name]["last"] = max(seen[name]["last"], last)
            if a != b:
                link(a, b, math.log10(1 + count), "observed_causal")
                flows[(a, b)].causal_count += count

    now_epoch = now.timestamp()
    parts: dict[str, PartV1] = {}
    for name in sorted(declared | set(seen) | {n for pair in flows for n in pair}):
        rec = seen.get(name)
        is_declared = name in declared
        recent = rec is not None and bool(rec["last"]) and now_epoch - rec["last"] <= live_window_sec
        if rec is not None and not is_declared and rec["count"] < min_observed_count and not recent:
            status = "transient"
        elif recent:
            status = "live"
        elif rec is not None:
            status = "dormant"
        else:
            status = "declared_only"
        parts[name] = PartV1(
            part_id=f"part:{name}",
            name=name,
            kind="endpoint" if ":" in name else "service",
            status=status,
            declared=is_declared,
            observed_ids=sorted(rec["ids"]) if rec else [],
            publish_count=rec["count"] if rec else 0,
            last_seen_at=_iso(rec["last"]) if rec else None,
        )

    flow_models = [
        FlowV1(
            source=f"part:{s}",
            target=f"part:{t}",
            weight=round(acc.weight, 4),
            evidence=sorted(acc.evidence),
            channels=sorted(acc.channels),
            causal_count=acc.causal_count,
        )
        for (s, t), acc in sorted(flows.items())
    ]

    # Undirected organ graph over every non-transient part (isolated parts become singletons).
    body = sorted(n for n, p in parts.items() if p.status != "transient")
    adj: dict[str, dict[str, float]] = {n: {} for n in body}
    for (s, t), acc in flows.items():
        if s in adj and t in adj:
            adj[s][t] = adj[s].get(t, 0.0) + acc.weight
            adj[t][s] = adj[t].get(s, 0.0) + acc.weight
    com = louvain(adj)
    q = modularity(adj, com) if adj else 0.0
    strength = {n: sum(adj[n].values()) for n in body}

    groups: dict[int, list[str]] = defaultdict(list)
    for n in body:
        groups[com[n]].append(n)

    organ_of: dict[str, str] = {}
    organs: list[OrganV1] = []
    for members in groups.values():
        members.sort(key=lambda n: (-strength[n], n))
        member_set = set(members)
        internal = external = 0.0
        ns: Counter[str] = Counter()
        for (s, t), acc in flows.items():
            if s in member_set and t in member_set:
                internal += acc.weight
                ns.update(x for x in (_namespace(c) for c in acc.channels) if x)
            elif (s in member_set) != (t in member_set) and s in adj and t in adj:
                external += acc.weight
        organ_id = "organ:" + hashlib.sha1("|".join(sorted(members)).encode()).hexdigest()[:10]
        for n in members:
            organ_of[n] = organ_id
        organs.append(OrganV1(
            organ_id=organ_id,
            members=[f"part:{n}" for n in members],
            hub=f"part:{members[0]}",
            namespaces=[k for k, _ in sorted(ns.items(), key=lambda kv: (-kv[1], kv[0]))[:NAMESPACE_HINTS]],
            internal_weight=round(internal, 4),
            external_weight=round(external, 4),
        ))
    organs.sort(key=lambda o: (-len(o.members), -o.internal_weight, o.organ_id))

    between: dict[tuple[str, str], float] = defaultdict(float)
    for (s, t), acc in flows.items():
        a, b = organ_of.get(s), organ_of.get(t)
        if a and b and a != b:
            between[(a, b)] += acc.weight
    organ_flows = [OrganFlowV1(source=a, target=b, weight=round(w, 4)) for (a, b), w in sorted(between.items())]

    in_organs = len(body)
    return BodyAtlasV1(
        as_of=now.astimezone(timezone.utc).isoformat(),
        sources=AtlasSourcesV1(
            catalog_path=catalog_path,
            catalog_channels=len(catalog),
            live=observed is not None,
            live_graph=live_graph if observed is not None else None,
            publish_edges=len(observed.publishes) if observed else 0,
            causal_edges=len(observed.causal) if observed else 0,
        ),
        parts=list(parts.values()),
        flows=flow_models,
        organs=organs,
        organ_flows=organ_flows,
        stats=AtlasStatsV1(
            parts_total=len(parts),
            parts_in_organs=in_organs,
            transient_parts=len(parts) - in_organs,
            flows=len(flow_models),
            organs=len(organs),
            singleton_organs=sum(1 for o in organs if len(o.members) == 1),
            modularity=round(q, 4),
            largest_organ_share=round(max((len(o.members) for o in organs), default=0) / in_organs, 4)
            if in_organs else 0.0,
        ),
    )
