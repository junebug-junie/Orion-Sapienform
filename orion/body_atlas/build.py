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
import re
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
from orion.body_atlas.sources import BodyInputs

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
# Flow weight one camera, store or process part adds, split over its edges (see _add_body).
BODY_PART_UNIT = 1.0


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
        self.relation = "signals"


_STORE_PREFIX = {"postgres": "pg", "chroma": "chroma", "falkordb": "falkor", "redis": "redis"}
_PROCESS_PREFIX = {"durable_workflow": "run", "substrate_reducer": "reducer"}
# part ids of the non-bus kinds (hosts, cameras, stores, processes) all start with one of these
BODY_PART_PREFIXES = tuple(f"part:{x}:" for x in ("host", "camera", *_STORE_PREFIX.values(), *_PROCESS_PREFIX.values()))


def match_part(raw: str, known: set[str], host_names: Iterable[str]) -> str | None:
    """A container / compose service / env-file directory name -> the atlas part it is.

    Rules, not a table: ``orion-athena-cortex-exec`` and ``orion-orion-athena-bus-core`` drop
    the ``orion-`` and ``<host>-`` prefixes; ``vision-retina-walkway`` gains ``orion-``; a
    compose replica suffix (``-1``) drops. None when nothing matches (counted as unmatched)."""
    hosts = "|".join(re.escape(h) for h in sorted(host_names, key=len, reverse=True))
    prefix = rf"^(?:orion-)*(?:(?:{hosts})-)?" if hosts else r"^(?:orion-)*"
    for base in (raw, re.sub(r"-\d+$", "", raw)):
        rest = re.sub(prefix, "", base)
        for cand in (base, rest, f"orion-{rest}"):
            if cand in known:
                return cand
    return None


def _status(last: float | None, now_epoch: float, window: int, *, seen: bool) -> str:
    if last and now_epoch - last <= window:
        return "live"
    return "dormant" if seen or last else "declared_only"


def build_atlas(
    catalog: Iterable[dict],
    observed: ObservedBus | None,
    *,
    now: datetime,
    catalog_path: str = str(CATALOG_RELPATH),
    live_graph: str | None = None,
    min_observed_count: int = DEFAULT_MIN_OBSERVED_COUNT,
    live_window_sec: int = DEFAULT_LIVE_WINDOW_SEC,
    body: BodyInputs | None = None,
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

    def link(src: str, dst: str, weight: float, evidence: str, channel: str | None = None,
             relation: str = "signals") -> None:
        if src == dst:
            return
        acc = flows[(src, dst)]
        acc.weight += weight
        acc.evidence.add(evidence)
        acc.relation = relation
        if channel:
            acc.channels.add(channel)

    # A channel is one hyperedge: it contributes one unit (declared) or log10(1 + messages)
    # (observed), split evenly over the producer -> consumer pairs it joins (clique expansion
    # normalized by edge size). Without this, broadcast channels glue the body into one blob:
    # live 2026-10-23, orion:system:health (every service's heartbeat) carried 29.5% of all flow
    # weight with pairwise weights and Q fell to 0.22.
    persisted: dict[str, str] = {}
    if body is not None:
        # move a persisted channel only when its table is a part with a writes link for it; the
        # producers are catalog producers, so always parts (never left with no flow at all)
        linked = {via for st in body.stores for _, rel, ev, via in st.links
                  if rel == "writes" and ev == "declared_route" and via}
        persisted = {ch: w for ch, w in body.persisted.items() if ch in linked}

    def consumers_of(entry: dict) -> list[str]:
        # with the body, a persisted channel's writer is reached through ``writes`` flows instead
        drop = persisted.get(entry["name"])
        return [q for q in _services(entry, "consumer_services") if q != drop]

    for entry in catalog:
        producers = _services(entry, "producer_services")
        consumers = consumers_of(entry)
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
            consumers = consumers_of(by_channel[channel])
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

    unmatched: dict[str, list[str]] = {}
    if body is not None:
        unmatched = _add_body(body, parts, flows, link, now_epoch=now_epoch, window=live_window_sec)

    flow_models = [
        FlowV1(
            source=f"part:{s}",
            target=f"part:{t}",
            weight=round(acc.weight, 4),
            evidence=sorted(acc.evidence),
            channels=sorted(acc.channels),
            causal_count=acc.causal_count,
            relation=acc.relation,
        )
        for (s, t), acc in sorted(flows.items())
    ]
    host_of = _placement(flows, parts)

    # Undirected organ graph over every non-transient part except hosts (isolated parts become
    # singletons). Hosts stay out: athena runs ~130 containers, so placement edges would glue
    # the body into one blob, and where a part runs is a different axis from what it does.
    body_parts = sorted(n for n, p in parts.items() if p.status != "transient" and p.kind != "host")
    adj: dict[str, dict[str, float]] = {n: {} for n in body_parts}
    for (s, t), acc in flows.items():
        if s in adj and t in adj:
            adj[s][t] = adj[s].get(t, 0.0) + acc.weight
            adj[t][s] = adj[t].get(s, 0.0) + acc.weight
    com = louvain(adj)
    q = modularity(adj, com) if adj else 0.0
    strength = {n: sum(adj[n].values()) for n in body_parts}

    groups: dict[int, list[str]] = defaultdict(list)
    for n in body_parts:
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
        kinds = Counter(parts[n].kind for n in members)
        placed = Counter(host_of[n] for n in members if n in host_of)
        organs.append(OrganV1(
            organ_id=organ_id,
            members=[f"part:{n}" for n in members],
            hub=f"part:{members[0]}",
            kinds=dict(sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0]))),
            placement=dict(sorted(placed.items(), key=lambda kv: (-kv[1], kv[0]))),
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

    in_organs = len(body_parts)
    hosts = sum(1 for p in parts.values() if p.kind == "host")
    return BodyAtlasV1(
        as_of=now.astimezone(timezone.utc).isoformat(),
        sources=AtlasSourcesV1(
            catalog_path=catalog_path,
            catalog_channels=len(catalog),
            live=observed is not None,
            live_graph=live_graph if observed is not None else None,
            publish_edges=len(observed.publishes) if observed else 0,
            causal_edges=len(observed.causal) if observed else 0,
            reads=list(body.reads) if body is not None else [],
        ),
        parts=list(parts.values()),
        flows=flow_models,
        organs=organs,
        organ_flows=organ_flows,
        stats=AtlasStatsV1(
            parts_total=len(parts),
            parts_in_organs=in_organs,
            transient_parts=sum(1 for p in parts.values() if p.status == "transient"),
            hosts=hosts,
            flows=len(flow_models),
            organs=len(organs),
            singleton_organs=sum(1 for o in organs if len(o.members) == 1),
            modularity=round(q, 4),
            largest_organ_share=round(max((len(o.members) for o in organs), default=0) / in_organs, 4)
            if in_organs else 0.0,
        ),
        known_gaps=list(body.known_gaps) if body is not None else [],
        unmatched=unmatched,
        missing_declared_stores=list(body.missing_declared_stores) if body is not None else [],
        redis_dynamic=dict(body.redis_dynamic) if body is not None else {},
        redis_private=dict(body.redis_private) if body is not None else {},
    )


def _add_body(body: BodyInputs, parts: dict[str, PartV1], flows: dict, link, *, now_epoch: float,
              window: int) -> dict[str, list[str]]:
    """Hosts, cameras, stores and processes as parts, and their flows.

    Each new part contributes at most ONE unit of flow weight, split evenly over its edges (the
    channel-hyperedge rule from v1): a store eight services touch adds 1/8 per edge, not 8, so
    no store, camera or process can glue the body together. A declared bus channel is also one
    unit, an observed one log10(1 + messages), so circulation still dominates organ structure.
    Returns raw names that matched no part, by source, for the reader to see."""
    known = {n for n, p in parts.items() if p.status != "transient"}
    host_names = [h.name for h in body.hosts]
    unmatched: dict[str, set[str]] = defaultdict(set)

    def add(name: str, **kw) -> None:
        parts[name] = PartV1(part_id=f"part:{name}", name=name, **kw)

    def spread(part: str, edges: list[tuple[str, str, str, str, str]]) -> None:
        # edges: (src, dst, relation, evidence, via)
        if not edges:
            return
        unit = BODY_PART_UNIT / len(edges)
        for src, dst, relation, evidence, via in edges:
            link(src, dst, unit, evidence, via or None, relation)

    # hosts + placement (runs_on)
    for h in body.hosts:
        status = _status(h.last_seen_epoch, now_epoch, window, seen=False) if h.monitored else "declared_only"
        facts = {"role": h.role, "hardware": h.hardware or "not recorded",
                 "telemetry": "biometrics" if h.monitored else "none"}
        if h.last_seen_epoch:
            facts["last_seen"] = _iso(h.last_seen_epoch) or ""
        add(f"host:{h.name}", kind="host", status=status, declared=True, facts=facts,
            last_seen_at=_iso(h.last_seen_epoch))
    hosts_by_alias = {a: h.name for h in body.hosts for a in h.aliases}
    per_host: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for pl in body.placements:
        host = hosts_by_alias.get(pl.host.lower())
        svc = match_part(pl.service, known, host_names)
        if host is None or svc is None or parts[svc].kind not in ("service", "endpoint"):
            unmatched[pl.evidence].add(f"{pl.service}@{pl.host}")
            continue
        per_host[host][pl.evidence].add(svc)
        link(svc, f"host:{host}", 1.0, pl.evidence, None, "runs_on")
    for host, ev in per_host.items():
        facts = parts[f"host:{host}"].facts
        if ev.get("observed_placement"):
            facts["services_observed"] = str(len(ev["observed_placement"]))
        if ev.get("declared_placement"):
            facts["services_declared"] = str(len(ev["declared_placement"]))

    # cameras (sees_through: camera -> reading service)
    for s in body.sensors:
        name = f"camera:{s.name}"
        edges, readers = [], set()
        for raw, where in s.readers:
            svc = match_part(raw, known, host_names)
            if svc is None:
                unmatched["declared_config"].add(raw)
                continue
            if svc not in readers:
                readers.add(svc)
                edges.append((name, svc, "sees_through", "declared_config", ""))
        facts = {"stream": s.name, "readers": str(len(readers))}
        on = [h.name for h in body.hosts if s.name in h.cameras]
        if on:
            facts["on_host"] = ",".join(on)
        if s.events is not None:
            facts["events_30d"] = str(s.events)
        add(name, kind="sensor", status=_status(s.last_seen_epoch, now_epoch, window, seen=s.events is not None),
            declared=True, last_seen_at=_iso(s.last_seen_epoch), facts=facts)
        spread(name, edges)
        for h in on:  # placement, like a service's: never an organ edge (hosts are not in organs)
            link(name, f"host:{h}", 1.0, "declared_placement", None, "runs_on")

    # stores (writes: producer -> store; reads: service -> store it reads; uses: direction
    # unknown). One edge per service: its strongest relation (writes > reads > uses), every
    # evidence kept. A route link's ``via`` is the persisted channel; a code link's is a module.
    rank = {"writes": 2, "reads": 1, "uses": 0}
    # Redis keys get their own rarity table: adding them must not move the shares of the stores
    # that were already weighed (N and df are per table)
    store_idf = {**_store_source_idf([s for s in body.stores if s.store != "redis"]),
                 **{f"redis|{k}": v for k, v in _store_source_idf([s for s in body.stores if s.store == "redis"]).items()}}
    for st in body.stores:
        name = f"{_STORE_PREFIX[st.store]}:{st.name}"
        per_svc: dict[str, list] = {}
        for raw, relation, evidence, via in st.links:
            svc = match_part(raw, known, host_names)
            if svc is None:
                unmatched[evidence].add(raw)
                continue
            cur = per_svc.get(svc)
            channel = via if evidence == "declared_route" else ""
            idf = store_idf[f"redis|{via}" if st.store == "redis" else via]
            if cur is None:
                per_svc[svc] = [relation, [evidence], channel, idf]
                continue
            if rank.get(relation, 0) > rank.get(cur[0], 0):
                cur[0] = relation
            if evidence not in cur[1]:
                cur[1].append(evidence)
            cur[2] = cur[2] or channel
            cur[3] = max(cur[3], idf)  # the service's most specific source counts
        status = "declared_only" if st.rows is None else ("live" if st.rows > 0 else "dormant")
        # Postgres rows are pg_stat_user_tables.n_live_tup: the statistics' estimate, not a count.
        # A Redis key holds a value or not (live: present); its facts say type, idle and ttl.
        if st.store == "redis":
            facts = {"store": "redis", **(st.facts or {"value": "not read"})}
        else:
            key = "rows_est" if st.store == "postgres" else "rows"
            facts = {"store": st.store, key: "not read" if st.rows is None else str(st.rows)}
        for rel, label in (("writes", "written_by"), ("reads", "read_by"), ("uses", "used_by")):
            who = sorted(svc for svc, v in per_svc.items() if v[0] == rel)
            if who:
                facts[label] = ", ".join(who[:3]) + (f" +{len(who) - 3}" if len(who) > 3 else "")
        if not per_svc and st.links:  # named, but by names that matched no part (see unmatched)
            facts["evidence"] = ("named by " + ", ".join(sorted({raw for raw, *_ in st.links})[:3]) +
                                 ", which matched no part")
        elif not per_svc:
            facts["evidence"] = "none: no service's code, env or route names it"
            if st.named_in_unimported:
                facts["evidence"] += f"; named only in {', '.join(st.named_in_unimported[:2])}, which no service imports"
        add(name, kind="store", status=status, declared=True, facts=facts)
        # one unit, split by how specific each service's source is (see _store_source_idf)
        total = sum(v[3] for v in per_svc.values())
        for svc, (rel, ev, ch, idf) in per_svc.items():
            link(svc, name, BODY_PART_UNIT * idf / total, ev[0], ch or None, rel)
            flows[(svc, name)].evidence.update(ev)

    _add_calls(body, parts, known, host_names, link, flows, unmatched)

    # processes: parts first (a spawns flow needs its child to exist), then flows
    for pr in body.processes:
        name = f"{_PROCESS_PREFIX[pr.family]}:{pr.name}"
        facts = {"family": pr.family, "runner": pr.runner or "unknown"}
        if pr.runs is not None:
            facts["runs_30d"] = str(pr.runs)
        if pr.last_run_epoch:
            facts["last_run"] = _iso(pr.last_run_epoch) or ""
        add(name, kind="process", declared=pr.declared, last_seen_at=_iso(pr.last_run_epoch), facts=facts,
            status=_status(pr.last_run_epoch, now_epoch, window, seen=pr.runs is not None))
    edges = {f"{_PROCESS_PREFIX[pr.family]}:{pr.name}": _process_edges(pr, parts, known, host_names, unmatched)
             for pr in body.processes}
    for _name, (src, dst, relation, evidence, weight) in _weigh_process_edges(edges):
        if src == dst:
            continue
        link(src, dst, weight, evidence[0], None, relation)
        flows[(src, dst)].evidence.update(evidence)
    for name, cands in edges.items():
        if len(cands) > 1:  # a runner-only process would just repeat its runner fact
            parts[name].facts["work_split"] = ", ".join(
                f"{t} {share:.0%}" for t, share in _shares(edges, name)[:3])
    return {k: sorted(v) for k, v in sorted(unmatched.items())}


def _add_calls(body: BodyInputs, parts: dict[str, PartV1], known: set[str], host_names: list[str], link,
               flows: dict, unmatched: dict[str, set[str]]) -> None:
    """HTTP calls (caller -> callee, ``calls``). Each callee adds ONE unit, split over its callers
    by how specific each caller's source is (``log(1 + N / df)``, df = how many of the N callees
    that module names): a widely-called service (orion-notify, called by a dozen) cannot glue its
    callers into one organ, and a client module naming every service keeps a small share of each.
    A host that matches no service part (``127.0.0.1``, a tailscale address, an outside site) is
    listed under ``unmatched["declared_url"]``, never guessed."""
    per_callee: dict[str, dict[str, str]] = defaultdict(dict)  # callee -> caller -> via
    for c in body.calls:
        callee = match_part(c.host, known, host_names) if c.host else None
        caller = match_part(c.caller, known, host_names)
        if callee is None or parts[callee].kind not in ("service", "endpoint"):
            unmatched["declared_url"].add(f"{c.host or c.url} (named by {c.caller})")
            continue
        if caller is None:
            unmatched["declared_url"].add(f"caller {c.caller} (names {callee})")
            continue
        if caller != callee:
            per_callee[callee].setdefault(caller, c.via)
    df: Counter = Counter()
    for callers in per_callee.values():
        for via in set(callers.values()):
            df[via] += 1
    n = max(len(per_callee), 1)
    for callee, callers in sorted(per_callee.items()):
        score = {caller: math.log(1 + n / df[via]) for caller, via in callers.items()}
        total = sum(score.values())
        for caller in sorted(callers):
            w = BODY_PART_UNIT * score[caller] / total
            acc = flows.get((caller, callee))
            if acc is not None and acc.relation != "calls":
                # the pair already has a flow (it signals on the bus): one flow per pair, so it
                # keeps its relation, and declared_url in its evidence says it also calls
                acc.weight += w
                acc.evidence.add("declared_url")
                continue
            link(caller, callee, w, "declared_url", None, "calls")


def _store_source_idf(stores) -> dict[str, float]:
    """Source (a link's ``via``: env file, code module, persisted channel) -> log(1 + N / df),
    df = how many of the N stores that source names. The process rule's rarity factor, applied
    to stores: a registry module naming every collection (orion/memory/vector_collection_policy.py,
    imported by orion-biometrics) keeps a small share of each, and the module that names one
    store keeps most of it. Equal sources give equal shares: the plain one-unit split."""
    df: Counter = Counter()
    for st in stores:
        for via in {via for _, _, _, via in st.links}:
            df[via] += 1
    n = max(len(stores), 1)
    return {via: math.log(1 + n / k) for via, k in df.items()}


@dataclass
class _Doer:
    relation: str  # runs / does / feeds (part -> process) or spawns (process -> part)
    evidence: set[str]
    runs: int


def _process_edges(pr, parts: dict[str, PartV1], known: set[str], host_names: list[str],
                   unmatched: dict[str, set[str]]) -> dict[str, _Doer]:
    """part -> evidence that it does this process's work. The runner comes first, so a runner
    also seen in the traces keeps relation ``runs``."""
    name = f"{_PROCESS_PREFIX[pr.family]}:{pr.name}"
    cands: dict[str, _Doer] = {}

    def put(target: str, relation: str, evidence: str, runs: int) -> None:
        d = cands.setdefault(target, _Doer(relation, set(), 0))
        d.evidence.add(evidence)
        d.runs = max(d.runs, runs)

    if pr.runner:
        runner = match_part(pr.runner, known, host_names)
        if runner is None:
            unmatched[pr.runner_evidence].add(pr.runner)
        else:
            put(runner, "runs", pr.runner_evidence, pr.runs or 1)
    for raw, evidence, runs in pr.doers:
        svc = match_part(raw, known, host_names)
        if svc is None or parts[svc].kind not in ("service", "endpoint"):
            unmatched[evidence].add(raw)
            continue
        put(svc, "feeds" if evidence == "declared_reducer_source" else "does", evidence, runs)
    for child, runs in pr.children:
        child_part = f"run:{child}"
        if child_part in parts and child_part != name:
            put(child_part, "spawns", "observed_child_run", runs)
        else:
            unmatched["observed_child_run"].add(child)
    return cands


def _scores(edges: dict[str, dict[str, _Doer]], name: str) -> dict[str, float]:
    """Each candidate's score: log(1 + runs) x log(1 + N / df), where df is how many of the N
    processes with any evidence the candidate appears in (inverse document frequency). A part in
    every process (the scheduler, the GPU lease) keeps a small share; one only this process uses
    keeps a large one. Same spirit as the channel hyperedge rule: plumbing cannot glue."""
    n = sum(1 for c in edges.values() if c)
    df = Counter(t for c in edges.values() for t in c)
    return {t: math.log1p(max(d.runs, 1)) * math.log1p(n / df[t]) for t, d in edges[name].items()}


def _shares(edges: dict[str, dict[str, _Doer]], name: str) -> list[tuple[str, float]]:
    scores = _scores(edges, name)
    total = sum(scores.values())
    return sorted(((t, s / total) for t, s in scores.items()), key=lambda kv: (-kv[1], kv[0]))


def _weigh_process_edges(edges: dict[str, dict[str, _Doer]]):
    """(process, (src, dst, relation, evidence, weight)): each process adds at most one unit
    (BODY_PART_UNIT), split over its candidates by score."""
    for name, cands in sorted(edges.items()):
        if not cands:
            continue
        for t, share in _shares(edges, name):
            d = cands[t]
            src, dst = (name, t) if d.relation == "spawns" else (t, name)
            yield name, (src, dst, d.relation, sorted(d.evidence), BODY_PART_UNIT * share)


def _placement(flows: dict, parts: dict[str, PartV1]) -> dict[str, str]:
    """part name -> host name, observed placement first, then declared."""
    out: dict[str, str] = {}
    for prefer in ("observed_placement", "declared_placement"):
        for (s, t), acc in sorted(flows.items()):
            if acc.relation == "runs_on" and prefer in acc.evidence and s not in out:
                out[s] = parts[t].name.removeprefix("host:")
    return out
