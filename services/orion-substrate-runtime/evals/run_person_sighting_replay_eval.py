"""Person sighting replay eval: does "seen recently" rise, fade, and stay spread out?

docs/superpowers/specs/2026-11-03-person-graph-sighting-refresh-design.md
acceptance 4, 5 and 7. Read-only in both modes.

The replay builds an in-memory substrate graph with one named cluster node per
linked camera cluster and one fenced person node per person. It steps time in
10-minute passes. Each pass runs the REAL refresh step
(app/person_graph_loop.py ``_refresh_sightings``, with each link's
``last_seen_at`` = its newest sighting so far) and then the REAL
``SubstrateDynamicsEngine.tick``. A person is active in an hour when any of
their clusters reads above 0.2 at any pass in that hour.

Pass criteria:

- domination: no single person holds more than 25% of the active person-hours;
- flood guard: a pass with no new sighting writes nothing;
- fenced: every person node's activation is byte-equal before and after;
- rest: a cluster with no sighting in the last 24 h of the replay is dormant
  (<= 0.08) at the end.

``--synthetic`` (default; also run by test_person_sighting_replay_eval.py):
three made-up people over 3 days. ``--live``: the last ``--days`` (default 14)
of walkway hour rollups from Postgres (``vision_rollup`` grain=hour, the
``named`` entry's individual ids and HH:MM times), read in a READ ONLY
transaction. Nothing is written anywhere.

Run:
  python services/orion-substrate-runtime/evals/run_person_sighting_replay_eval.py
  POSTGRES_URI=postgresql://... python .../run_person_sighting_replay_eval.py --live --days 14
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
SERVICE = HERE.parents[1]
REPO = HERE.parents[3]
for _p in (str(REPO), str(SERVICE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from orion.substrate.adapters.vision_individual import (  # noqa: E402
    entity_node_id_for_individual,
    map_vision_individual_label_to_substrate,
)
from orion.substrate.dynamics import SubstrateDynamicsEngine  # noqa: E402
from orion.substrate.materializer import SubstrateGraphMaterializer  # noqa: E402
from orion.substrate.store import InMemorySubstrateGraphStore  # noqa: E402

STEP = timedelta(minutes=10)
ACTIVE_THRESHOLD = 0.2
MAX_PERSON_SHARE = 0.25
DORMANT = 0.08

# sightings: person_id -> individual_id -> sorted sighting times.
Sightings = dict[str, dict[str, list[datetime]]]


def replay(sightings: Sightings, *, start: datetime, end: datetime,
           names: dict[str, str] | None = None) -> dict[str, Any]:
    from app.person_graph_loop import LinkRow, PersonRow, _refresh_sightings, person_node

    names = names or {}
    store = InMemorySubstrateGraphStore()
    materializer = SubstrateGraphMaterializer(store=store)
    named_at = start - timedelta(days=1)
    persons = [PersonRow(pid, names.get(pid, pid), "active", named_at) for pid in sorted(sightings)]
    referent = _referent_node(named_at)
    materializer.apply_record(_record_of(referent))
    for p in persons:
        materializer.apply_record(_record_of(person_node(p, now=named_at)))
        for iid in sorted(sightings[p.person_id]):
            # Today's live shape: no half-life and no label_set_at. The first pass upgrades it.
            materializer.apply_record(_record_of(_legacy_shape(map_vision_individual_label_to_substrate(
                individual_id=iid, label=p.display_name, kind="person", ask_id=f"ask-{iid}",
                answered_at=named_at).nodes[0])))
            _link_fenced(store, entity_node_id_for_individual(iid), p.person_id, referent.node_id, named_at)
    engine = SubstrateDynamicsEngine(store=store)
    # Before ANY tick: a broken fence lifts the target on the very first one.
    fenced_ids = [p.person_id for p in persons] + [referent.node_id]
    fenced_before = {nid: store.get_node_by_id(nid).signals.activation.model_dump() for nid in fenced_ids}
    # Birth tick at naming, as live: every cluster takes its frozen 0.625-shaped seed.
    engine.tick(now=named_at + timedelta(minutes=1))

    clusters = {iid: pid for pid, by in sightings.items() for iid in by}
    active: dict[str, set[datetime]] = defaultdict(set)
    flood_writes = 0
    passes = 0
    t = start
    prev_seen: dict[str, datetime | None] = {}
    while t <= end:
        links = []
        new_sighting = False
        for iid, pid in sorted(clusters.items()):
            times = [s for s in sightings[pid][iid] if s <= t]
            seen = max(times) if times else None
            if seen is not None and seen != prev_seen.get(iid):
                new_sighting = True
            prev_seen[iid] = seen
            links.append(LinkRow(f"l-{iid}", pid, iid, last_seen_at=seen))
        written, _ = asyncio.run(_refresh_sightings(persons, links, store=store))
        if passes > 0 and not new_sighting and written:
            flood_writes += written
        passes += 1
        engine.tick(now=t)
        hour = t.replace(minute=0, second=0, microsecond=0)
        for iid, pid in clusters.items():
            node = store.get_node_by_id(entity_node_id_for_individual(iid))
            if node.signals.activation.activation > ACTIVE_THRESHOLD:
                active[pid].add(hour)
        t += STEP

    fenced_after = {nid: store.get_node_by_id(nid).signals.activation.model_dump() for nid in fenced_ids}
    hours = {pid: len(active.get(pid, ())) for pid in sightings}
    total = sum(hours.values())
    shares = {pid: (h / total if total else 0.0) for pid, h in hours.items()}
    top = max(shares, key=lambda k: shares[k]) if shares else None
    quiet_cutoff = end - timedelta(hours=24)
    not_resting = []
    for iid, pid in clusters.items():
        last = max(sightings[pid][iid], default=None)
        if last is None or last <= quiet_cutoff:
            value = store.get_node_by_id(entity_node_id_for_individual(iid)).signals.activation.activation
            if value > DORMANT:
                not_resting.append((iid, round(value, 4)))
    return {
        "start": start.isoformat(), "end": end.isoformat(), "passes": passes,
        "persons": len(sightings), "clusters": len(clusters),
        "replay_hours": int((end - start).total_seconds() // 3600) + 1,
        "active_person_hours": total,
        "active_hours_by_person": {names.get(pid, pid): h for pid, h in sorted(hours.items(), key=lambda kv: -kv[1])},
        "top_person": names.get(top, top) if top else None,
        "top_share": round(shares[top], 4) if top else 0.0,
        "max_share": MAX_PERSON_SHARE,
        "flood_writes": flood_writes,
        "fenced_changed": sorted(pid for pid in fenced_before if fenced_before[pid] != fenced_after[pid]),
        "not_resting": not_resting,
    }


REFERENT_ID = "referent-replay-0000"


def _referent_node(at: datetime) -> Any:
    """A chat referent (people.yaml person) like Juniper's: fenced by its producer."""
    from orion.core.schemas.cognitive_substrate import EntityNodeV1, SubstrateProvenanceV1, SubstrateTemporalWindowV1

    return EntityNodeV1(
        node_id=REFERENT_ID, label="Referent", entity_type="person", anchor_scope="world",
        subject_ref=f"referent:{REFERENT_ID}", temporal=SubstrateTemporalWindowV1(observed_at=at),
        provenance=SubstrateProvenanceV1(authority="user_asserted", source_kind="memory_referent",
                                         source_channel="people.yaml", producer="memory.referents"))


def _legacy_shape(node: Any) -> Any:
    from orion.core.schemas.cognitive_substrate import SubstrateActivationV1
    from orion.substrate.adapters.vision_individual import LABEL_SET_AT_KEY

    return node.model_copy(update={
        "metadata": {k: v for k, v in node.metadata.items() if k != LABEL_SET_AT_KEY},
        "signals": node.signals.model_copy(update={"activation": SubstrateActivationV1()}),
    })


def _link_fenced(store: Any, cluster_id: str, person_id: str, referent_id: str, at: datetime) -> None:
    """The live edges into fenced nodes: an instance_of claim (semantic_projection) to the person,
    and a plain, propagating observed_in edge to the referent. Neither may lift its target."""
    from orion.core.schemas.cognitive_substrate import NodeRefV1, SubstrateEdgeV1, SubstrateProvenanceV1
    from orion.substrate.adapters._common import make_temporal

    prov = SubstrateProvenanceV1(authority="user_asserted", source_kind="replay", source_channel="replay",
                                 producer="replay")
    for predicate, target, role in (("instance_of", person_id, "semantic_projection"),
                                    ("observed_in", referent_id, "legacy_unreviewed")):
        extra = {"assertion_id": f"assertion:{cluster_id}|{target}", "assertion_revision": 1} \
            if role == "semantic_projection" else {}
        edge = SubstrateEdgeV1(
            edge_id=f"replay-{predicate}-{cluster_id}-{target}",
            source=NodeRefV1(node_id=cluster_id, node_kind="entity"),
            target=NodeRefV1(node_id=target, node_kind="entity"),
            predicate=predicate, edge_role=role, temporal=make_temporal(observed_at=at),
            confidence=0.9, provenance=prov, **extra)
        store.upsert_edge(identity_key=edge.edge_id, edge=edge)


def _record_of(node: Any) -> Any:
    from orion.core.schemas.cognitive_substrate import SubstrateGraphRecordV1

    return SubstrateGraphRecordV1(anchor_scope="world", subject_ref=node.subject_ref, nodes=[node])


def verdict(r: dict[str, Any]) -> list[str]:
    fails = []
    if r["top_share"] > r["max_share"]:
        fails.append(f"domination: {r['top_person']} holds {r['top_share']:.1%} of active person-hours "
                     f"(> {r['max_share']:.0%})")
    if r["flood_writes"]:
        fails.append(f"flood: {r['flood_writes']} writes on passes with no new sighting")
    if r["fenced_changed"]:
        fails.append(f"fenced person node activation changed: {r['fenced_changed']}")
    if r["not_resting"]:
        fails.append(f"clusters not at rest 24 h after their last sighting: {r['not_resting'][:5]}")
    if not r["active_person_hours"]:
        fails.append("empty: no person was ever active (no sighting reached the graph)")
    return fails


def synthetic(*, dominant: bool = False) -> dict[str, Any]:
    """Five people, each seen twice over 3 days. ``dominant``: person A is also seen every
    3 hours, which must fail the domination check."""
    start = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(days=3)

    def at(day: int, hour: int, minute: int = 0) -> datetime:
        return start + timedelta(days=day, hours=hour, minutes=minute)

    sightings: Sightings = {
        "person-a": {"a1": [at(0, 9, 5), at(0, 9, 7)], "a2": [at(1, 9, 3)]},
        "person-b": {"b1": [at(0, 17, 22), at(1, 17, 40)]},
        "person-c": {"c1": [at(0, 12, 0), at(1, 1, 0)], "c2": []},
        "person-d": {"d1": [at(0, 3, 0), at(1, 6, 30)]},
        "person-e": {"e1": [at(0, 20, 15), at(1, 13, 0)]},
    }
    if dominant:
        sightings["person-a"]["a1"] = sorted(at(0, 0) + timedelta(hours=3 * k) for k in range(16))
    return replay(sightings, start=start, end=end, names={k: k[-1].upper() for k in sightings})


LIVE_ROLLUPS_SQL = (
    "SELECT period_start, rollup->'named' AS named FROM vision_rollup "
    "WHERE grain = 'hour' AND stream_id = %s AND period_start >= %s ORDER BY period_start"
)
LIVE_LINKS_SQL = (
    "SELECT l.person_id, l.source_ref, p.display_name FROM person_link l "
    "JOIN person p ON p.person_id = l.person_id "
    "JOIN vision_individual v ON v.individual_id = l.source_ref "
    "WHERE l.source_kind = 'vision_individual' AND l.revoked_at IS NULL "
    "AND p.status <> 'not_a_person' AND NOT coalesce(v.label_copied, false)"
)


def live_sightings(dsn: str, *, days: int, stream: str, now: datetime) -> tuple[Sightings, dict[str, str], datetime]:
    import psycopg2

    since = (now - timedelta(days=days)).replace(minute=0, second=0, microsecond=0)
    conn = psycopg2.connect(dsn)
    try:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            cur.execute(LIVE_LINKS_SQL)
            link_of: dict[str, str] = {}
            names: dict[str, str] = {}
            for pid, iid, name in cur.fetchall():
                link_of[iid] = pid
                names[pid] = name
            cur.execute(LIVE_ROLLUPS_SQL, (stream, since))
            rows = cur.fetchall()
    finally:
        conn.rollback()
        conn.close()
    sightings: Sightings = defaultdict(lambda: defaultdict(list))
    for period_start, named in rows:
        if period_start.tzinfo is None:
            period_start = period_start.replace(tzinfo=timezone.utc)
        for entry in (named or {}).values():
            entry = entry or {}
            minutes = []
            for hhmm in entry.get("times") or []:
                try:
                    minutes.append(int(str(hhmm).split(":")[1]))
                except (IndexError, ValueError):
                    continue
            stamps = [period_start + timedelta(minutes=m) for m in (minutes or [0])]
            for iid in entry.get("individual_ids") or []:
                pid = link_of.get(iid)
                if pid is None:
                    continue  # no live, non-copied link: no node to refresh
                sightings[pid][iid].extend(stamps)
    out = {pid: {iid: sorted(set(ts)) for iid, ts in by.items()} for pid, by in sightings.items()}
    return out, names, since


def _dsn(uri: str) -> str:
    scheme, sep, rest = uri.partition("://")
    return f"{scheme.split('+', 1)[0]}{sep}{rest}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--stream", default="walkway")
    ap.add_argument("--dsn", default=os.environ.get("POSTGRES_URI", ""))
    args = ap.parse_args(argv)
    if args.live:
        if not args.dsn:
            print("POSTGRES_URI or --dsn required for --live", file=sys.stderr)
            return 2
        now = datetime.now(timezone.utc)
        sightings, names, since = live_sightings(_dsn(args.dsn), days=args.days, stream=args.stream, now=now)
        result = replay(sightings, start=since, end=now, names=names)
    else:
        result = synthetic()
    fails = verdict(result)
    result["verdict"] = "PASS" if not fails else "FAIL"
    result["failures"] = fails
    print(json.dumps(result, indent=2, default=str))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
