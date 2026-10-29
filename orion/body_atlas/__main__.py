"""``python -m orion.body_atlas``: build the atlas, print a capped summary, write the full JSON.

The printout stays under ``MAX_LINES`` / ``MAX_CHARS`` so a chat or curiosity turn can run this
through Bash without flooding its context; the full atlas goes to ``--out``. A requested live
read that fails is an error, never a quiet declared-only atlas.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from orion.body_atlas.build import CATALOG_RELPATH, build_atlas, fetch_observed, load_catalog
from orion.body_atlas.model import BodyAtlasV1
from orion.body_atlas.sources import default_live_readers, gather_body

MAX_LINES = 40
MAX_CHARS = 4000
DEFAULT_GRAPH = "orion_bus_synapse"
_REPO_ROOT = Path(__file__).resolve().parents[2]


class _RoGraphReader:
    """``GRAPH.RO_QUERY`` over plain redis-py, for scalar-column reads only.

    orion.graph.falkor_client.RedisGraphQueryClient needs ``redis.commands.graph``, which
    redis-py 6+ dropped; the harness container (where chat and curiosity run this) has 8.1.
    Without ``--compact`` a collection column would come back as its string repr (see that
    client's docstring), so this reader is only safe for the scalar queries in build.py.
    RO_QUERY makes FalkorDB itself refuse any write.
    """

    def __init__(self, uri: str, graph: str) -> None:
        import redis

        self._r = redis.Redis.from_url(uri, decode_responses=True, socket_timeout=10.0, socket_connect_timeout=3.0)
        self._graph = graph

    def graph_query(self, cypher: str) -> list[dict]:
        reply = self._r.execute_command("GRAPH.RO_QUERY", self._graph, cypher)
        header, rows = reply[0], reply[1] if len(reply) > 1 else []
        return [dict(zip(header, row)) for row in rows]

    def close(self) -> None:
        self._r.close()


_KIND_ORDER = ("service", "endpoint", "sensor", "store", "process")
_KIND_SHORT = {"service": "svc", "endpoint": "endpoint", "sensor": "camera", "store": "store", "process": "process"}


def _kinds(counts: dict[str, int]) -> str:
    return ", ".join(f"{counts[k]} {_KIND_SHORT.get(k, k)}" for k in _KIND_ORDER if counts.get(k))


def _cap(lines: list[str], tail: str) -> str:
    """At most MAX_LINES lines (tail included) and MAX_CHARS chars; the tail line always stays,
    because it is how the reader finds what was cut."""
    if len(lines) > MAX_LINES - 1:
        lines = lines[: MAX_LINES - 2] + [f"(showing {MAX_LINES - 2} of {len(lines)} lines; the rest is in the file)"]
    text = "\n".join(lines)
    if len(text) + 1 + len(tail) > MAX_CHARS:
        note = "(cut at 4000 chars)"
        text = text[: MAX_CHARS - len(tail) - len(note) - 2].rsplit("\n", 1)[0] + "\n" + note
    return text + "\n" + tail


def _host_line(p) -> str:
    f = p.facts
    svc = " ".join(x for x in (f"{f['services_observed']} running" if "services_observed" in f else "",
                               f"{f['services_declared']} declared" if "services_declared" in f else "") if x)
    seen = f" last seen {p.last_seen_at[:16]}" if p.last_seen_at and p.status != "live" else ""
    return (f"  {p.name.removeprefix('host:')}: {f.get('role', '?')}, {f.get('hardware', 'not recorded')}; "
            f"{p.status}{seen}, telemetry {f.get('telemetry', '?')}" + (f"; services {svc}" if svc else ""))


def summarize(atlas: BodyAtlasV1, *, out: str | None) -> str:
    s = atlas.stats
    src = atlas.sources
    whole = bool(src.reads)
    mode = ("live bus + catalog" if src.live else "catalog only") + (" + whole body" if whole else "")
    kinds = Counter(p.kind for p in atlas.parts if p.status != "transient" and p.kind != "host")
    rel = Counter(f.relation for f in atlas.flows)
    lines = [
        f"body atlas as of {atlas.as_of} ({mode})",
        f"parts {s.parts_total}: {_kinds(kinds)}, {s.hosts} hosts, {s.transient_parts} transient · "
        f"organs {s.organs} ({s.singleton_organs} singletons)",
        "flows " + ", ".join(f"{n} {r}" for r, n in sorted(rel.items(), key=lambda kv: -kv[1])) +
        f" · Q={s.modularity} (~0.3+ = real organs) · largest organ {s.largest_organ_share:.0%}",
    ]
    hosts = [p for p in atlas.parts if p.kind == "host"]
    if hosts:
        lines.append("bodies (machines; not in organs):")
        lines += [_host_line(p) for p in sorted(hosts, key=lambda p: (p.status != "live", p.name))]
    unread = [r for r in src.reads if r.status == "unread"]
    if unread:
        lines.append(f"UNREAD {len(unread)} source(s), this body is missing what they hold:")
        lines += [f"  {r.name}: {r.detail[:110]}" for r in unread]
    names = {p.part_id: p.name for p in atlas.parts}
    status = {p.part_id: p.status for p in atlas.parts}
    multi = [o for o in atlas.organs if len(o.members) > 1]
    footer = []
    if atlas.known_gaps:
        footer.append(f"not seen (no source yet): {len(atlas.known_gaps)} gaps, listed in the full atlas")
    if multi:
        footer.append("every member of one organ: python -P -m orion.dig body <organ_id>")
    budget = MAX_LINES - len(lines) - len(footer) - 2
    for i, organ in enumerate(multi[: max(budget, 0)], start=1):
        live = sum(1 for m in organ.members if status[m] == "live")
        top = ", ".join(names[m] for m in organ.members[:5])
        more = f" +{len(organ.members) - 5}" if len(organ.members) > 5 else ""
        on = " on " + " ".join(f"{h} {n}" for h, n in organ.placement.items()) if organ.placement else ""
        label = f'"{organ.name}" ' if organ.name else ""
        lines.append(f"{i:>2}. {label}{organ.organ_id} [{len(organ.members)}: {_kinds(organ.kinds) or '-'}; "
                     f"{live} live]{on}: {top}{more}")
    if len(multi) > max(budget, 0):
        lines.append(f"showing {max(budget, 0)} of {len(multi)} multi-part organs")
    lines += footer
    tail = f"full atlas: {out}" if out else "full atlas: not written (pass --out PATH)"
    return _cap(lines, tail)


def organ_view(atlas: BodyAtlasV1, ref: str, *, out: str | None) -> str:
    """Every member of one organ, grouped by kind, with status and facts; capped like the body."""
    from orion.body_atlas.names import find_organ

    organ = find_organ(atlas, ref)
    by_id = {p.part_id: p for p in atlas.parts}
    hosts_of: dict[str, set[str]] = {}
    for f in atlas.flows:
        if f.relation == "runs_on":
            hosts_of.setdefault(f.source, set()).add(by_id[f.target].name.removeprefix("host:"))
    label = f' "{organ.name}"' if organ.name else ""
    on = " · on " + ", ".join(f"{h} {n}" for h, n in organ.placement.items()) if organ.placement else ""
    lines = [f"organ {organ.organ_id}{label}: {len(organ.members)} members ({_kinds(organ.kinds)}){on}",
             f"hub {by_id[organ.hub].name} · namespaces {'/'.join(organ.namespaces) or '-'} · "
             f"internal {organ.internal_weight} / external {organ.external_weight}"]
    for kind in _KIND_ORDER:
        members = [by_id[m] for m in organ.members if by_id[m].kind == kind]
        if not members:
            continue
        lines.append(f"{kind} ({len(members)}), strongest first:")
        for p in members:
            facts = ", ".join(f"{k}={v}" for k, v in p.facts.items() if k not in ("store", "family", "on_host"))
            on = f" on {','.join(sorted(hosts_of[p.part_id]))}" if p.part_id in hosts_of else ""
            lines.append(f"  {p.name} [{p.status}]{on}" + (f" {facts}" if facts else ""))
    tail = f"full atlas (flows too): {out}" if out else "full atlas: not written (pass --out PATH)"
    return _cap(lines, tail)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -P -m orion.body_atlas", description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=_REPO_ROOT, help="repo checkout to read the catalog from")
    ap.add_argument("--no-live", action="store_true", help="catalog only (e.g. a past checkout)")
    ap.add_argument("--falkor-uri", default=os.environ.get("FALKORDB_URI"), help="default: $FALKORDB_URI")
    ap.add_argument("--graph", default=os.environ.get("FALKORDB_BUS_GRAPH", DEFAULT_GRAPH))
    ap.add_argument("--out", type=Path, help="write the full atlas JSON here")
    ap.add_argument("--bus-only", action="store_true",
                    help="v1 atlas: bus parts only (no hosts, cameras, stores, processes)")
    ap.add_argument("--organ", help="print every member of this organ instead of the body")
    args = ap.parse_args(argv)

    catalog = load_catalog(args.root)
    if not catalog:
        print(f"no channel catalog at {args.root / CATALOG_RELPATH}", file=sys.stderr)
        return 2
    observed = None
    if not args.no_live:
        if not args.falkor_uri:
            print("live read needs --falkor-uri or $FALKORDB_URI (or pass --no-live)", file=sys.stderr)
            return 2
        client = _RoGraphReader(args.falkor_uri, args.graph)
        try:
            observed = fetch_observed(client)
        except Exception as exc:  # report the real failure; no declared-only fallback
            print(f"live read failed ({args.graph} at {args.falkor_uri}): {exc}", file=sys.stderr)
            return 1
        finally:
            client.close()
        if not observed.publishes:
            print(f"live graph {args.graph} has no PUBLISHES edges; is orion-bus-mirror writing it?",
                  file=sys.stderr)
            return 1

    body = None
    if not args.bus_only:
        live = None if args.no_live else default_live_readers(args.falkor_uri)
        body = gather_body(args.root, catalog, live=live)
    atlas = build_atlas(catalog, observed, now=datetime.now(timezone.utc),
                        catalog_path=str(args.root / CATALOG_RELPATH),
                        live_graph=args.graph if observed is not None else None, body=body)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(atlas.model_dump_json(indent=2), encoding="utf-8")
    out = str(args.out) if args.out else None
    if args.organ:
        from orion.body_atlas.names import NamingError

        try:
            print(organ_view(atlas, args.organ, out=out))
        except NamingError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        return 0
    print(summarize(atlas, out=out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
