"""``python -m orion.body_atlas``: build the atlas, print a capped summary, write the full JSON.

The printout stays under ``MAX_LINES`` / ``MAX_CHARS`` so a chat or curiosity turn can run this
through Bash without flooding its context; the full atlas goes to ``--out``. A requested live
read that fails is an error, never a quiet declared-only atlas.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from orion.body_atlas.build import CATALOG_RELPATH, build_atlas, fetch_observed, load_catalog
from orion.body_atlas.model import BodyAtlasV1

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


def summarize(atlas: BodyAtlasV1, *, out: str | None) -> str:
    s = atlas.stats
    src = atlas.sources
    lines = [
        f"body atlas as of {atlas.as_of} ({'live bus + catalog' if src.live else 'catalog only'})",
        f"parts {s.parts_total} ({s.parts_in_organs} in organs, {s.transient_parts} transient) · "
        f"flows {s.flows} · organs {s.organs} ({s.singleton_organs} singletons)",
        f"modularity Q={s.modularity} (~0.3+ = real organs) · largest organ {s.largest_organ_share:.0%} of parts",
    ]
    names = {p.part_id: p.name for p in atlas.parts}
    status = {p.part_id: p.status for p in atlas.parts}
    multi = [o for o in atlas.organs if len(o.members) > 1]
    budget = MAX_LINES - len(lines) - 2
    for i, organ in enumerate(multi[:budget], start=1):
        live = sum(1 for m in organ.members if status[m] == "live")
        shown = ", ".join(names[m] for m in organ.members[:4])
        more = f" +{len(organ.members) - 4}" if len(organ.members) > 4 else ""
        ns = "/".join(organ.namespaces) or "-"
        lines.append(f"{i:>2}. [{len(organ.members)} parts, {live} live] ns={ns}: {shown}{more}")
    if len(multi) > budget:
        lines.append(f"showing {budget} of {len(multi)} multi-part organs")
    tail = f"full atlas: {out}" if out else "full atlas: not written (pass --out PATH)"
    text = "\n".join(lines)
    if len(text) + 1 + len(tail) > MAX_CHARS:
        # keep the path line: it is how the reader finds what was cut
        note = "(cut at 4000 chars)"
        text = text[: MAX_CHARS - len(tail) - len(note) - 2].rsplit("\n", 1)[0] + "\n" + note
    return text + "\n" + tail


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -P -m orion.body_atlas", description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=_REPO_ROOT, help="repo checkout to read the catalog from")
    ap.add_argument("--no-live", action="store_true", help="catalog only (e.g. a past checkout)")
    ap.add_argument("--falkor-uri", default=os.environ.get("FALKORDB_URI"), help="default: $FALKORDB_URI")
    ap.add_argument("--graph", default=os.environ.get("FALKORDB_BUS_GRAPH", DEFAULT_GRAPH))
    ap.add_argument("--out", type=Path, help="write the full atlas JSON here")
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

    atlas = build_atlas(catalog, observed, now=datetime.now(timezone.utc),
                        catalog_path=str(args.root / CATALOG_RELPATH),
                        live_graph=args.graph if observed is not None else None)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(atlas.model_dump_json(indent=2), encoding="utf-8")
    print(summarize(atlas, out=str(args.out) if args.out else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
