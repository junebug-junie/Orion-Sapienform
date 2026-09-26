#!/usr/bin/env python3
"""Read-only agency episode audit; use --input for an offline metadata replay.

Live mode reads PostgreSQL through ORION_PG_DSN and the local worldview graph.
It never connects to the event bus, dispatches, marks consumed or updates state.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orion.autonomy.agency_episode import reconstruct


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Replay a metadata snapshot; no connections")
    parser.add_argument("--snapshot", type=Path, help="Save metadata snapshot to a NEW owner-only local file")
    parser.add_argument("--limit", type=int, default=10, choices=range(1, 51), metavar="1..50")
    parser.add_argument("--graph-host", default="127.0.0.1")
    parser.add_argument("--graph-port", type=int, default=6380)
    parser.add_argument("--graph-name", default="orion_worldview")
    args = parser.parse_args()
    try:
        if args.input:
            bundle = json.loads(args.input.read_text())
        else:
            import psycopg2
            from orion.autonomy.agency_episode_reader import collect
            from orion.curiosity.worldview import WorldviewReader

            dsn = os.environ.get("ORION_PG_DSN")
            if not dsn:
                parser.error("Live reads require ORION_PG_DSN; use --input for replay.")
            conn = psycopg2.connect(dsn, connect_timeout=5,
                options="-c default_transaction_read_only=on -c statement_timeout=5000")
            conn.autocommit = True
            reader = WorldviewReader(host=args.graph_host, port=args.graph_port, graph_name=args.graph_name)
            try:
                bundle = collect(conn, reader, limit=args.limit)
            finally:
                conn.close()
                if reader._client is not None:
                    reader._client.close()
        report = reconstruct(bundle)
        if args.snapshot:
            fd = os.open(args.snapshot, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as output:
                json.dump(bundle, output, indent=2, default=str, sort_keys=True)
                output.write("\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2 if report["conflicts"] or any(s["status"] != "ok" for s in report["sources"].values()) else 0
    except Exception as exc:
        print(f"Agency audit failed ({type(exc).__name__}); no complete report. Connection details suppressed.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
