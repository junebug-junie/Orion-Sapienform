#!/usr/bin/env python3
"""Rebuild the memory referent projection in the substrate graph from Postgres.

The only correct rebuild: it re-projects referent nodes and memory evidence (by clearing the
projector's ledger) AND replays every accepted co-occurrence assertion's latest applied
decision (truncating the ledger alone would leave those out: the journal records them as
already applied). Writes nothing new to the journal. Respects the reader readiness gate.

    python scripts/rebuild_referent_graph.py --dsn postgresql://... --falkor-uri redis://... [--graph orion_substrate]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "services" / "orion-memory-consolidation"))


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--falkor-uri", required=True)
    parser.add_argument("--graph", default="orion_substrate")
    args = parser.parse_args()

    import asyncpg

    from app.referent_projector import build_projector

    pool = await asyncpg.create_pool(dsn=args.dsn, min_size=1, max_size=2)
    try:
        settings = SimpleNamespace(FALKORDB_URI=args.falkor_uri, FALKORDB_SUBSTRATE_GRAPH=args.graph)
        projector = await asyncio.to_thread(build_projector, pool, settings)
        tick = await projector.rebuild()
        if tick.blocked is not None:
            print(f"refused: readers not ready, missing={list(tick.blocked.missing)}", file=sys.stderr)
            return 2
        print(f"rebuilt: nodes={tick.nodes} memories={tick.memories} assertions_replayed={tick.assertions_applied}")
        return 0
    finally:
        await pool.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
