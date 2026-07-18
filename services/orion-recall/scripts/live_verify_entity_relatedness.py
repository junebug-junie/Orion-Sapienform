#!/usr/bin/env python3
"""Re-runnable live-data sanity check for app/storage/falkor_entity_relatedness.py.

Per this repo's own metric-quality gate (CLAUDE.md section 0A, step 4: "Pull
real data and look at it... Config existing or code compiling is not
evidence"), the module's docstrings assert specific ranking behavior against
real orion_recall data (Jaccard correctly ranking "tesla" above "athena"
relative to "nvidia" despite athena's higher raw co-occurrence count). That
claim previously existed only as prose -- this script re-derives it from the
live graph so it's a checkable artifact, not a narrated one.

Run from inside a container with FALKORDB_URI/FALKORDB_RECALL_GRAPH set
(e.g. `docker exec orion-athena-recall python3 scripts/live_verify_entity_relatedness.py`).
Exits non-zero if the graph doesn't back up the documented claim -- this is a
sanity check, not a unit test with fixed expected data (the graph is live and
grows), so it asserts *structure* (Jaccard ranks the more-specific entity
above the higher-degree one) rather than exact scores.
"""
from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "/app")

from app.storage.falkor_entity_relatedness import (  # noqa: E402
    fetch_bridging_turns,
    fetch_entity_mention_timeline,
    fetch_related_entities,
)


async def main() -> int:
    related = await fetch_related_entities(name="nvidia", max_results=10)
    print("fetch_related_entities(nvidia):")
    for r in related:
        print(f"  {r['name']!r:20} shared={r['shared_turns']:3} jaccard={r['jaccard']:.4f}")

    by_name = {r["name"]: r for r in related}
    ok = True
    if "tesla" in by_name and "athena" in by_name:
        if by_name["tesla"]["jaccard"] <= by_name["athena"]["jaccard"]:
            print("FAIL: expected tesla's Jaccard score to rank above athena's (higher-degree hub entity)")
            ok = False
        else:
            print("OK: tesla ranks above athena, as documented")
    else:
        print("SKIP: tesla/athena no longer both co-occur with nvidia in the live graph -- graph has drifted, not a failure")

    bridge = await fetch_bridging_turns(entity_a="nvidia", entity_b="atlas")
    print(f"\nfetch_bridging_turns(nvidia, atlas): mode={bridge['mode']} results={len(bridge['results'])}")

    timeline = await fetch_entity_mention_timeline(name="nvidia", max_results=5)
    print(f"\nfetch_entity_mention_timeline(nvidia): {len(timeline)} mentions")
    for t in timeline:
        print(f"  {t['ts']}  {t['turn_id']}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
