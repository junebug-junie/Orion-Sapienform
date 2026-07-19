#!/usr/bin/env python3
"""Live before/after evidence for RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED,
run against real recall queries and the real orion_recall graph -- not a
single hand-picked example (the gap PR #1209/#1211 disclosed and left open).

Per this repo's metric-quality gate (CLAUDE.md section 0A step 4): pulls
real data and looks at it, rather than trusting the mechanism is correct
just because it's reachable. Runs each query through process_recall twice
(flag off, flag on) and reports whether/how the selected bundle changed.

Run from inside orion-athena-recall:
  docker exec orion-athena-recall python3 scripts/live_verify_entity_relatedness_boost.py
"""
from __future__ import annotations

import asyncio
import sys
from uuid import uuid4

sys.path.insert(0, "/app")

import app.settings as settings_mod
from app.worker import process_recall
from orion.core.contracts.recall import RecallQueryV1

# Real entities confirmed live in orion_recall (Phase 0/1 verification) plus
# a couple of no-entity control queries to confirm the boost stays a true
# no-op when there's nothing to boost on.
QUERIES = [
    ("tell me about Nvidia", "chat.general.v1"),
    ("what do we know about Atlas", "chat.general.v1"),
    ("anything about Juniper and Orion", "chat.general.v1"),
    ("Tesla gpu setup", "chat.general.v1"),
    ("how's it going today", "chat.general.v1"),  # no real entities -- control
    ("what's the weather like", "chat.general.v1"),  # no real entities -- control
]


async def _run_one(fragment: str, profile: str) -> dict:
    q = RecallQueryV1(fragment=fragment, profile=profile, node_id="eval-script")
    bundle, decision = await process_recall(q, corr_id=str(uuid4()), diagnostic=True)
    return {
        "selected_ids": [i.id for i in bundle.items],
        "backend_counts": decision.backend_counts,
        "top_item": {
            "source": bundle.items[0].source,
            "score": bundle.items[0].score,
            "snippet": (bundle.items[0].snippet or "")[:120],
        }
        if bundle.items
        else None,
    }


async def main() -> int:
    results = []
    for fragment, profile in QUERIES:
        settings_mod.settings.RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED = False
        before = await _run_one(fragment, profile)
        settings_mod.settings.RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED = True
        after = await _run_one(fragment, profile)

        changed = before["selected_ids"] != after["selected_ids"]
        results.append({"fragment": fragment, "changed": changed, "before": before, "after": after})

        print(f"\n=== {fragment!r} ===")
        print(f"  changed order/selection: {changed}")
        print(f"  before top: {before['top_item']}")
        print(f"  after  top: {after['top_item']}")

    changed_count = sum(1 for r in results if r["changed"])
    print(f"\n=== SUMMARY: {changed_count}/{len(results)} queries changed with the flag on ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
