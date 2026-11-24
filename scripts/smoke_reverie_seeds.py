#!/usr/bin/env python3
"""Live, read-only smoke for reveries that ask (Patch 1): print one real seed of each kind and resolve its refs.

Runs orion-thought's own readers (services/orion-thought/app/reverie.py SEED_READERS) against the live
database and Chroma, with no spent refs, then re-reads every ref from its own table:

  vision_percept_expectation:<id>         -> the expectation row exists and is `missed`
  self_bet_bin:<faculty>:<lo>-<hi>        -> settled bets exist for that faculty in that bin
  episode_memory|stance|person_fact:<id>  -> the record is in the live serving snapshot (recall's gate)

Writes nothing. Needs POSTGRES_URI reachable from where it runs (on the host: the published port), and
ORION_REVERIE_FAR_MEMORY_CHROMA_URL (on the host: http://localhost:8500). Material text may name walkway
people; it is printed to the terminal only.

    POSTGRES_URI=postgresql://postgres:...@localhost:55432/conjourney \\
    ORION_REVERIE_FAR_MEMORY_CHROMA_URL=http://localhost:8500 \\
      python3 scripts/smoke_reverie_seeds.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "services" / "orion-thought"), str(REPO)]

from app import store  # noqa: E402
from app.reverie import SEED_READERS  # noqa: E402


def resolve(ref: str, live_docs: set[str]) -> str:
    kind, _, rest = ref.partition(":")
    if kind == "vision_percept_expectation":
        rows = store._rows("SELECT status FROM vision_percept_expectation WHERE expectation_id = :i", {"i": rest})
        return f"row status={rows[0]['status']}" if rows else "MISSING"
    if kind == "self_bet_bin":
        faculty, _, span = rest.rpartition(":")
        lo, _, hi = span.partition("-")
        rows = store._rows(
            "SELECT count(*) AS n FROM prediction_turn_self_bet WHERE faculty = :f AND stated_p >= :lo "
            "AND stated_p < :hi AND status IN ('happened','not_happened') AND settled_at > now() - interval '7 days'",
            {"f": faculty, "lo": float(lo), "hi": float(hi)})
        return f"{rows[0]['n']} settled bets" if rows and rows[0]["n"] else "MISSING"
    return "live record" if ref in live_docs else "NOT LIVE"


async def main() -> int:
    live = {r.doc_id for r in await store.load_live_lived_memory()}
    ok = True
    for kind, reader in SEED_READERS.items():
        try:
            seeds = await reader(set())
        except Exception as exc:  # noqa: BLE001
            print(f"[{kind}] READ FAILED: {type(exc).__name__}: {exc}")
            ok = False
            continue
        if not seeds:
            print(f"[{kind}] no material right now")
            continue
        s = seeds[0]
        print(f"[{kind}] candidates={len(seeds)} thread_key={s.thread_key} evidence={s.evidence}")
        print("  material: " + s.material.replace("\n", "\n            "))
        for ref in s.refs:
            verdict = resolve(ref, live)
            ok = ok and verdict not in ("MISSING", "NOT LIVE")
            print(f"  ref {ref} -> {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
