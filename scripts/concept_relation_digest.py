#!/usr/bin/env python3
"""Threshold-tuning report + belief-revision digest for
`maybe_resolve_concept_relation()` (orion/memory/crystallization/concept_relation.py).

Context: that function classifies every new crystallization against similar existing
ones via a real LLM call (same/refines/contradicts/unrelated, with a confidence float
gated by CONCEPT_RELATION_CONFIDENCE_FLOOR, default 0.6). Until this patch, only the
*decisive* outcome reached a log line -- every "unrelated" decision and every sub-floor
"contradicts"/"refines" decision vanished silently. Those decisions are now written to
`memory_concept_relation_decisions` (see orion/core/storage/sql/memory_crystallizations.sql)
by `insert_concept_relation_decision()` regardless of outcome. This script is the reader
that makes that table worth having -- a write-only log is the same "theater" pattern this
repo has already killed twice (see docs/superpowers/specs/2026-07-13-recall-followups-
loop-retirement-saturation-gate-spec.md, section 1).

Two things happen per run, over every row with digested = false:

1. Threshold-tuning report (printed, and available via --json): call volume, the
   same/refines/contradicts/unrelated distribution, and specifically how many
   contradicts/refines decisions had floor_cleared = false -- i.e. the LLM judged a real
   relation but confidence landed under the floor. Those are near-misses invisible
   until now; this report exists so a human can decide whether CONCEPT_RELATION_
   CONFIDENCE_FLOOR needs retuning. Nothing here is auto-applied.

2. Belief-revision digest: for every row where relation is contradicts/refines/same AND
   floor_cleared = true (i.e. the ones concept_relation.py actually acted on -- attached
   a link or reinforced a match), the report counts it as actionable. Every such row is
   still marked digested, and the row itself remains the durable, queryable record in
   `memory_concept_relation_decisions` -- Orion's real trace of revising its beliefs
   over time. AGENTS.md's "error correction... coherent action over time" prerequisite
   is served by that table being real and queryable, not by mirroring it into a second
   store.

   As of 2026-08-20 this digest no longer ALSO writes a `reflection`-kind crystallization
   per actionable row (removed live 2026-08-20 after the operator reported "seeing dups
   in the crystallization [browser] pretty frequently": every relation comparison Orion
   ran against itself was landing in the same list as real remembered content, and had
   grown to 57% (356/620) of all active crystallizations -- pure duplicated bookkeeping,
   not new memory. `memory_concept_relation_decisions` already IS that bookkeeping's
   real, queryable home; auto-writing it a second time as `kind="reflection"` crystallizations
   (bypassing manual review entirely -- `requires_manual_review=False`, `approval_mode=
   "auto_policy"`) added no information, just noise in front of a human. The write path
   (`_build_reflection_crystallization` et al.) was removed rather than left dead; git
   history has the exact shape if it's ever wanted back. `reflections_created` stays on
   `DigestReport` and is now always `[]` -- kept for `--json` output-shape stability,
   not because anything is produced anymore.

No LLM call happens anywhere in this script -- it is pure aggregation over decisions
`concept_relation.py` already classified. It never auto-supersedes or mutates any
existing crystallization's status, and as of 2026-08-20 it creates nothing at all --
it only reports on and marks-digested rows `concept_relation.py` already wrote.

Not a service loop -- a standalone script you run on demand or via cron, same category
as scripts/check_activation_saturation.py.

Usage:
    POSTGRES_URI=postgresql://user:pass@host:port/db python scripts/concept_relation_digest.py
    python scripts/concept_relation_digest.py --postgres-uri postgresql://...
    python scripts/concept_relation_digest.py --json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# Running as `python scripts/concept_relation_digest.py` puts scripts/ on sys.path[0],
# which shadows stdlib modules (same issue documented in
# scripts/check_inner_state_registry.py / scripts/check_activation_saturation.py).
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_RELATIONS = ("same", "refines", "contradicts", "unrelated")

_SELECT_PENDING_SQL = """
    SELECT decision_id, candidate_crystallization_id, target_crystallization_id,
           relation, confidence, floor_cleared, decided_at
    FROM memory_concept_relation_decisions
    WHERE digested = false
    ORDER BY decided_at
"""

_MARK_DIGESTED_SQL = """
    UPDATE memory_concept_relation_decisions
    SET digested = true
    WHERE decision_id = ANY($1::uuid[])
"""


@dataclass
class DigestReport:
    decisions_seen: int
    relation_counts: dict[str, int]
    near_miss_counts: dict[str, int]  # contradicts/refines with floor_cleared=false
    reflections_created: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions_seen": self.decisions_seen,
            "relation_counts": self.relation_counts,
            "near_miss_counts": self.near_miss_counts,
            "reflections_created": self.reflections_created,
        }


async def _run_digest(postgres_uri: str) -> DigestReport:
    import asyncpg

    conn = await asyncpg.connect(postgres_uri)
    try:
        async with conn.transaction():
            rows = [dict(r) for r in await conn.fetch(_SELECT_PENDING_SQL)]

            relation_counts = {r: 0 for r in _RELATIONS}
            near_miss_counts = {"contradicts": 0, "refines": 0}
            # Always empty now -- kept on DigestReport/--json for output-shape stability.
            # See the module docstring, "2026-08-20": this digest no longer writes
            # reflection-kind crystallizations at all. memory_concept_relation_decisions
            # (this table, already durable and queryable) is the real record; mirroring
            # it into memory_crystallizations too was pure duplicated bookkeeping that had
            # grown to 57% of all active crystallizations.
            reflections_created: list[str] = []

            for row in rows:
                relation = row["relation"]
                relation_counts[relation] = relation_counts.get(relation, 0) + 1

                if relation in near_miss_counts and not row["floor_cleared"]:
                    near_miss_counts[relation] += 1

            if rows:
                decision_ids = [r["decision_id"] for r in rows]
                await conn.execute(_MARK_DIGESTED_SQL, decision_ids)

            return DigestReport(
                decisions_seen=len(rows),
                relation_counts=relation_counts,
                near_miss_counts=near_miss_counts,
                reflections_created=reflections_created,
            )
    finally:
        await conn.close()


def _print_report(report: DigestReport) -> None:
    print(
        f"concept_relation_digest: {report.decisions_seen} decision(s) since last run "
        f"(digested = false -> true)"
    )
    if report.decisions_seen == 0:
        print("  nothing new to report.")
        return

    print("  relation distribution:")
    for relation in _RELATIONS:
        print(f"    {relation}: {report.relation_counts.get(relation, 0)}")

    print("  near-misses (relation judged, confidence below CONCEPT_RELATION_CONFIDENCE_FLOOR):")
    for relation, count in report.near_miss_counts.items():
        print(f"    {relation}: {count}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--postgres-uri",
        default=os.getenv("POSTGRES_URI", ""),
        help="Postgres DSN. Defaults to $POSTGRES_URI (e.g. services/orion-hub/.env).",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of prose.")
    args = parser.parse_args(argv)

    if not args.postgres_uri.strip():
        print(
            "concept_relation_digest: no --postgres-uri given and $POSTGRES_URI is unset. "
            "Check services/orion-hub/.env for POSTGRES_URI.",
            file=sys.stderr,
        )
        return 2

    try:
        report = asyncio.run(_run_digest(args.postgres_uri))
    except Exception as exc:
        print(f"concept_relation_digest: run failed -- {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict()))
    else:
        _print_report(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
