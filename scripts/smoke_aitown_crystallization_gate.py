#!/usr/bin/env python3
"""Replay the ai-town formation gate over the LIVE governor queue. Read-only.

Config being set and unit tests passing are not evidence the gate does anything
(CLAUDE.md: runtime truth beats config truth). This pulls every real `proposed`
crystallization, reconstructs each one's window turns and their true
source_platform from chat_history_log.client_meta, feeds them through the same
build_crystallization_from_window() + resolve_formation_policy() the service
runs, and reports what the gate would actually have decided.

It writes nothing. Its output doubles as the pre-flight manifest for the
bulk-reject of the backlog (scripts/bulk_reject_aitown_proposals.py), which
takes its target list from the same query.

    python3 scripts/smoke_aitown_crystallization_gate.py [--dsn ...] [--json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.memory.consolidation_gate import ConsolidationGateResult  # noqa: E402
from orion.memory.crystallization.formation_policy import (  # noqa: E402
    DEFAULT_DISCARD_PLATFORMS,
    FormationPolicy,
    resolve_formation_policy,
)
from orion.memory.crystallization.intake_consolidation_window import (  # noqa: E402
    build_crystallization_from_window,
)

DEFAULT_DSN = os.environ.get(
    "ORION_SQL_DSN", "postgresql://postgres:postgres@localhost:55432/conjourney"
)

# One row per proposed crystallization, with its window's turns resolved to
# their real platform. LEFT JOIN on chat_history_log AND aitown_chat_history_log
# because a window can reference a correlation_id whose chat row was pruned --
# those turns must read as unknown/None (which forces the window to "mixed",
# i.e. still reviewed), never be silently dropped from the unanimity check.
#
# AI Town chat-history table split (docs/superpowers/specs/2026-08-19-aitown-
# table-split-phase2-recall-migration-design.md). The historical AI-Town
# rows were moved (not copied) out of chat_history_log into
# aitown_chat_history_log via a one-off backfill script, run live against
# this deployment's Postgres on 2026-08-19 -- verified: chat_history_log
# went from 1,747 to 170 rows, aitown_chat_history_log now holds 1,577. The
# corresponding orion-sql-writer WRITE-PATH change (routing new rows to one
# table instead of Phase 1's additive dual-write) is a separate PR, not
# merged as of this commit -- until it lands, new AI-Town turns can still
# land in chat_history_log too (Phase 1's dual-write, gated by
# SQL_WRITER_AITOWN_DUAL_WRITE_ENABLED, default off). Without this second
# join, every AI-Town turn whose row lives only in aitown_chat_history_log
# would resolve to NULL here, and every AI-Town-unanimous window would
# misclassify as "mixed" instead of "external", defeating the exact thing
# this script exists to verify. COALESCE prefers chat_history_log when a
# correlation_id matches both -- a real possibility until the write-path
# cutover merges, not just defensive padding.
QUERY = """
WITH prop AS (
    SELECT crystallization_id,
           provenance->>'memory_window_id' AS window_id,
           subject,
           created_at,
           kind
    FROM memory_crystallizations
    WHERE status = 'proposed'
),
window_turns AS (
    SELECT memory_window_id,
           (jsonb_array_elements(turn_correlation_ids))->>'correlation_id' AS correlation_id
    FROM memory_consolidation_windows
)
SELECT p.crystallization_id,
       p.window_id,
       p.subject,
       p.created_at,
       p.kind,
       w.correlation_id,
       COALESCE(h.prompt, a.prompt) AS prompt,
       COALESCE(h.response, a.response) AS response,
       COALESCE(h.client_meta, a.client_meta)->'external_room'->>'platform' AS platform
FROM prop p
LEFT JOIN window_turns w ON w.memory_window_id = p.window_id
LEFT JOIN chat_history_log h ON h.correlation_id = w.correlation_id
LEFT JOIN aitown_chat_history_log a ON a.correlation_id = w.correlation_id
ORDER BY p.created_at DESC, w.correlation_id
"""


def find_leaked(
    service_written: list[tuple[str, str | None]],
    *,
    platforms: frozenset[str] = DEFAULT_DISCARD_PLATFORMS,
) -> list[tuple[str, str | None]]:
    """Queued crystallizations the DEPLOYED service stamped with an allowlisted
    platform. Any such row means the live gate did not fire on it.

    Extracted as a function purely so its FAIL branch can be proved reachable in
    a test. The previous version of this check was inline and was provably always
    empty -- it compared the replay's own platform resolution against the replay's
    own policy decision, which are equivalent by construction. A check that cannot
    fail is worse than no check, because it reads as passing evidence.
    """
    return [row for row in service_written if str(row[1] or "") in platforms]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--json", action="store_true", help="emit machine-readable result")
    ap.add_argument("--show", type=int, default=15, help="how many queued subjects to print")
    args = ap.parse_args()

    import psycopg2

    conn = psycopg2.connect(args.dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(QUERY)
            rows = cur.fetchall()
            # Independent of the replay: what the DEPLOYED service actually
            # stamped on rows it left in the queue. The replay below recomputes
            # provenance from live chat rows, so it can only ever agree with
            # itself; this reads what the running worker wrote.
            cur.execute(
                "SELECT crystallization_id::text, provenance->>'source_platform' "
                "FROM memory_crystallizations "
                "WHERE status = 'proposed' AND provenance ? 'source_platform'"
            )
            db_written_platforms = cur.fetchall()
    finally:
        conn.close()

    # group turns per crystallization, preserving window identity
    grouped: dict[str, dict] = {}
    for cid, window_id, subject, created_at, kind, corr, prompt, response, platform in rows:
        entry = grouped.setdefault(
            cid,
            {
                "window_id": window_id,
                "subject": subject,
                "created_at": created_at,
                "kind": kind,
                "turns": [],
            },
        )
        if corr is None:
            continue
        entry["turns"].append(
            {
                "correlation_id": corr,
                "prompt": prompt or "",
                "response": response or "",
                "spark_meta": {},
                "source_platform": platform,
            }
        )

    auto: list[tuple[str, str]] = []
    queued: list[tuple[str, str]] = []
    reasons: Counter[str] = Counter()
    platforms: Counter[str] = Counter()

    for cid, entry in grouped.items():
        crys = build_crystallization_from_window(
            memory_window_id=entry["window_id"] or "unknown",
            turns=entry["turns"],
            # The live backlog is 621/621 dominant_shift=STANCE, which is what
            # makes it a GATED_KIND and therefore queue-bound today. Replaying
            # with STANCE is the honest reproduction of the current state, not
            # a convenient choice.
            gate=ConsolidationGateResult(
                action="propose", dominant_shift="STANCE", grammar_event_ids=[]
            ),
        )
        # Real stored kind, not the one dominant_shift="STANCE" reconstructs --
        # DISCARD now applies to every kind (the old EXTERNAL_PLATFORM_BYPASSABLE_KINDS
        # stance-only restriction was retired 2026-08-16), but the replay should
        # still reflect what was actually stored, not the STANCE reconstruction.
        if entry.get("kind"):
            crys.kind = entry["kind"]
        entry["resolved_platform"] = crys.provenance.get("source_platform")
        platforms[str(entry["resolved_platform"])] += 1
        policy, why = resolve_formation_policy(crys)
        subject = " ".join((entry["subject"] or "").split())[:90]
        if policy == FormationPolicy.DISCARD:
            auto.append((cid, subject))
        else:
            queued.append((cid, subject))
            reasons[";".join(why)] += 1

    total = len(grouped)
    if args.json:
        print(
            json.dumps(
                {
                    "total_proposed": total,
                    "would_discard": len(auto),
                    "would_stay_queued": len(queued),
                    "discard_ids": [c for c, _ in auto],
                    "queued_ids": [c for c, _ in queued],
                    "window_platforms": dict(platforms),
                    "queue_reasons": dict(reasons),
                    "service_written_platforms": [list(r) for r in db_written_platforms],
                },
                indent=2,
            )
        )
        return 0

    print(f"live proposed crystallizations: {total}")
    print(f"  would DISCARD (never become a memory):  {len(auto)}")
    print(f"  would STAY QUEUED (real review work):   {len(queued)}")
    print()
    print("resolved window platform (unanimous across all turns, else None):")
    for name, n in platforms.most_common():
        print(f"  {name:>12}: {n}")
    print()
    print("why the survivors stayed queued:")
    for name, n in reasons.most_common():
        print(f"  {name:>24}: {n}")
    print()
    print(f"survivors (first {args.show}):")
    for cid, subject in queued[: args.show]:
        print(f"  {cid[:8]}  {subject}")

    # What this script can and cannot prove -- stated plainly, because two
    # previous versions of this check were wrong in opposite directions.
    #
    # It REPLAYS live rows through the real policy functions. Both the platform
    # resolution and the policy decision therefore come from the same pure code
    # path, which means any check comparing one against the other is a
    # tautology. The first attempt (`if total and not auto: fail`) inverted once
    # the gate worked; the second (`resolved_platform is allowlisted but not in
    # auto`) was provably always empty, since with dominant_shift hardcoded to
    # STANCE and sensitivity hardcoded to "private" those two conditions are
    # literally equivalent. A replay cannot detect an inert gate. Only the
    # deployed service can.
    #
    # So the check is now about the SERVICE's own output, which is independent
    # evidence: a crystallization written by the live consolidation worker from
    # an allowlisted platform must never be sitting in `proposed`. That row was
    # produced by the deployed code, not by this script.
    leaked = find_leaked(db_written_platforms)
    if leaked:
        print(
            f"\nFAIL: the live service left {len(leaked)} allowlisted-platform "
            f"crystallization(s) in the review queue -- the deployed gate is not "
            f"firing: {[r[0] for r in leaked[:5]]}",
            file=sys.stderr,
        )
        return 1

    if not db_written_platforms:
        print(
            "\nINCONCLUSIVE: no queued crystallization carries provenance."
            "source_platform, so the deployed gate has not been exercised yet. "
            "The replay above only shows the policy functions are self-consistent. "
            "This is NOT evidence the live gate works.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
