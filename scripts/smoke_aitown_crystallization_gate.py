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
# their real platform. LEFT JOIN on chat_history_log because a window can
# reference a correlation_id whose chat row was pruned -- those turns must read
# as unknown/None (which forces the window to "mixed", i.e. still reviewed),
# never be silently dropped from the unanimity check.
QUERY = """
WITH prop AS (
    SELECT crystallization_id,
           provenance->>'memory_window_id' AS window_id,
           subject,
           created_at
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
       w.correlation_id,
       h.prompt,
       h.response,
       h.client_meta->'external_room'->>'platform' AS platform
FROM prop p
LEFT JOIN window_turns w ON w.memory_window_id = p.window_id
LEFT JOIN chat_history_log h ON h.correlation_id = w.correlation_id
ORDER BY p.created_at DESC, w.correlation_id
"""


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
    finally:
        conn.close()

    # group turns per crystallization, preserving window identity
    grouped: dict[str, dict] = {}
    for cid, window_id, subject, created_at, corr, prompt, response, platform in rows:
        entry = grouped.setdefault(
            cid,
            {"window_id": window_id, "subject": subject, "created_at": created_at, "turns": []},
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
        platforms[str(crys.provenance.get("source_platform"))] += 1
        policy, why = resolve_formation_policy(crys)
        subject = " ".join((entry["subject"] or "").split())[:90]
        if policy == FormationPolicy.AUTO_ACTIVATE:
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
                    "would_auto_activate": len(auto),
                    "would_stay_queued": len(queued),
                    "auto_activate_ids": [c for c, _ in auto],
                    "queued_ids": [c for c, _ in queued],
                    "window_platforms": dict(platforms),
                    "queue_reasons": dict(reasons),
                },
                indent=2,
            )
        )
        return 0

    print(f"live proposed crystallizations: {total}")
    print(f"  would AUTO-ACTIVATE (leave the queue): {len(auto)}")
    print(f"  would STAY QUEUED (real review work):  {len(queued)}")
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

    if total and not auto:
        print("\nFAIL: gate matched nothing on live data -- it is inert.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
