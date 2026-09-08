#!/usr/bin/env python3
"""Print what Orion WOULD say to Claude, and what the Claude budget says.

Nothing here spends anything. No `claude` subprocess, no publish to
`orion:room:claude:request`, no write of any kind -- two read-only queries and
some arithmetic. Run it on the host (it needs `~/.claude/projects` and
FalkorDB, both localhost).

    python3 scripts/report_ask_claude_dry_run.py
    python3 scripts/report_ask_claude_dry_run.py --json >> /tmp/ask-claude-dry-run/log.jsonl

WHY A SCRIPT AND NOT A PANEL, FOR NOW. The deliverable of a dry run is an
ANSWER, not a dashboard -- `docs/superpowers/specs/2026-08-28-consequential-
action-space-and-power-budget-design.md` names "stage 1 stalls as another
telemetry channel nobody consumes" as the risk this shape avoids. The question
is whether the trigger discriminates: it prints every live prior with its
numbers, selected or not, so the selectivity is visible rather than asserted.
A panel is worth building once the trigger has earned arming.

READ THE POPULATION, NOT THE VERDICT. `MIN_TIMES_TESTED` and
`MAX_SETTLED_CONFIDENCE` in `orion/autonomy/ask_claude_trigger.py` are stated
defaults, not calibrated constants. The `stuck` column is what they select;
the `tested`/`conf` columns are the evidence for whether that selection is
sane. If every prior is stuck, or none ever is, the knobs are wrong and this
output is how that shows up.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.autonomy.ask_claude_trigger import (  # noqa: E402
    MAX_LIMIT_STALENESS_SEC,
    MAX_SETTLED_CONFIDENCE,
    MIN_TIMES_TESTED,
    decide,
)
from orion.curiosity.worldview import (  # noqa: E402
    LIVE_PRIORS_CYPHER,
    WorldviewReader,
    WorldviewUnavailable,
    Prior,
)
from orion.dev_economics.rate_limit_events import observe  # noqa: E402


def _row_to_prior(row: dict) -> Prior:
    """Build a `Prior` from a `LIVE_PRIORS_CYPHER` row.

    Field names come from the same Cypher the live loop uses, so this cannot
    drift from what Orion is actually shown -- deliberately not a second
    hand-written projection of the graph.
    """
    def _f(key: str):
        v = row.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    return Prior(
        prior_id=str(row.get("prior_id") or ""),
        claim=str(row.get("claim") or ""),
        confidence=_f("confidence"),
        status=str(row.get("status") or ""),
        times_tested=int(row.get("times_tested") or 0),
        formed_from=str(row.get("formed_from") or ""),
        last_tested_at=str(row.get("last_tested_at") or ""),
    )


def _read_priors(host: str, port: int, graph: str) -> tuple[list[Prior], str | None]:
    """Returns (priors, unavailable_reason). An unreachable graph is NOT an
    empty worldview -- same distinction `read_snapshot` draws, and for the same
    reason: a broken ACL after a FalkorDB restart must not read as "Orion has
    formed no priors"."""
    reader = WorldviewReader(host=host, port=port, graph_name=graph)
    try:
        rows = reader.query(LIVE_PRIORS_CYPHER)
    except WorldviewUnavailable as exc:
        return [], str(exc)[:200]
    except Exception as exc:  # noqa: BLE001 - a read-only report must not traceback
        return [], f"{type(exc).__name__}: {exc}"[:200]
    return [_row_to_prior(r) for r in rows], None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window-hours", type=float, default=5.0,
                    help="Trailing window for the limit read. 5 = session limit (default), 168 = weekly.")
    ap.add_argument("--projects-root", default=str(Path.home() / ".claude" / "projects"))
    ap.add_argument("--graph-host", default=os.environ.get("HUB_CURIOSITY_GRAPH_HOST", "127.0.0.1"))
    ap.add_argument("--graph-port", type=int, default=int(os.environ.get("HUB_CURIOSITY_GRAPH_PORT", "6380")))
    ap.add_argument("--graph-name", default=os.environ.get("HUB_CURIOSITY_GRAPH_OWN", "orion_worldview"))
    ap.add_argument("--json", action="store_true", help="One JSON object, for appending to a log.")
    args = ap.parse_args()

    limit = observe(window_hours=args.window_hours, root=args.projects_root)
    priors, unavailable = _read_priors(args.graph_host, args.graph_port, args.graph_name)
    decision = decide(priors=priors, limit=limit)

    if args.json:
        print(json.dumps({
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "window_hours": args.window_hours,
            "worldview_unavailable_reason": unavailable,
            "decision": {
                **{k: v for k, v in asdict(decision).items() if k != "assessments"},
                "assessments": [asdict(a) | {"stuck": a.stuck} for a in decision.assessments],
            },
        }, default=str))
        return 0

    print("=== Claude budget (orion/dev_economics/rate_limit_events.py) ===")
    print(f"  window            : {args.window_hours}h")
    print(f"  state             : {limit.state}")
    print(f"  observed          : {limit.observed}"
          + ("   <- UNOBSERVED is not 'clear'" if not limit.observed else ""))
    print(f"  times limit bound : {limit.event_count}")
    print(f"  resets at         : {limit.resets_at}")
    print(f"  staleness         : {limit.staleness_sec}s (refuse above {MAX_LIMIT_STALENESS_SEC}s)")
    print(f"  files scanned     : {limit.scanned_file_count}")

    print()
    print("=== Orion's live priors (orion_worldview, Orion's own graph) ===")
    if unavailable:
        # Loud, and distinct from "no priors" -- see _read_priors.
        print(f"  UNAVAILABLE: {unavailable}")
    elif not decision.assessments:
        print("  none live")
    else:
        print(f"  stuck = tested >= {MIN_TIMES_TESTED} AND confidence <= {MAX_SETTLED_CONFIDENCE} (or unrecorded)")
        print(f"  {'tested':>6} {'conf':>6} {'status':<10} {'stuck':<6} claim")
        for a in sorted(decision.assessments, key=lambda x: -x.times_tested):
            conf = f"{a.confidence:.2f}" if a.confidence is not None else "none"
            claim = a.claim if len(a.claim) <= 88 else a.claim[:85] + "..."
            print(f"  {a.times_tested:>6} {conf:>6} {a.status:<10} {'YES' if a.stuck else '-':<6} {claim}")

    print()
    print("=== Decision (DRY RUN -- nothing was sent, nothing was spent) ===")
    if decision.would_ask:
        print("  would ask Claude : YES")
        print(f"  about prior      : {decision.subject_prior_id}")
        print(f"  claim            : {decision.subject_claim}")
    else:
        print("  would ask Claude : no")
        print(f"  refused because  : {decision.refused}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
