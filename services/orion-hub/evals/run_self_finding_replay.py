"""Which self findings WOULD reach Juniper now (read-only, live DB). Self-monitor E3.

    PYTHONPATH=services/orion-hub:. POSTGRES_URI=postgresql://... \
        .venv/bin/python -P services/orion-hub/evals/run_self_finding_replay.py [--json]

Reads the live bet sources through ``orion.self_monitor.report.collect`` (the same reducer the
durable-runs host publishes), stamps each record the way the host's first evaluation does
(``record.advance(None, ...)``: held status = candidate, ``status_since`` = now), then runs the
loop's pure ``evaluate`` on an empty cursor: the first evaluation after deploy, where every faculty
already ``broken`` is told once. Later transitions need the host's hourly history (a change must hold
3 evaluations); ``replay`` runs such a sequence of states, which the offline eval plants.
Prints each finding and the prompt Orion would compose from. Writes nothing, sends nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional

from orion.schemas.self_monitor import SelfMonitorStateV1
from orion.self_monitor.record import advance


def held(state: SelfMonitorStateV1, previous: Optional[SelfMonitorStateV1] = None) -> SelfMonitorStateV1:
    """Stamp held status the way the host does (first evaluation when ``previous`` is None)."""
    prev = {r.faculty: r for r in previous.faculties} if previous else {}
    return state.model_copy(update={"faculties": [
        advance(prev.get(r.faculty), r, now=state.computed_at) for r in state.faculties]})


def replay(raw_states: list[SelfMonitorStateV1]) -> list[list[dict]]:
    """Unstamped hourly states -> the host's held states -> the findings each evaluation adds."""
    from scripts.self_finding_outreach import OutreachState, evaluate

    cursor, previous, out = OutreachState(), None, []
    for raw in raw_states:
        previous = held(raw, previous)
        cursor, new = evaluate(cursor, previous, now=raw.computed_at)
        out.append([f.as_dict() for f in new])
    return out


def would_tell(current: SelfMonitorStateV1) -> dict:
    from scripts.self_finding_outreach import OutreachState, build_self_finding_prompt, evaluate

    _, new = evaluate(OutreachState(), current, now=current.computed_at)
    counts: dict[str, int] = {}
    for r in current.faculties:
        counts[r.status] = counts.get(r.status, 0) + 1
    return {
        "computed_at": current.computed_at.isoformat(),
        "status_counts": dict(sorted(counts.items())),
        "source_errors": list(current.source_errors),
        "findings": [f.as_dict() for f in new],
        "prompt": build_self_finding_prompt(new) if new else None,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=float, default=7.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    url = os.environ.get("POSTGRES_URI") or os.environ.get("DATABASE_URL")
    if not url:
        print("POSTGRES_URI or DATABASE_URL required", file=sys.stderr)
        return 2
    from orion.self_monitor.report import collect

    live = collect(url, now=datetime.now(timezone.utc), window_days=args.days)
    out = would_tell(held(live))
    if args.json:
        print(json.dumps(out, indent=2))
        return 0
    print(f"self-monitor state {out['computed_at']}  statuses {out['status_counts']}")
    for err in out["source_errors"]:
        print(f"SOURCE ERROR {err}")
    print(f"would tell {len(out['findings'])} finding(s):")
    for f in out["findings"]:
        print(f"  {f['finding_id']}  kind={f['kind']}  from={f['from_status']}")
        for fact in f["facts"]:
            print(f"    {fact}")
    if out["prompt"]:
        print("\n--- prompt Orion would compose from ---\n" + out["prompt"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
