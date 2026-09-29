#!/usr/bin/env python3
"""Replay capability:llm_inference's failure reading, per-window (old) vs rolling (new).

    docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc \\
        "select event_json from grammar_events where source_service='orion-llm-gateway'
         and created_at > now()-interval '72 hours' order by created_at" > gw.jsonl
    python scripts/analysis/replay_llm_inference_failure_window.py --events gw.jsonl

Both readings are then held the way the field digester holds them: a node's
``inference_failure_pressure`` channel keeps its last written value until the
next receipt carrying the hint (``mode="replace"``, not in NODE_DECAY_CHANNELS).
Each reading is sampled once per minute of the replay span.

old: one window's ``upstream_failed / (served + upstream_failed)``, no floor
     (the pre-2026-09-29 ``extract.inference_failure_pressure``, frozen here).
new: the live reducer (``reduce_llm_inference_trace_events``), which reads
     ``failure_window.failure_reading`` over the node's last 600 s of windows.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from orion.schemas.grammar import GrammarEventV1  # noqa: E402
from orion.substrate.llm_inference_loop.extract import _parse_kv  # noqa: E402
from orion.substrate.llm_inference_loop.pipeline import empty_llm_inference_projection  # noqa: E402
from orion.substrate.llm_inference_loop.reducer import reduce_llm_inference_trace_events  # noqa: E402

ROLE_NODE_WINDOW = "llm_inference_window_observed"


def old_value(summary: str) -> float | None:
    kv = _parse_kv(summary)
    served = int(kv.get("served", "0") or 0)
    failed = int(kv.get("upstream_failed", "0") or 0)
    attempted = served + failed
    return None if attempted <= 0 else min(1.0, failed / attempted)


def load(path: Path) -> list[GrammarEventV1]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(GrammarEventV1.model_validate(json.loads(line)))
    return out


def replay(events: list[GrammarEventV1]) -> dict:
    by_trace: dict[str, list[GrammarEventV1]] = defaultdict(list)
    for e in events:
        by_trace[e.trace_id or ""].append(e)
    traces = sorted(
        ((min(e.emitted_at for e in evs), tid, evs) for tid, evs in by_trace.items() if tid),
        key=lambda t: t[0],
    )
    projection = empty_llm_inference_projection(now=traces[0][0]) if traces else None
    old_writes: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    new_writes: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    new_scopes: dict[str, int] = defaultdict(int)
    failed_windows = []
    for ts, _tid, evs in traces:
        for e in evs:
            if e.atom and e.atom.semantic_role == ROLE_NODE_WINDOW:
                kv = _parse_kv(e.atom.summary or "")
                node = f"llm_node:{kv.get('node', '?')}"
                v = old_value(e.atom.summary or "")
                if v is not None:
                    old_writes[node].append((e.emitted_at, v))
                if int(kv.get("upstream_failed", "0") or 0) > 0:
                    failed_windows.append((e.emitted_at, node, int(kv["upstream_failed"]), kv.get("workers")))
        projection, receipt = reduce_llm_inference_trace_events(events=evs, projection=projection, now=ts)
        for d in receipt.state_deltas:
            hint = (d.after or {}).get("pressure_hints", {}).get("inference_failure_pressure")
            if hint is not None:
                new_writes[d.target_id].append((ts, float(hint)))
                if hint > 0:
                    new_scopes[str(d.after["failure_window"]["scope"])] += 1
    start = traces[0][0] if traces else None
    end = traces[-1][0] if traces else None
    return {
        "old": old_writes,
        "new": new_writes,
        "new_scopes": dict(new_scopes),
        "failed_windows": failed_windows,
        "start": start,
        "end": end,
    }


def held_minutes(writes: list[tuple[datetime, float]], start: datetime, end: datetime) -> list[float | None]:
    out: list[float | None] = []
    writes = sorted(writes)
    i, cur = 0, None
    t = start
    while t <= end:
        while i < len(writes) and writes[i][0] <= t:
            cur = writes[i][1]
            i += 1
        out.append(cur)
        t += timedelta(minutes=1)
    return out


def describe(name: str, minutes: list[float | None]) -> str:
    vals = [v for v in minutes if v is not None]
    nz = [v for v in vals if v > 0]
    runs, run = [], 0
    for v in minutes:
        if v is not None and v > 0:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    return (
        f"{name}: minutes={len(minutes)} nonzero_minutes={len(nz)} "
        f"max={max(vals, default=0):.3f} mean_when_nonzero={(sum(nz) / len(nz)) if nz else 0:.3f} "
        f"episodes={len(runs)} longest_hold_min={max(runs, default=0)} "
        f"minutes_at_1.0={sum(1 for v in vals if v >= 1.0)} "
        f"minutes_>=0.5={sum(1 for v in vals if v >= 0.5)}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--events", type=Path, required=True)
    args = ap.parse_args()
    r = replay(load(args.events))
    if r["start"] is None:
        print("no gateway traces")
        return 0
    print(f"span {r['start'].isoformat()} .. {r['end'].isoformat()}")
    fw = r["failed_windows"]
    print(
        f"windows with any upstream failure: {len(fw)}; "
        f"max failures in one window: {max((f[2] for f in fw), default=0)}"
    )
    for node in sorted(set(r["old"]) | set(r["new"])):
        print(f"## {node}")
        print(describe("old", held_minutes(r["old"].get(node, []), r["start"], r["end"])))
        print(describe("new", held_minutes(r["new"].get(node, []), r["start"], r["end"])))
    print(f"new nonzero writes by scope: {r['new_scopes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
