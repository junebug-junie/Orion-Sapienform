#!/usr/bin/env python3
"""One-time relabel of choice-built personality tendencies (#3444), with the CLAUDE.md section 14 protocol.

A tendency's label is set when choices fold into it and is never recomputed, so tendencies built before
#3379 keep labels like "a thread from my letter over a promise I made ..." or "... (+14 more)". This
rebuilds every non-retired ``origin='choice'`` node's label from its own stored choices with the CURRENT
label maker (``orion/personality/relabel.py``). Only ``label`` and ``label_similarity`` move.

    # 1. Dry run (default): read-only against Postgres; writes the snapshot and report, changes nothing.
    PYTHONPATH=. .venv/bin/python scripts/personality_relabel_choice_nodes.py [--dsn postgresql://...]

    # 2. Apply (production write: needs Juniper's approval) the REVIEWED dry run. Re-plans read-only and
    #    snapshots again, then asks orion-durable-runs to write exactly the reviewed changes under its
    #    reducer tick lock, in one transaction.
    PYTHONPATH=. .venv/bin/python scripts/personality_relabel_choice_nodes.py --apply \
        --from /tmp/personality-relabel/<dry-run> [--durable-runs-url http://127.0.0.1:8124]

Why an endpoint for the write: the reducer's tick lock is an in-process ``asyncio.Lock`` inside
orion-durable-runs, and a tick writes a node's whole row back when it folds a choice. A direct SQL write
from here could land between a tick's load and its write and be overwritten with the old label. So
``--apply`` calls ``POST /personality/relabel`` with ``expected`` = this run's [old_label, new_label] per
node, read from the reviewed dry run's before_after.csv (``--from``): the service writes only those exact
changes. A label that moved since the review, or that the current rules would now make differently, is
refused and reported.

Files under /tmp/personality-relabel/<run>/: snapshot.jsonl (every planned node's current label and
label_similarity: the rollback source), progress.log, before_after.csv, report.md, and apply_response.json
after --apply. Rollback: for each snapshot row, UPDATE personality_disposition SET label = old_label,
label_similarity = old_similarity WHERE disposition_id = ... (stop durable-runs' personality tick first).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.personality.graph import category_led  # noqa: E402
from orion.personality.relabel import (  # noqa: E402
    RELABEL_EVIDENCE_SQL,
    RELABEL_NODES_SQL,
    Relabel,
    plan_relabel,
    relabel_view,
)

OUT_ROOT = Path("/tmp/personality-relabel")
DEFAULT_DSN = os.environ.get("RECALL_PG_DSN", "postgresql://postgres:postgres@127.0.0.1:55432/conjourney")
DEFAULT_URL = os.environ.get("DURABLE_RUNS_URL", "http://127.0.0.1:8124")
# The issue's acceptance check (#3444): no non-retired choice label leads with one of these.
STALE_LEADS = ("a thread from my letter", "a promise i made", "a read,", "a daydream:")


class Progress:
    def __init__(self, path: Path, total: int):
        self.path, self.total, self.t0, self.errors = path, total, time.monotonic(), 0

    def line(self, title: str, done: int, anomalies: int = 0) -> None:
        pct = 100.0 * done / self.total if self.total else 100.0
        el = time.monotonic() - self.t0
        rate = done / el if el > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else 0.0
        msg = (f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {title} {pct:.0f}% eta={eta:.1f}s "
               f"rows={done}/{self.total} rate={rate:.1f}/s errors={self.errors} anomalies={anomalies}")
        with self.path.open("a") as fh:
            fh.write(msg + "\n")
        print(msg)


def stale_lead(label: str) -> bool:
    low = (label or "").strip().lower()
    return low.startswith(STALE_LEADS) or " (+" in low


async def read_plan(dsn: str) -> list[Relabel]:
    import asyncpg

    conn = await asyncpg.connect(dsn, timeout=10)
    try:
        async with conn.transaction(readonly=True):
            nodes = [dict(r) for r in await conn.fetch(RELABEL_NODES_SQL)]
            evidence = [dict(r) for r in await conn.fetch(RELABEL_EVIDENCE_SQL)]
    finally:
        await conn.close()
    return plan_relabel(nodes, evidence)


def write_files(run_dir: Path, plan: list[Relabel], progress: Progress) -> list[Relabel]:
    changes = [r for r in plan if r.changed]
    with (run_dir / "snapshot.jsonl").open("w") as fh:
        for r in plan:
            fh.write(json.dumps(relabel_view(r), sort_keys=True) + "\n")
    progress.line("snapshot_written", len(plan), sum(1 for r in plan if r.anomalies))
    with (run_dir / "before_after.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["disposition_id", "status", "evidence", "old_label", "new_label", "old_similarity",
                    "new_similarity", "base_evidence_id", "anomalies"])
        for r in changes:
            w.writerow([r.disposition_id, r.status, r.evidence, r.old_label, r.new_label, f"{r.old_similarity:.6f}",
                        f"{r.new_similarity:.6f}", r.base_evidence_id, ";".join(r.anomalies)])
    return changes


def report(run_dir: Path, plan: list[Relabel], changes: list[Relabel], *, applied: Optional[dict]) -> str:
    before_stale = sum(1 for r in plan if stale_lead(r.old_label))
    # After an apply, "after" is what was actually written; a refused node keeps its old label.
    written = set(applied.get("written_ids") or []) if applied is not None else None
    after_stale = [r for r in plan if stale_lead(
        r.new_label if r.changed and (written is None or r.disposition_id in written) else r.old_label)]
    category_after = [r for r in plan if category_led(r.new_label.rsplit(", ", 1)[0])]
    anomalies = [r for r in plan if r.anomalies]
    lines = [
        f"# Personality relabel ({'APPLIED' if applied else 'DRY RUN'}) {run_dir.name}",
        "",
        f"- Verdict: {'applied, see written/refused below' if applied else 'dry run only, nothing written'}",
        f"- Non-retired choice nodes: {len(plan)}",
        f"- Labels that change: {len(changes)}",
        f"- Labels with an old category lead or '(+k more)' tail: before {before_stale}, after {len(after_stale)}",
        f"- Labels still category-led after (only a category was ever stored): {len(category_after)}",
        f"- Nodes with anomalies: {len(anomalies)}",
        f"- Needs another pass: {'yes' if after_stale else 'no'}",
        "",
    ]
    if applied is not None:
        lines += [f"- Written: {applied.get('written')} ; refused (drifted since snapshot): "
                  f"{len(applied.get('refused_ids') or [])}", ""]
    lines += ["## Examples (first 10 changes)", ""]
    for r in changes[:10]:
        lines.append(f"- `{r.disposition_id}`: {r.old_label!r} -> {r.new_label!r}")
    if after_stale:
        lines += ["", "## Still stale after", ""] + [f"- `{r.disposition_id}`: {r.new_label!r}" for r in after_stale]
    if anomalies:
        lines += ["", "## Anomalies", ""] + [f"- `{r.disposition_id}`: {', '.join(r.anomalies)}" for r in anomalies]
    lines += ["", "## Files", ""] + [f"- {p}" for p in sorted(run_dir.iterdir())] + [f"- {run_dir / 'report.md'}"]
    text = "\n".join(lines) + "\n"
    (run_dir / "report.md").write_text(text)
    return text


def reviewed_pairs(run_dir: Path) -> dict[str, list[str]]:
    """disposition_id -> [old_label, new_label] from a dry run's before_after.csv."""
    with (run_dir / "before_after.csv").open(newline="") as fh:
        return {r["disposition_id"]: [r["old_label"], r["new_label"]] for r in csv.DictReader(fh)}


def post_apply(url: str, expected: dict[str, list[str]]) -> dict:
    import httpx

    resp = httpx.post(url.rstrip("/") + "/personality/relabel", json={"apply": True, "expected": expected},
                      timeout=120.0)
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--apply", action="store_true", help="write through orion-durable-runs (production write)")
    ap.add_argument("--from", dest="reviewed", type=Path,
                    help="the reviewed dry run directory; --apply writes only its before_after.csv pairs")
    ap.add_argument("--durable-runs-url", default=DEFAULT_URL)
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    args = ap.parse_args()
    reviewed: Optional[dict[str, list[str]]] = None
    if args.apply:
        if args.reviewed is None or not (args.reviewed / "before_after.csv").exists():
            print("--apply needs --from <reviewed dry run dir> (with before_after.csv)", file=sys.stderr)
            return 5
        reviewed = reviewed_pairs(args.reviewed)

    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ("-apply" if args.apply else "-dry")
    run_dir = args.out / run
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        plan = asyncio.run(read_plan(args.dsn))
    except Exception as exc:  # noqa: BLE001
        print(f"postgres read failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    progress = Progress(run_dir / "progress.log", len(plan))
    progress.line("planned", len(plan), sum(1 for r in plan if r.anomalies))
    if len(plan) > 100_000:
        print("more than 100k nodes: stop and ask Juniper (CLAUDE.md section 14)", file=sys.stderr)
        return 3
    changes = write_files(run_dir, plan, progress)
    applied = None
    if args.apply:
        if not reviewed:
            print("the reviewed dry run has nothing to change")
        else:
            try:
                applied = post_apply(args.durable_runs_url, reviewed)
            except Exception as exc:  # noqa: BLE001
                progress.errors += 1
                progress.line("apply_failed", 0)
                print(f"apply failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                report(run_dir, plan, changes, applied=None)
                return 4
            (run_dir / "apply_response.json").write_text(json.dumps(applied, indent=2, sort_keys=True))
            progress.line("applied", int(applied.get("written") or 0))
    print(report(run_dir, plan, changes, applied=applied))
    print(f"tail -f {run_dir / 'progress.log'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
