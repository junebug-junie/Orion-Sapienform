"""Replay one real day of topic-foundry ticks with and without the repeat-work guards.

Read-only. Reads ``topic_foundry_runs`` / ``topic_foundry_segments`` and each
run's ``documents.jsonl`` artifact; writes nothing.

"Before" is what really happened that day. "After" replays the same train
requests under the two guards added 2026-10-30:

1. A train request is answered by an earlier same-spec_hash run that
   completed within ``--window-secs`` (no new run).
2. Enrichment makes an LLM call only for segment text not seen in an earlier
   run (text-hash cache). The cache is seeded with every run that completed
   before the day, i.e. the steady state once the cache has filled.

Usage (from the repo root, on the host that runs Postgres):

    python3 services/orion-topic-foundry/evals/replay_repeat_work.py --day 2026-10-29

Exit code 1 if "after" does not cut both train runs and LLM calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

PSQL = ["docker", "exec", "orion-athena-sql-db", "psql", "-U", "postgres", "-d", "conjourney", "-AtF", "\t", "-c"]
RUNS_ROOT = Path("/mnt/telemetry/models/topic-foundry/runs")


def q(sql: str) -> list[list[str]]:
    out = subprocess.run(PSQL + [sql], check=True, capture_output=True, text=True).stdout
    return [line.split("\t") for line in out.splitlines() if line]


def ts(raw: str) -> datetime | None:
    return datetime.fromisoformat(raw) if raw else None


def text_hashes(run_id: str) -> list[str]:
    path = RUNS_ROOT / run_id / "documents.jsonl"
    if not path.exists():
        return []
    hashes = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                text = str(json.loads(line).get("text", ""))
            except json.JSONDecodeError:
                continue
            if text:
                hashes.append(hashlib.sha256(text.encode()).hexdigest())
    return hashes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True, help="UTC day, YYYY-MM-DD")
    ap.add_argument("--window-secs", type=int, default=86400)
    args = ap.parse_args()
    day = datetime.fromisoformat(args.day)
    nxt = (day + timedelta(days=1)).date().isoformat()
    window = timedelta(seconds=args.window_secs)

    runs = q(
        "select m.name, r.run_id, r.spec_hash, r.status, r.created_at, coalesce(r.completed_at::text,'') "
        "from topic_foundry_runs r join topic_foundry_models m using(model_id) "
        f"where r.created_at < '{nxt}' and r.created_at >= timestamptz '{args.day}' - interval '3 days' "
        "order by r.created_at"
    )
    by_model: dict[str, list[dict]] = defaultdict(list)
    for name, run_id, spec_hash, status, created, completed in runs:
        by_model[name].append(
            {"run_id": run_id, "spec_hash": spec_hash, "status": status, "created": ts(created), "completed": ts(completed)}
        )

    enriched_before = dict(
        (row[0], int(row[1]))
        for row in q(
            "select m.name, count(*) from topic_foundry_segments s join topic_foundry_runs r using(run_id) "
            "join topic_foundry_models m on m.model_id = r.model_id "
            f"where s.enriched_at >= '{args.day}' and s.enriched_at < '{nxt}' group by 1"
        )
    )

    totals = defaultdict(int)
    print(f"Replay of {args.day}, dedup window {args.window_secs}s\n")
    print(f"{'model':40} {'train before':>12} {'train after':>11} {'LLM before':>10} {'LLM after':>9}")
    for name, model_runs in sorted(by_model.items()):
        prior = [r for r in model_runs if r["created"] < day.replace(tzinfo=r["created"].tzinfo)]
        today = [r for r in model_runs if r not in prior]
        if not today:
            continue
        # Steady-state cache: everything completed before the day.
        cache: set[str] = set()
        for r in prior:
            if r["status"] == "complete":
                cache.update(text_hashes(r["run_id"]))
        kept: list[dict] = [r for r in prior]
        trains_after = 0
        llm_after = 0
        for r in today:
            tick = r["created"]
            dedup = any(
                k["spec_hash"] == r["spec_hash"]
                and k["status"] == "complete"
                and k["completed"] is not None
                and k["completed"] <= tick
                and tick - k["completed"] <= window
                for k in kept
            )
            if dedup:
                continue
            trains_after += 1
            kept.append(r)
            if r["status"] == "complete":
                fresh = set(text_hashes(r["run_id"])) - cache
                llm_after += len(fresh)
                cache.update(fresh)
        before_llm = enriched_before.get(name, 0)
        print(f"{name:40} {len(today):>12} {trains_after:>11} {before_llm:>10} {llm_after:>9}")
        totals["train_before"] += len(today)
        totals["train_after"] += trains_after
        totals["llm_before"] += before_llm
        totals["llm_after"] += llm_after
    print(
        f"{'TOTAL':40} {totals['train_before']:>12} {totals['train_after']:>11} "
        f"{totals['llm_before']:>10} {totals['llm_after']:>9}"
    )
    print(
        "\nLLM after = distinct new segment texts (steady state). Failed runs never dedup;"
        " they are counted as trains in both columns."
    )
    ok = totals["train_after"] < totals["train_before"] and totals["llm_after"] < totals["llm_before"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
