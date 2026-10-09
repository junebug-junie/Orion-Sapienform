#!/usr/bin/env python3
"""Re-extract the Bonsai-vs-Q4 replay fixture from live Postgres, read-only.

Every query runs inside BEGIN READ ONLY with default_transaction_read_only=on. Writes the committed
fixture orion/evals/model_replay/fixtures/tasks.v1.jsonl (task picks: orion/evals/model_replay/extract.py).

    PYTHONPATH=. .venv/bin/python scripts/extract_model_replay_fixture.py [--check]

--check: build in memory and diff against the committed file; exit 1 on drift.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.evals.model_replay import extract  # noqa: E402
from orion.evals.model_replay.fixture import FIXTURE_PATH, dump_tasks, load_tasks  # noqa: E402

PG_CONTAINER = "orion-athena-sql-db"


def ro_query(sql: str, container: str = PG_CONTAINER) -> list[dict]:
    """Rows as dicts, via psql in the DB container, inside a READ ONLY transaction."""
    # One row_to_json per line: JSON escapes newlines, so a line is exactly one row.
    script = f"BEGIN READ ONLY;\nSELECT row_to_json(t) FROM ({sql}) t;\nROLLBACK;\n"
    out = subprocess.run(
        ["docker", "exec", "-i", "-e", "PGOPTIONS=-c default_transaction_read_only=on", container,
         "psql", "-U", "postgres", "-d", "conjourney", "-X", "-At", "-v", "ON_ERROR_STOP=1", "-f", "-"],
        input=script, capture_output=True, text=True, check=True, timeout=120,
    ).stdout
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def _ids(values) -> str:
    return ",".join("'" + v.replace("'", "''") + "'" for v in values)


def build() -> list:
    cur_ids = [r[0] for r in extract.CURIOSITY_RUNS]
    ss_ids = [r[0] for r in extract.SELF_SENSE_SNAPSHOTS]
    rd_ids = [r[0] for r in extract.READING_RUNS]
    runs = ro_query(f"SELECT run_id, request, terminal FROM durable_admission_runs "
                    f"WHERE run_id IN ({_ids(cur_ids + ss_ids)})")
    by_id = {r["run_id"]: r for r in runs}
    reading = ro_query(
        f"SELECT DISTINCT ON (t.run_id) t.run_id, t.request_json, a.terminal FROM reading_durable_turn t "
        f"LEFT JOIN durable_admission_runs a USING (run_id) WHERE t.run_id IN ({_ids(rd_ids)}) "
        f"AND t.stage = 1 ORDER BY t.run_id, t.attempt")
    return extract.build_tasks(
        {k: by_id[k] for k in cur_ids if k in by_id},
        {k: by_id[k] for k in ss_ids if k in by_id},
        {r["run_id"]: r for r in reading},
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    tasks = build()
    print(f"{len(tasks)} tasks: {extract.task_counts(tasks)}")
    if args.check:
        committed = [t.model_dump(mode="json") for t in load_tasks()]
        fresh = [t.model_dump(mode="json") for t in tasks]
        if committed != fresh:
            print("DRIFT: committed fixture differs from a fresh extraction")
            return 1
        print("fixture matches")
        return 0
    dump_tasks(tasks)
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
