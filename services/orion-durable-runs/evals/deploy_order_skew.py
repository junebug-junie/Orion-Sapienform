"""Periodic eval: the 2026-09-26 deploy-order incident, replayed. Disposable Postgres DSN only.

durable-runs 4.5 came up ~20 s before the 4.5 GPU pool; the old pool refused every hold request
(``invalid:... extra_forbidden``) and 11 resumed runs were failed terminally. This replays that
window with 11 runs (6 curiosity, 5 self-sense, the live mix) against a pool that answers like the
old one for a while, then like the new one, through the real durable-runs runtime, the real
in-process pool, the real client + codec and real Postgres checkpoints.

Checks:
  no_run_failed         -- zero run.failed during or after the skew window
  all_completed         -- every run completes once the pool is upgraded
  refusals_visible      -- every run has >=1 run.waiting_resource event with transient=true and the
                           pool's invalid: reason (never silent)
  no_retry_storm        -- acquire RPCs during the skew window stay within what the backoff allows
                           (per run: at most 1 + log2(window/base) + 1 asks), not one per tick
  one_hold_per_run      -- each run ends with exactly one pool hold (the same request id throughout)

Prints inspectable JSON; exits non-zero on any failed check. No model inference or production bus.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

SERVICE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SERVICE.parents[1]), str(SERVICE), str(SERVICE / "tests")]

from test_admission_runtime_postgres import request, runtime, with_database  # noqa: E402
from test_pool_refusal_postgres import scripted  # noqa: E402
from pool_fixture import InProcessPool  # noqa: E402

RUNS = 11
SKEW_WINDOW_SEC = 1.2
BASE_SEC = 0.1
TICK_SEC = 0.02


async def scenario(pool, saver, store):
    gpu = await InProcessPool().boot()
    rt = runtime(pool, saver, store, gpu=gpu, DURABLE_RUNS_POOL_RETRY_BASE_SEC=BASE_SEC,
                 DURABLE_RUNS_POOL_RETRY_MAX_SEC=0.4, DURABLE_RUNS_HOLD_STATUS_POLL_SEC=0.05)
    bus = scripted(rt)
    bus.old_pool = True
    ids = []
    for i in range(RUNS):
        workflow = "curiosity.investigate" if i < 6 else "self_sense_eval"
        req = request(f"skew-eval-{i:02d}", workflow=workflow)
        ids.append(req.run_id)
        await rt.submit(req)

    async def drive_all():
        for run_id in ids:
            row = await store.get_run(run_id)
            if row and not row.get("terminal"):
                await rt._drive(row)

    started = time.monotonic()
    while time.monotonic() - started < SKEW_WINDOW_SEC:
        await drive_all()
        await asyncio.sleep(TICK_SEC)
    skew_acquires = bus.acquires
    bus.old_pool = False
    for _ in range(100):
        await drive_all()
        if all([(await store.get_run(r))["terminal"] for r in ids]):
            break
        await asyncio.sleep(TICK_SEC)
    await rt.close()

    histories = {r: await store.history(r) for r in ids}
    terminals = {r: (await store.get_run(r))["terminal"] for r in ids}
    refusals = {r: [e["detail"] for e in h if e["event"] == "run.waiting_resource"
                    and e["detail"].get("transient")] for r, h in histories.items()}
    allowed = RUNS * (2 + math.ceil(math.log2(SKEW_WINDOW_SEC / BASE_SEC)))
    holds = {r: [l["request_id"] for l in gpu.leases(holder=f"durable-runs:{r}")] for r in ids}
    checks = {
        "no_run_failed": not any(e["event"] == "run.failed" for h in histories.values() for e in h),
        "all_completed": all(t == "completed" for t in terminals.values()),
        "refusals_visible": all(v and all(d["reason"].startswith("invalid:") for d in v) for v in refusals.values()),
        "no_retry_storm": skew_acquires <= allowed,
        "one_hold_per_run": all(v == [f"{r}:1"] for r, v in holds.items()),
    }
    return {"runs": RUNS, "skew_window_sec": SKEW_WINDOW_SEC, "acquires_during_skew": skew_acquires,
            "acquires_allowed": allowed, "refusals_per_run": {r: len(v) for r, v in refusals.items()},
            "terminals": terminals, "checks": checks}


async def main() -> int:
    if not os.environ.get("ORION_ADMISSION_TEST_DSN"):
        print("ORION_ADMISSION_TEST_DSN (a disposable database) is required", file=sys.stderr)
        return 2
    holder: dict = {}

    async def run(pool, saver, store):
        holder["report"] = await scenario(pool, saver, store)

    await asyncio.wait_for(with_database(run), 120)
    report = holder["report"]
    failed = [name for name, ok in report["checks"].items() if not ok]
    report["verdict"] = "FAIL" if failed else "PASS"
    report["failed_checks"] = failed
    print(json.dumps(report, indent=2, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
