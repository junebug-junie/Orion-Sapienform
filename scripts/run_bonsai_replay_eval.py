#!/usr/bin/env python3
"""Bonsai-vs-Q4 replay quality test: 30 real tasks, both models, writes stubbed, one verdict.

    ORION_BUS_URL=redis://100.92.216.81:6379/0 PYTHONPATH=. .venv/bin/python scripts/run_bonsai_replay_eval.py --dry-run
    ORION_BUS_URL=redis://100.92.216.81:6379/0 PYTHONPATH=. .venv/bin/python scripts/run_bonsai_replay_eval.py

(or `make eval-bonsai-replay ARGS=--dry-run`.) Results: ~/.orion/model-replay/<UTC timestamp>/
summary.json, report.md (verdict first), blind_sheet.md + blind_key.json + hand_grades.csv,
transcripts/, results.jsonl (resumable with --out <that dir>), holds.jsonl (every pool hold taken).

Per task it holds gpu1 (Q4, class memory_distill -> role agent) and gpu2 (Bonsai, class agent ->
role agent-gpu2, both slots) through the pool's operator verbs, checks each grant's role and
profile, runs both models side by side, and releases both holds -- also on error, Ctrl-C,
SIGTERM and SIGHUP. After a SIGKILL: --release-leftovers <out dir>.
Design + no-write proof: orion/evals/model_replay/sandbox.py, docs/superpowers/pr-reports/2026-10-09-bonsai-replay-eval-pr.md.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.evals.model_replay import runner  # noqa: E402
from orion.evals.model_replay.fixture import FIXTURE_PATH, load_tasks  # noqa: E402
from orion.evals.model_replay.pool_hold import (  # noqa: E402
    SEATS, BusControlTransport, HoldLedger, attach_factory, release_leftovers,
)

PRIMARY_CHECKOUT = Path("/mnt/scripts/Orion-Sapienform")
DEFAULT_POOL_HTTP = "http://100.92.216.81:8127"
DEFAULT_PROD_GRAPH = "redis://127.0.0.1:6380/0"
ACTOR = "bonsai-replay-eval"


def select_tasks(args: argparse.Namespace):
    tasks = load_tasks(FIXTURE_PATH)
    if args.only:
        kinds = set(args.only.split(","))
        tasks = [t for t in tasks if t.kind in kinds]
    if args.task_ids:
        ids = set(args.task_ids.split(","))
        tasks = [t for t in tasks if t.task_id in ids]
    if args.limit:
        tasks = tasks[: args.limit]
    return tasks


def preflight(pool_http: str, image: str) -> list[str]:
    """Read-only checks before taking any hold: the seats serve the expected models, the sandbox image exists."""
    import httpx

    problems = []
    try:  # the production prompt compiler and validators every task needs; fail now, not at task 1
        import orion.harness.runner  # noqa: F401
        import orion.schemas.world_pulse_read  # noqa: F401
        import orion.thought.stance_react  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        problems.append(f"cannot import production prompt/validator code ({exc}); run with the repo .venv")
    try:
        state = httpx.get(f"{pool_http}/v1/pool", timeout=10).json()
        roles = {r["role"]: r for r in state.get("roles", [])}
        for m, spec in SEATS.items():
            r = roles.get(spec.expect_role) or {}
            if r.get("profile_name") != spec.expect_profile:
                problems.append(f"{spec.expect_role} serves {r.get('profile_name')!r} ({r.get('status')}), "
                                f"want {spec.expect_profile} for {m}")
        if state.get("mode") != "enforce":
            problems.append(f"pool mode is {state.get('mode')!r}: operator holds are refused outside enforce")
        if state.get("actuation_paused"):
            problems.append("pool actuation is paused")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"pool state unreadable at {pool_http}: {exc}")
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
        problems.append(f"sandbox image {image} not present locally")
    return problems


async def live(args: argparse.Namespace, tasks) -> int:
    from orion.evals.model_replay.graph_scratch import COPIED_GRAPHS, ProdGraphSource, prod_falkordb_image
    from orion.evals.model_replay.sandbox import DockerReadOnly, FetchCache, HostHttpGet, ReadOnlyPsql

    bus_url = os.environ.get("ORION_BUS_URL")
    if not bus_url:
        print("ORION_BUS_URL is not set (use redis://100.92.216.81:6379/0)")
        return 2
    problems = preflight(args.pool_http, args.sandbox_image)
    if problems:
        print("PREFLIGHT FAILED, no hold taken:\n  - " + "\n  - ".join(problems))
        return 2
    out = args.out
    print(f"results -> {out}")
    source = ProdGraphSource.connect(args.prod_graph_url)
    dumps = {g: source.dump(g) for g in COPIED_GRAPHS}
    print("graph copies taken (DUMP): " + ", ".join(f"{g} {len(p)} bytes" for g, p in dumps.items()))
    run_tag = out.name
    try:
        http = HostHttpGet()
        cfg = runner.ReplayConfig(out_dir=out, tasks=tasks, repo=args.repo, grant_timeout_sec=args.grant_timeout_sec)
        async with BusControlTransport(bus_url) as transport:
            loop = asyncio.get_running_loop()
            factory = runner.LiveRigFactory(repo=args.repo, run_tag=run_tag, sandbox_image=args.sandbox_image,
                                            graph_image=prod_falkordb_image(), graph_dumps=dumps,
                                            sql=ReadOnlyPsql(), http=http, docker_ro=DockerReadOnly(),
                                            fetch_cache=FetchCache(http),
                                            attach=lambda m, g: attach_factory(transport.bus, m, g), loop=loop)
            runner.install_signal_unwind(loop, asyncio.current_task())
            summary = await runner.run_replay(cfg, transport, factory)
    finally:
        leftover = subprocess.run(["docker", "ps", "-aq", "--filter", "label=orion.model_replay=1",
                                   "--filter", f"name=orion-replay-{run_tag}"], capture_output=True, text=True).stdout.split()
        if leftover:
            subprocess.run(["docker", "rm", "-f", *leftover], capture_output=True)
    print("\n" + summary["decision"]["verdict"])
    print(f"report: {out / 'report.md'}")
    return 0


async def leftovers(args: argparse.Namespace) -> int:
    bus_url = os.environ.get("ORION_BUS_URL")
    if not bus_url:
        print("ORION_BUS_URL is not set")
        return 2
    async with BusControlTransport(bus_url) as transport:
        released = await release_leftovers(transport, HoldLedger(args.release_leftovers / "holds.jsonl"), ACTOR)
    print(f"released {len(released)} leftover hold(s): {released}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="print the plan; no pool, no containers, no network")
    ap.add_argument("--out", type=Path, default=Path.home() / ".orion" / "model-replay"
                    / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    ap.add_argument("--only", help="comma list of kinds: curiosity,self_sense,reading,stance_react")
    ap.add_argument("--task-ids", help="comma list of task ids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repo", type=Path, default=PRIMARY_CHECKOUT if PRIMARY_CHECKOUT.exists()
                    else Path(__file__).resolve().parents[1])
    ap.add_argument("--pool-http", default=DEFAULT_POOL_HTTP)
    ap.add_argument("--prod-graph-url", default=DEFAULT_PROD_GRAPH)
    ap.add_argument("--sandbox-image", default="orion-harness-governor-harness-governor:latest")
    ap.add_argument("--grant-timeout-sec", type=float, default=1800.0)
    ap.add_argument("--release-leftovers", type=Path, metavar="OUT_DIR",
                    help="release holds a killed run left behind (reads OUT_DIR/holds.jsonl)")
    ap.add_argument("--finalize", type=Path, metavar="OUT_DIR", help="re-score and rewrite the report from results.jsonl")
    args = ap.parse_args()
    if args.release_leftovers:
        return asyncio.run(leftovers(args))
    tasks = select_tasks(args)
    if args.finalize:
        cfg = runner.ReplayConfig(out_dir=args.finalize, tasks=tasks, repo=args.repo)
        print(runner.finalize(cfg)["decision"]["verdict"])
        return 0
    if args.dry_run:
        cfg = runner.ReplayConfig(out_dir=args.out, tasks=tasks, repo=args.repo, dry_run=True)
        p = runner.plan(cfg)
        print(json.dumps({k: v for k, v in p.items() if k != "tasks"}, indent=1))
        for t in p["tasks"]:
            print(f"  {t['task_id']:<40} {t['kind']:<13} timeout {t['timeout_sec']:>6.0f}s  expect {t['expect']}"
                  f"{'  fetch=fail' if t['fetch'] == 'fail' else ''}{'  write-claim check' if t['write_claim_check'] else ''}")
        print(f"\nDRY RUN: nothing held, started or sent. Would write results to {args.out}")
        return 0
    t0 = time.time()
    rc = asyncio.run(live(args, tasks))
    print(f"wall {(time.time() - t0) / 3600:.2f} h")
    return rc


if __name__ == "__main__":
    sys.exit(main())
