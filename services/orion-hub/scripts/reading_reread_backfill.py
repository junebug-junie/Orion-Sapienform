#!/usr/bin/env python3
"""Re-read finished reads whose page text was never retained (Juniper 2026-10-14).

Reads before text retention (~2026-10-11), or whose fetch came back a 404, have no usable
retained text, so Stage 2 never saw a claim block and could not link anything. This puts
each one back through the REAL reading path, --concurrency at a time: ``operator.reread_read`` sets
the same seed row back to Stage 1 pending, and Hub's own workers do the rest (Stage 1
reading turn on the agent lane via durable runs, text retained, then Stage 2 with
reading_meaning_target_v1 active). No shadow queue, no direct model call.

CLAUDE.md section 14 protocol. Everything lands in --out (default /tmp/reading-reread-backfill):

    # from services/orion-hub, PYTHONPATH=.:../..
    python3 scripts/reading_reread_backfill.py snapshot          # read-only: targets, URL probe, snapshot
    python3 scripts/reading_reread_backfill.py run --yes [--concurrency 3]  # after Hub is deployed
    tail -f /tmp/reading-reread-backfill/progress.log
    python3 scripts/reading_reread_backfill.py report            # report.md + before_after.csv
    python3 scripts/reading_reread_backfill.py rollback          # dry run; --yes to withdraw links,
                                                                 # --restore-seeds to put rows back

Targets: reads whose Stage 2 finished in (--until - --days, --until] with no usable retained
text (fetch_text.load_retained_texts returns nothing). Each URL is probed with one plain GET
first; 404/410 or an unresolvable host is recorded ``gone`` and never re-read.

Pacing: up to --concurrency reads in flight (default 1). A new read starts only while the
reading queue has no live work and the GPU pool has nothing but this run's own reads (same
check as the resonance judge run, minus our own leases). --concurrency N > 1 is refused
unless Hub runs HUB_WORLD_PULSE_READ_MAX_IN_FLIGHT >= N (older Hub reads one at a time).
A read not finished within --per-read-timeout-min stops new starts (anomaly); reads already
in flight are still waited for. Resumable: every requeue is written to results.jsonl as
``in_flight`` before it is waited on, so a stopped run resumes by waiting on those reads,
never requeueing them; a requeued row with no record (stopped pre-concurrency run) is
recognized from the live row and progress.log. Stop with Ctrl-C / kill <pid>; restart:

    python3 scripts/reading_reread_backfill.py run --yes --concurrency 3

What a re-read writes (all snapshotted first): the seed row (status, handoff_json,
stage2_result_json, trace ids), new reading_durable_turn rows (new run ids), new
reading_document_snapshot rows (content-addressed, insert-only), new journal_entries rows
(new trace ids), substrate concept nodes re-materialized by Stage 1 (merged by identity),
and substrate_graph_journal proposals/decisions for the claims Stage 2 makes. Rollback
appends ``rejected`` decisions for every link decided for these seeds since the run
started (both reading policies), and with --restore-seeds writes the snapshotted row
values back. It does not delete journal rows, snapshots, or concept nodes (append-only
stores); those are listed in the report.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HUB_ROOT.parents[1]
for p in (REPO_ROOT, HUB_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from orion.world_pulse_read.assertions import MEANING_POLICY, POLICY  # noqa: E402

DEFAULT_OUT = "/tmp/reading-reread-backfill"
HUB_CONTAINER = "orion-athena-hub"
# Code that must be live in Hub before a re-read is safe (checked in the container).
DEPLOY_MARKERS = (
    ("/app/orion/world_pulse_read/assertions.py", MEANING_POLICY),
    # Without the stale-sweep exclusion a re-queued digest item is skipped on the next tick.
    ("/app/orion/world_pulse_read/queue.py", "operator reread"),
)
GONE_STATUSES = frozenset({404, 410})
SEED_COLUMNS = ("status", "attempts", "last_error", "claimed_at", "completed_at", "trace_id",
                "handoff_json", "handoff_at", "stage2_status", "stage2_attempts", "stage2_error",
                "stage2_claimed_at", "stage2_completed_at", "stage2_trace_id", "stage2_result_json")

TARGETS_SQL = """
SELECT seed_id, kind, url, title, stage2_completed_at, handoff_json FROM world_pulse_read_seed
WHERE stage2_status = 'done' AND handoff_json IS NOT NULL AND duplicate_of IS NULL
  AND stage2_completed_at > $2::timestamptz - make_interval(days => $1)
  AND stage2_completed_at <= $2::timestamptz
ORDER BY stage2_completed_at
"""
SEED_ROW_SQL = "SELECT * FROM world_pulse_read_seed WHERE seed_id = $1"
DURABLE_SQL = "SELECT * FROM reading_durable_turn WHERE seed_id = $1 ORDER BY stage, attempt"
JOURNAL_FOR_SEED_SQL = """
SELECT event_id, event_kind, proposal_id, decision_id, target_id, revision, payload, recorded_at
FROM substrate_graph_journal
WHERE (event_kind = 'decision' AND payload->'evidence_refs' ? ('reading_seed:' || $1))
   OR (event_kind = 'proposal' AND proposal_id IN (
        SELECT proposal_id FROM substrate_graph_journal
        WHERE event_kind = 'decision' AND payload->'evidence_refs' ? ('reading_seed:' || $1)))
ORDER BY recorded_at, event_id
"""
JOURNAL_ENTRIES_SQL = "SELECT entry_id, source_ref, created_at FROM journal_entries WHERE source_ref = ANY($1::text[])"
NEW_RUNS_SQL = "SELECT stage, attempt, run_id, created_at, consumed_at FROM reading_durable_turn WHERE seed_id = $1 AND created_at >= $2 ORDER BY stage, attempt"
# Anything else in flight on the reading queue: a re-read waits for it, so the backfill
# never takes a Stage 1/2 turn ahead of a live read (re-queued rows keep their old
# created_at, which CLAIM_SQL would otherwise put first). Stale digest items the sweep is
# about to skip are not live work.
QUEUE_BUSY_SQL = """
SELECT count(*) FROM world_pulse_read_seed
WHERE seed_id <> ALL($1::text[]) AND duplicate_of IS NULL
  AND (status IN ('pending', 'claimed') OR (status = 'done' AND stage2_status IN ('pending', 'claimed')))
  AND NOT (kind = 'digest_item' AND status = 'pending' AND created_at < now() - ($2 * interval '1 second'))
"""
DECISIONS_SINCE_SQL = """
SELECT payload->>'policy' AS policy, payload->>'resulting_state' AS state, target_id, decision_id
FROM substrate_graph_journal
WHERE event_kind = 'decision' AND recorded_at >= $2 AND payload->'evidence_refs' ? ('reading_seed:' || $1)
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _jsonable(v: Any) -> Any:
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    return v


def _row_dict(row: Any) -> dict[str, Any]:
    out = {}
    for k, v in dict(row).items():
        if k.endswith("_json") or k == "payload":
            v = json.loads(v) if isinstance(v, str) else v
        out[k] = _jsonable(v)
    return out


def resolve_dsn(explicit: str | None) -> str:
    if explicit:
        return explicit
    for key in ("RECALL_PG_DSN", "POSTGRES_URI", "DATABASE_URL"):
        value = str(os.getenv(key, "")).strip()
        if value:
            return value
    return "postgresql://postgres:postgres@127.0.0.1:55432/conjourney"


class Progress:
    """progress.log: one line per event with title, percent, ETA, processed/total, rate,
    errors, anomalies (CLAUDE.md section 14)."""

    def __init__(self, out: Path, total: int, done: int = 0) -> None:
        self.path, self.total, self.done = out / "progress.log", total, done
        self.started, self.base = time.monotonic(), done
        self.errors = self.anomalies = 0

    def line(self, title: str, **extra: Any) -> str:
        n = self.done - self.base
        elapsed = max(1e-6, time.monotonic() - self.started)
        rate = n / elapsed * 3600
        left = self.total - self.done
        eta = f"{left / (n / elapsed) / 60:.0f}m" if n else "unknown"
        pct = 100.0 * self.done / max(1, self.total)
        text = (f"{_now().isoformat(timespec='seconds')} {title} | {pct:.0f}% | eta {eta} | "
                f"{self.done}/{self.total} | {rate:.1f}/h | errors {self.errors} | anomalies {self.anomalies}")
        if extra:
            text += " | " + " ".join(f"{k}={v}" for k, v in extra.items())
        with self.path.open("a") as fh:
            fh.write(text + "\n")
        print(text, flush=True)
        return text


async def probe(client: Any, url: str) -> dict[str, Any]:
    """One plain GET. gone = 404/410 or a host that does not resolve; anything else (403 bot
    walls, 5xx, timeouts) is inconclusive and left to the real reader, which records it."""
    host = urlsplit(url).hostname or ""
    try:
        await asyncio.to_thread(socket.getaddrinfo, host, None)
    except OSError as exc:
        return {"probe": "gone", "http_status": None, "detail": f"dns:{type(exc).__name__}"}
    try:
        resp = await client.get(url, follow_redirects=True, timeout=20.0,
                                headers={"User-Agent": "Mozilla/5.0 (orion reading backfill probe)"})
    except Exception as exc:  # noqa: BLE001
        return {"probe": "inconclusive", "http_status": None, "detail": type(exc).__name__}
    status = resp.status_code
    verdict = "gone" if status in GONE_STATUSES else "fetchable" if status < 400 else "inconclusive"
    return {"probe": verdict, "http_status": status, "final_url": str(resp.url)}


def _subject_nodes(handoff: Any) -> list[dict[str, Any]]:
    """Read-only Falkor snapshot of the concept nodes this read wrote (what Stage 1 re-merges)."""
    try:
        from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig
        from orion.world_pulse_read.assertions import resolve_reading_subjects

        store = FalkorSubstrateStore(FalkorSubstrateStoreConfig(
            uri=os.getenv("FALKORDB_URI", "redis://127.0.0.1:6380"), ensure_indexes=False))
        out = []
        for e in resolve_reading_subjects(store, handoff):
            node = store.get_node_by_id(e.node_id)
            out.append(node.model_dump(mode="json") if node is not None else {"node_id": e.node_id, "missing": True})
        return out
    except Exception as exc:  # noqa: BLE001
        return [{"error": f"{type(exc).__name__}: {exc}"[:300]}]


async def cmd_snapshot(conn: Any, args: argparse.Namespace, out: Path) -> int:
    import httpx

    from orion.schemas.world_pulse_read import WorldPulseReadHandoffV1
    from orion.world_pulse_read.fetch_text import load_retained_texts

    until = datetime.fromisoformat(args.until) if args.until else _now()
    until = until if until.tzinfo else until.replace(tzinfo=timezone.utc)
    targets, snaps = [], []
    async with conn.transaction(readonly=True):
        rows = await conn.fetch(TARGETS_SQL, args.days, until)
        prog = Progress(out, len(rows))
        prog.line("snapshot_start", window_days=args.days, until=until.isoformat(), reads=len(rows))
        async with httpx.AsyncClient() as client:
            for row in rows:
                handoff = WorldPulseReadHandoffV1.model_validate(_row_dict(row)["handoff_json"])
                texts = await load_retained_texts(conn, handoff.read_evidence)
                prog.done += 1
                if texts:
                    continue
                full = _row_dict(await conn.fetchrow(SEED_ROW_SQL, row["seed_id"]))
                traces = [t for t in (full.get("trace_id"), full.get("stage2_trace_id")) if t]
                refs = [f"world_pulse_read:{full.get('trace_id')}", f"world_pulse_read_stage2:{full.get('stage2_trace_id')}"]
                result = full.get("stage2_result_json") or {}
                claims = result.get("relationship_claims") or []
                p = await probe(client, row["url"])
                target = {"seed_id": row["seed_id"], "kind": row["kind"], "url": row["url"],
                          "title": row["title"], "stage2_completed_at": _jsonable(row["stage2_completed_at"]),
                          "why_no_text": ("no_retained_text" if not any(e.content_sha256 for e in handoff.read_evidence)
                                          else "retained_text_is_an_http_error_or_unverifiable"),
                          "before_concepts": len(handoff.concept_candidates) if hasattr(handoff, "concept_candidates") else None,
                          "before_claims": len(claims),
                          "before_accepted": sum((c.get("receipt") or {}).get("outcome") == "accepted_provisional" for c in claims),
                          **p}
                targets.append(target)
                snaps.append({
                    "seed_id": row["seed_id"], "seed_row": full,
                    "durable_turns": [_row_dict(r) for r in await conn.fetch(DURABLE_SQL, row["seed_id"])],
                    "graph_journal": [_row_dict(r) for r in await conn.fetch(JOURNAL_FOR_SEED_SQL, row["seed_id"])],
                    "journal_entries": [_row_dict(r) for r in await conn.fetch(JOURNAL_ENTRIES_SQL, refs)],
                    "retained_sha256": sorted({e.content_sha256 for e in handoff.read_evidence if e.content_sha256}),
                    "subject_nodes": await asyncio.to_thread(_subject_nodes, handoff),
                    "traces": traces,
                })
                prog.line(f"target {row['seed_id'][:60]}", probe=p["probe"], http=p.get("http_status"))
    (out / "targets.json").write_text(json.dumps(targets, indent=1, default=str))
    with (out / "snapshot.jsonl").open("w") as fh:
        for s in snaps:
            fh.write(json.dumps(s, default=str) + "\n")
    (out / "seeds.txt").write_text("".join(t["seed_id"] + "\n" for t in targets))
    counts: dict[str, int] = {}
    for t in targets:
        counts[t["probe"]] = counts.get(t["probe"], 0) + 1
    size = (out / "snapshot.jsonl").stat().st_size
    prog.line("snapshot_done", targets=len(targets), by_probe=json.dumps(counts, separators=(",", ":")),
              snapshot_bytes=size)
    if size > 100 * 1024 * 1024:
        print("snapshot exceeds 100MB: stop and ask Juniper (CLAUDE.md section 14)", file=sys.stderr)
        return 4
    return 0


def deploy_check() -> list[str]:
    """What is missing from the running Hub for a re-read to be safe. [] = ready."""
    missing = []
    for path, marker in DEPLOY_MARKERS:
        r = subprocess.run(["docker", "exec", HUB_CONTAINER, "grep", "-q", marker, path],
                           capture_output=True, text=True)
        if r.returncode != 0:
            missing.append(f"{path} lacks {marker!r} ({(r.stderr or '').strip()[:120] or 'not found'})")
    r = subprocess.run(["docker", "exec", HUB_CONTAINER, "printenv", "HUB_WORLD_PULSE_READ_MEANING_TARGET_LINKS_ENABLED"],
                       capture_output=True, text=True)
    if r.stdout.strip().lower() not in {"true", "1", "yes"}:
        missing.append(f"HUB_WORLD_PULSE_READ_MEANING_TARGET_LINKS_ENABLED in {HUB_CONTAINER} is {r.stdout.strip()!r}")
    return missing


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every whole JSON line; a line cut off by a kill mid-write is skipped, not fatal."""
    out = []
    if not path.exists():
        return out
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except ValueError:
            print(f"skipping unparseable line in {path.name}: {ln[:80]!r}", file=sys.stderr)
    return out


async def _pool_quiet(client: Any, pool_url: str, wait_sec: float) -> list[str]:
    from evals.run_reading_resonance_eval import wait_for_quiet_pool

    return await wait_for_quiet_pool(client, pool_url, wait_sec, None)


class RestoreRefused(RuntimeError):
    pass


async def restore_seed(conn: Any, snap: dict[str, Any]) -> None:
    """Write the snapshotted values of every SEED_COLUMNS column back onto the seed row.
    Refused while the row is in flight (a stage pending or claimed): a worker finishing
    later would overwrite the restore with its own result. Once both stages are terminal,
    a still-unconsumed run is a finished one (as in operator.reread_read) and is consumed,
    so the restored row cannot be bound to it again."""
    async with conn.transaction():
        live = await conn.fetchrow("SELECT status, stage2_status FROM world_pulse_read_seed "
                                   "WHERE seed_id = $1 FOR UPDATE", snap["seed_id"])
        if live is None:
            raise RestoreRefused("seed_missing")
        in_flight = live["status"] in ("pending", "claimed") or (
            live["status"] == "done" and live["stage2_status"] in ("pending", "claimed"))
        if in_flight:
            raise RestoreRefused(f"in_flight status={live['status']} stage2={live['stage2_status']}")
        await conn.execute("UPDATE reading_durable_turn SET consumed_at = now() "
                           "WHERE seed_id = $1 AND consumed_at IS NULL", snap["seed_id"])
        await _write_seed(conn, snap)


async def _write_seed(conn: Any, snap: dict[str, Any]) -> None:
    row = snap["seed_row"]
    sets, vals = [], []
    for i, col in enumerate(SEED_COLUMNS, start=2):
        v = row.get(col)
        if col.endswith("_json") and v is not None:
            sets.append(f"{col} = ${i}::jsonb")
            v = json.dumps(v)
        elif col.endswith("_at") and v is not None:
            sets.append(f"{col} = ${i}::timestamptz")
            v = datetime.fromisoformat(v)
        else:
            sets.append(f"{col} = ${i}")
        vals.append(v)
    await conn.execute(f"UPDATE world_pulse_read_seed SET {', '.join(sets)} WHERE seed_id = $1",
                       snap["seed_id"], *vals)


async def _settled(conn: Any, seed_id: str) -> tuple[str, Any] | None:
    row = await conn.fetchrow(SEED_ROW_SQL, seed_id)
    if row["status"] in ("failed", "skipped"):
        return "stage1_" + row["status"], row
    if row["status"] == "done" and row["stage2_status"] in ("done", "failed", "skipped"):
        return "stage2_" + row["stage2_status"], row
    return None


def _parse_ts(v: Any) -> datetime | None:
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        d = datetime.fromisoformat(str(v))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _requeued_at_from_log(out: Path, targets: list[dict[str, Any]]) -> dict[str, datetime]:
    """When each seed was (last) requeued, from progress.log ("<ts> requeued <seed> | ... url=<url[:80]>").
    Recovers the start of a read that was requeued by a run stopped before it wrote an
    ``in_flight`` record (any run of the pre-concurrency code). Lines written before 2026-10-14
    carry only the first 60 characters of the seed id, which several digest_item ids share: an
    ambiguous prefix is resolved by the line's url=, and dropped if that does not settle it."""
    found: dict[str, datetime] = {}
    path = out / "progress.log"
    if not path.exists():
        return found
    for ln in path.read_text(errors="replace").splitlines():
        parts = ln.split(" ", 2)
        if len(parts) != 3 or parts[1] != "requeued":
            continue
        ts = _parse_ts(parts[0])
        key = parts[2].split(" |", 1)[0].strip()
        if ts is None or not key:
            continue
        hits = [t for t in targets if t["seed_id"] == key]
        if not hits and len(key) == 60:
            hits = [t for t in targets if t["seed_id"].startswith(key)]
            if len(hits) > 1:
                url = ln.split(" url=", 1)[1].split(" ", 1)[0] if " url=" in ln else None
                hits = [t for t in hits if url is not None and t["url"][:80] == url]
        if len(hits) == 1:
            found[hits[0]["seed_id"]] = ts
    return found


def _already_requeued(live: Any, snap: dict[str, Any] | None) -> bool:
    """True when the live row shows this run already put the seed back through the reader:
    a stage is in flight, or a stage finished at a different time than the snapshot."""
    if live is None:
        return False
    if live["status"] in ("pending", "claimed") or (
            live["status"] == "done" and live["stage2_status"] in ("pending", "claimed")):
        return True
    old = (snap or {}).get("seed_row") or {}
    for col in ("completed_at", "stage2_completed_at"):
        if _parse_ts(live[col]) != _parse_ts(old.get(col)):
            return True
    return False


# The run_ids of this backfill's own reads still in flight; their GPU-pool leases are
# held by "durable-runs:<run_id>" and are not someone else's work.
OWN_RUNS_SQL = """
SELECT run_id FROM reading_durable_turn WHERE seed_id = ANY($1::text[]) AND consumed_at IS NULL
"""


def _foreign_busy(state: Any, own_runs: set[str], own_waiting: int) -> list[str]:
    """pool_busy_reasons minus this backfill's own in-flight reads: their leases (holder
    ``durable-runs:<run_id>``), and up to ``own_waiting`` of the queue/backlog depth (our
    own holds waiting for a seat). Anything else still pauses the next start."""
    from evals.run_reading_resonance_eval import pool_busy_reasons

    if not isinstance(state, dict):
        return pool_busy_reasons(state)
    own = _own_lease_ids(state, own_runs)
    leases = [x for x in state.get("leases") or [] if not (isinstance(x, dict) and x.get("lease_id") in own)]
    depth = sum(int(v or 0) for key in ("queue_depth", "backlog_depth") for v in (state.get(key) or {}).values())
    trimmed = dict(state, leases=leases)
    if depth <= own_waiting:
        trimmed["queue_depth"], trimmed["backlog_depth"] = {}, {}
    return pool_busy_reasons(trimmed)


def _own_lease_ids(state: dict[str, Any], own_runs: set[str]) -> set[str]:
    """This run's holds (holder ``durable-runs:<run_id>``) and the gateway calls running inside
    them (``hold_lease_id`` = one of those holds)."""
    holders = {f"durable-runs:{r}" for r in own_runs}
    leases = [x for x in state.get("leases") or [] if isinstance(x, dict)]
    holds = {x.get("lease_id") for x in leases if x.get("holder") in holders}
    return holds | {x.get("lease_id") for x in leases if x.get("hold_lease_id") in holds}


async def _pool_busy(client: Any, pool_url: str, wait_sec: float, own_runs: set[str]) -> list[str]:
    """Like _pool_quiet, but this backfill's own in-flight reads are not contention."""
    if not own_runs:
        return await _pool_quiet(client, pool_url, wait_sec)
    from orion.schemas.gpu_pool import TERMINAL_STATUSES

    deadline = time.monotonic() + max(0.0, wait_sec)
    while True:
        try:
            resp = await client.get(pool_url.rstrip("/") + "/v1/pool", timeout=10.0)
            resp.raise_for_status()
            state = resp.json()
            own = _own_lease_ids(state, own_runs)
            own_waiting = sum(1 for x in state.get("leases") or []
                              if isinstance(x, dict) and x.get("lease_id") in own
                              and x.get("status") not in TERMINAL_STATUSES and x.get("status") != "granted")
            reasons = _foreign_busy(state, own_runs, own_waiting)
        except Exception as exc:  # noqa: BLE001 -- unreadable pool = busy (no state, no start)
            reasons = [f"pool state unreadable: {type(exc).__name__}"]
        if not reasons or time.monotonic() >= deadline:
            return reasons
        await asyncio.sleep(5.0)


def hub_parallel_check(concurrency: int) -> list[str]:
    """--concurrency > 1 needs Hub's reading workers to run that many reads at once;
    on the old one-claim-per-tick Hub the extra reads only queue and run out their timers."""
    if concurrency <= 1:
        return []
    r = subprocess.run(["docker", "exec", HUB_CONTAINER, "grep", "-q", "max_in_flight",
                        "/app/scripts/world_pulse_read_pipeline.py"], capture_output=True, text=True)
    if r.returncode != 0:
        return [f"{HUB_CONTAINER} reading pipeline has no max_in_flight (Hub reads one at a time)"]
    r = subprocess.run(["docker", "exec", HUB_CONTAINER, "printenv", "HUB_WORLD_PULSE_READ_MAX_IN_FLIGHT"],
                       capture_output=True, text=True)
    raw = r.stdout.strip()
    value = int(raw) if raw.isdigit() else 3   # unset = Hub's settings default (3)
    if value < concurrency:
        return [f"HUB_WORLD_PULSE_READ_MAX_IN_FLIGHT in {HUB_CONTAINER} is {raw or 'unset'!r} "
                f"(< --concurrency {concurrency})"]
    return []


async def cmd_run(pool: Any, args: argparse.Namespace, out: Path) -> int:
    """Up to --concurrency reads in flight. A new read starts only while the reading queue
    has no live work and the pool has no work but this run's own reads. Every requeue is
    recorded ``in_flight`` before it is waited on, so a run stopped at any point resumes by
    waiting on those reads, never requeueing them (a requeued row whose run left no record,
    e.g. under the pre-concurrency code, is recognized from the live row and progress.log)."""
    import httpx

    from orion.world_pulse_read.operator import OperatorActionError, reread_read

    if not args.yes:
        print("run writes to live stores; pass --yes (after Hub is deployed with this code)", file=sys.stderr)
        return 1
    concurrency = max(1, int(args.concurrency))
    if not args.skip_deploy_check:
        missing = deploy_check() + hub_parallel_check(concurrency)
        if missing:
            print("REFUSED: Hub is not on the reread code yet:\n  " + "\n  ".join(missing), file=sys.stderr)
            return 5
    targets = json.loads((out / "targets.json").read_text())
    target_ids = [t["seed_id"] for t in targets]
    snaps = {s["seed_id"]: s for s in _load_jsonl(out / "snapshot.jsonl")}
    results_path = out / "results.jsonl"
    latest = {r["seed_id"]: r for r in _load_jsonl(results_path)}  # last record per seed wins
    meta = out / "run_meta.json"
    if not meta.exists():
        meta.write_text(json.dumps({"run_started_at": _now().isoformat()}))
    run_started = _parse_ts(json.loads(meta.read_text()).get("run_started_at")) or _now()
    # A read that is requeued but not settled is not done: on resume it is waited for
    # (never re-queued) first.
    waiting = {s: r for s, r in latest.items() if r["outcome"] in WAITING_OUTCOMES}
    done_ids = set(latest) - set(waiting)
    log_requeued = _requeued_at_from_log(out, targets)
    adopted: list[dict[str, Any]] = []
    async with pool.acquire() as conn:
        for t in targets:
            sid = t["seed_id"]
            if t["probe"] == "gone" or sid in latest:
                continue
            if _already_requeued(await conn.fetchrow(SEED_ROW_SQL, sid), snaps.get(sid)):
                started = log_requeued.get(sid) or run_started
                waiting[sid] = {"seed_id": sid, "outcome": "in_flight", "started_at": started.isoformat(),
                                "adopted": True}
                adopted.append(waiting[sid])
            elif sid in log_requeued:
                # Requeued, then restored to the snapshot (a re-read that did not finish), and the
                # run stopped before recording it: done, not to be read again.
                rec = {"seed_id": sid, "outcome": "restored_unrecorded",
                       "started_at": log_requeued[sid].isoformat(), "adopted": True}
                latest[sid] = rec
                done_ids.add(sid)
                adopted.append(rec)
    todo_new = [t for t in targets if t["probe"] != "gone" and t["seed_id"] not in done_ids
                and t["seed_id"] not in waiting]
    if args.limit:
        todo_new = todo_new[: args.limit]
    by_id = {t["seed_id"]: t for t in targets}
    resume = [by_id[s] for s in target_ids if s in waiting]
    prog = Progress(out, len(done_ids) + len(resume) + len(todo_new), done=len(done_ids))
    prog.line("run_start", todo=len(todo_new), resuming=len(resume), already_done=len(done_ids),
              gone_skipped=sum(t["probe"] == "gone" for t in targets), concurrency=concurrency)

    def record(rec: dict[str, Any]) -> None:
        with results_path.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    for rec in adopted:   # written once, so a second stop resumes from the file alone
        record(rec)

    timed_out = False
    tasks: dict[asyncio.Task, str] = {}

    def spawn(t: dict[str, Any], started: datetime, consumed: list[str]) -> None:
        tasks[asyncio.create_task(_settle_one(pool, args, t, started, consumed, snaps, prog, record))] = t["seed_id"]

    for t in resume:
        prog.line(f"resume waiting on {t['seed_id']}")
        w = waiting[t["seed_id"]]
        spawn(t, _parse_ts(w.get("started_at")) or run_started, list(w.get("consumed_finished_runs") or []))

    queue = list(todo_new)
    async with httpx.AsyncClient() as client:
        while queue or tasks:
            if queue and not timed_out and len(tasks) < concurrency:
                t = queue[0]
                seed_id = t["seed_id"]
                async with pool.acquire() as conn:
                    own = [tasks[k] for k in tasks]
                    own_runs = {r["run_id"] for r in await conn.fetch(OWN_RUNS_SQL, own)} if own else set()
                # --gate none: re-reads run through the normal reading path at BACKGROUND
                # priority, so the pool already orders them behind live work and the hub's
                # HUB_WORLD_PULSE_READ_MAX_IN_FLIGHT bounds them. The quiet-pool gate (built for
                # the system-priority judge) only serialises them -- live 2026-10-14 it paused
                # --concurrency 3 behind a single unrelated background hold for 30+ min.
                busy = [] if args.gate == "none" else await _pool_busy(client, args.pool_url, 30.0, own_runs)
                if not busy and args.gate != "none":
                    async with pool.acquire() as conn:
                        queued = await conn.fetchval(QUEUE_BUSY_SQL, target_ids, args.digest_item_max_age_sec)
                    busy = [f"{queued} live read(s) in flight on the reading queue"] if queued else []
                if busy:
                    prog.line(f"paused before {seed_id[:50]}", reason=busy[0][:120], in_flight=len(tasks))
                    if tasks:
                        await _wait_some(tasks, args.pause_sec)
                    else:
                        await asyncio.sleep(args.pause_sec)
                    timed_out |= _reap(tasks, prog)
                    continue
                queue.pop(0)
                started = _now()
                async with pool.acquire() as conn:
                    try:
                        consumed = (await reread_read(conn, seed_id)).get("consumed_finished_runs", [])
                    except OperatorActionError as exc:
                        prog.errors += 1
                        prog.done += 1
                        record({"seed_id": seed_id, "outcome": f"refused:{exc.code}", "started_at": started})
                        prog.line(f"refused {seed_id[:50]}", code=exc.code)
                        continue
                # Recorded before anything waits on it: a run stopped from here on resumes
                # by waiting on this read, never by requeueing it.
                record({"seed_id": seed_id, "outcome": "in_flight", "started_at": started,
                        "consumed_finished_runs": consumed})
                prog.line(f"requeued {seed_id}", url=t["url"][:80], consumed_old_runs=len(consumed),
                          in_flight=len(tasks) + 1)
                spawn(t, started, consumed)
                continue
            if not tasks:
                break  # timed out with nothing left in flight, or nothing to do
            await _wait_some(tasks, args.pause_sec if queue and not timed_out else None)
            timed_out |= _reap(tasks, prog)
    if timed_out:
        prog.line("ANOMALY timeout: stopped starting new reads (resume with run --yes)",
                  minutes=args.per_read_timeout_min, not_started=len(queue))
        return 3
    prog.line("run_done")
    return 0


WAITING_OUTCOMES = frozenset({"in_flight", "timeout_still_running"})


async def _wait_some(tasks: dict[asyncio.Task, str], timeout: float | None) -> None:
    await asyncio.wait(list(tasks), timeout=timeout, return_when=asyncio.FIRST_COMPLETED)


def _reap(tasks: dict[asyncio.Task, str], prog: Progress) -> bool:
    """Drop finished tasks. True when any of them timed out (stop starting new reads)."""
    timed_out = False
    for task in [k for k in tasks if k.done()]:
        seed_id = tasks.pop(task)
        exc = task.exception()
        if exc is not None:
            prog.errors += 1
            prog.line(f"ERROR settling {seed_id[:50]} (resume waits on it)", error=f"{type(exc).__name__}: {exc}"[:200])
            timed_out = True  # unknown state: do not keep starting reads on top of it
        elif task.result() == "timeout":
            timed_out = True
    return timed_out


async def _settle_one(pool: Any, args: argparse.Namespace, t: dict[str, Any], started: datetime,
                      consumed: list[str], snaps: dict[str, Any], prog: Progress, record: Any) -> str:
    """Wait for one requeued read to settle, then record it (restoring the old row if the
    re-read did not finish). Returns "timeout" or the settled outcome."""
    from orion.schemas.world_pulse_read import WorldPulseReadHandoffV1
    from orion.world_pulse_read.fetch_text import load_retained_texts

    seed_id = t["seed_id"]
    deadline = time.monotonic() + args.per_read_timeout_min * 60
    settled = None
    while True:
        async with pool.acquire() as conn:
            settled = await _settled(conn, seed_id)
        if settled or time.monotonic() >= deadline:
            break
        await asyncio.sleep(args.poll_sec)
    if settled is None:
        prog.anomalies += 1
        record({"seed_id": seed_id, "outcome": "timeout_still_running", "started_at": started,
                "consumed_finished_runs": consumed})
        prog.line(f"ANOMALY timeout {seed_id[:50]}: no new reads start (resume with run --yes)",
                  minutes=args.per_read_timeout_min)
        return "timeout"
    outcome, row = settled
    async with pool.acquire() as conn:
        runs = [_row_dict(r) for r in await conn.fetch(NEW_RUNS_SQL, seed_id, started)]
        decisions = [_row_dict(r) for r in await conn.fetch(DECISIONS_SINCE_SQL, seed_id, started)]
        handoff = WorldPulseReadHandoffV1.model_validate(_row_dict(row)["handoff_json"])
        texts = await load_retained_texts(conn, handoff.read_evidence)
        restored, new_row = False, None
        if outcome != "stage2_done":
            # The re-read did not finish: put the old finished read back as it was, so
            # the URL still counts as read and handoff + Stage 2 result stay a pair.
            # The new row is kept in results.jsonl.
            new_row = {c: _row_dict(row).get(c) for c in SEED_COLUMNS}
            try:
                await restore_seed(conn, snaps[seed_id])
                restored = True
            except RestoreRefused as exc:
                prog.anomalies += 1
                prog.line(f"ANOMALY restore refused {seed_id[:50]}", reason=str(exc)[:120])
    result = _row_dict(row).get("stage2_result_json") or {}
    claims = result.get("relationship_claims") or [] if outcome == "stage2_done" else []
    reasons: dict[str, int] = {}
    for c in claims:
        r = (c.get("receipt") or {}).get("reason", "?")
        reasons[r] = reasons.get(r, 0) + 1
    rec = {
        "seed_id": seed_id, "url": t["url"], "outcome": outcome, "started_at": started,
        "finished_at": _now(), "run_ids": [r["run_id"] for r in runs], "restored_old_row": restored,
        "consumed_finished_runs": consumed, "new_row_if_restored": new_row,
        "stage1_error": row["last_error"], "stage2_error": row["stage2_error"],
        "retained_chars": [len(x.text) for x in texts], "claims": len(claims), "reasons": reasons,
        "links_quote_rule": sum(d["policy"] == POLICY for d in decisions),
        "links_meaning_rule": sum(d["policy"] == MEANING_POLICY for d in decisions),
        "decisions": decisions,
    }
    if outcome != "stage2_done" or not texts:
        prog.anomalies += outcome == "stage2_done" and not texts
        prog.errors += outcome != "stage2_done"
    prog.done += 1
    record(rec)
    prog.line(f"{outcome} {seed_id[:50]}", text=bool(texts), claims=len(claims),
              quote_links=rec["links_quote_rule"], meaning_links=rec["links_meaning_rule"],
              runs=",".join(rec["run_ids"]))
    return outcome


def cmd_report(out: Path) -> int:
    targets = json.loads((out / "targets.json").read_text())
    results = {r["seed_id"]: r for r in _load_jsonl(out / "results.jsonl")}
    rows = []
    for t in targets:
        r = results.get(t["seed_id"], {})
        rows.append({"seed_id": t["seed_id"], "url": t["url"], "probe": t["probe"], "http_status": t.get("http_status"),
                     "before_claims": t["before_claims"], "before_accepted": t["before_accepted"],
                     "outcome": r.get("outcome", "skipped_gone" if t["probe"] == "gone" else "not_run"),
                     "after_retained_chars": sum(r.get("retained_chars") or []), "after_claims": r.get("claims", 0),
                     "after_links_quote_rule": r.get("links_quote_rule", 0),
                     "after_links_meaning_rule": r.get("links_meaning_rule", 0),
                     "run_ids": " ".join(r.get("run_ids") or [])})
    with (out / "before_after.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["seed_id"])
        w.writeheader()
        w.writerows(rows)
    unsettled = ("not_run", *WAITING_OUTCOMES)  # requeued but not settled is not done
    ran = [x for x in rows if x["outcome"] not in ("skipped_gone", *unsettled)]
    text_now = sum(1 for x in ran if x["after_retained_chars"])
    links_q = sum(x["after_links_quote_rule"] for x in rows)
    links_m = sum(x["after_links_meaning_rule"] for x in rows)
    errors = sum(1 for x in ran if x["outcome"] != "stage2_done")
    pending = sum(1 for x in rows if x["outcome"] in unsettled)
    verdict = ("NOT RUN" if not ran else "INCOMPLETE" if pending else
               "DONE" if errors == 0 else "DONE_WITH_ERRORS")
    md = [f"# Reading reread backfill report ({_now().isoformat(timespec='seconds')})", "",
          f"Verdict: **{verdict}**", "",
          f"- targets: {len(rows)} (gone, skipped: {sum(x['probe'] == 'gone' for x in rows)})",
          f"- re-read: {len(ran)}; now with retained text: {text_now}; errors: {errors}; not run yet: {pending}",
          f"- links made: {links_q} by reading_quote_rule_v1, {links_m} by reading_meaning_target_v1",
          f"- needs another pass: {'yes' if pending or errors else 'no'}", "",
          "| seed | probe | outcome | text chars | claims | quote links | meaning links |",
          "|---|---|---|---|---|---|---|"]
    md += [f"| {x['seed_id'][:48]} | {x['probe']} {x['http_status'] or ''} | {x['outcome']} | "
           f"{x['after_retained_chars']} | {x['after_claims']} | {x['after_links_quote_rule']} | "
           f"{x['after_links_meaning_rule']} |" for x in rows]
    md += ["", "Rollback: `python3 scripts/reading_reread_backfill.py rollback --yes [--restore-seeds]`",
           f"Files: {out}/targets.json, snapshot.jsonl, seeds.txt, results.jsonl, progress.log, "
           "before_after.csv, report.md"]
    (out / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:9]))
    return 0


async def cmd_rollback(pool: Any, args: argparse.Namespace, out: Path) -> int:
    from orion.substrate.graph_journal import RevisionConflict, SubstrateGraphJournal
    from orion.world_pulse_read.meaning_links import list_policy_links, withdrawal_for

    seeds = [ln.strip() for ln in (out / "seeds.txt").read_text().splitlines() if ln.strip()]
    meta = json.loads((out / "run_meta.json").read_text()) if (out / "run_meta.json").exists() else {}
    since = datetime.fromisoformat(meta["run_started_at"]) if meta.get("run_started_at") else None
    if since is None:
        print("no run_meta.json: the backfill never ran; nothing to roll back")
        return 0
    journal = SubstrateGraphJournal(pool)
    async with pool.acquire() as conn:
        links = await list_policy_links(conn, policies=[MEANING_POLICY, POLICY], since=since, seed_ids=seeds)
    written = 0
    for link in links:
        d = withdrawal_for(link, recorded_at=_now(), reason="reading reread backfill rollback")
        line = {"assertion_id": link.decided.target_id, "policy": link.decided.policy, "status": link.status,
                "withdrawal": d.decision_id if d else None}
        if d is not None and args.yes:
            try:
                line["written"] = bool(await journal.append(d))
            except RevisionConflict:
                line["written"] = False
            written += bool(line.get("written"))
        print(json.dumps(line))
    restored = 0
    if args.restore_seeds:
        snaps = _load_jsonl(out / "snapshot.jsonl")
        results = {r["seed_id"] for r in _load_jsonl(out / "results.jsonl")}
        for snap in snaps:
            if snap["seed_id"] not in results:
                continue
            if args.yes:
                try:
                    async with pool.acquire() as conn:
                        await restore_seed(conn, snap)
                except RestoreRefused as exc:
                    print(json.dumps({"seed_id": snap["seed_id"], "restore": "refused", "reason": str(exc)}))
                    continue
            restored += 1
    print(json.dumps({"dry_run": not args.yes, "links": len(links), "withdrawals_written": written,
                      "seed_rows_restored" if args.yes else "seed_rows_to_restore": restored}))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Re-read reads with no retained text through the real reading path.")
    ap.add_argument("action", choices=("snapshot", "run", "report", "rollback"))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dsn", default=None)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--until", default="2026-10-14T00:10:00+00:00",
                    help="window end (default: the window #2647's replay and judge run used)")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="run: at most N new reads this invocation")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="run: reads in flight at once (needs Hub HUB_WORLD_PULSE_READ_MAX_IN_FLIGHT >= N)")
    ap.add_argument("--pool-url", default=os.environ.get("GPU_POOL_URL", "http://127.0.0.1:8127"))
    ap.add_argument("--pause-sec", type=float, default=60.0)
    ap.add_argument("--gate", choices=("pool", "none"), default="pool",
                    help="run: 'pool' pauses new starts while any foreign pool lease/queue or live read "
                         "exists (conservative, serialises); 'none' starts up to --concurrency and lets "
                         "the pool's background priority + the hub's in-flight cap do the pacing")
    ap.add_argument("--poll-sec", type=float, default=30.0)
    ap.add_argument("--per-read-timeout-min", type=float, default=240.0)
    ap.add_argument("--digest-item-max-age-sec", type=float, default=5 * 86400.0,
                    help="Hub's HUB_WORLD_PULSE_READ_DIGEST_ITEM_MAX_AGE_DAYS, in seconds (queue gate)")
    ap.add_argument("--skip-deploy-check", action="store_true")
    ap.add_argument("--restore-seeds", action="store_true", help="rollback: also write snapshotted seed rows back")
    return ap.parse_args(argv)


async def _amain(args: argparse.Namespace) -> int:
    import asyncpg

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.action == "report":
        return cmd_report(out)
    try:
        pool = await asyncpg.create_pool(resolve_dsn(args.dsn), min_size=1,
                                         max_size=max(2, int(getattr(args, "concurrency", 1)) + 1))
    except Exception as exc:  # noqa: BLE001
        print(f"UNKNOWN: database unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    try:
        if args.action == "snapshot":
            async with pool.acquire() as conn:
                return await cmd_snapshot(conn, args, out)
        if args.action == "run":
            return await cmd_run(pool, args, out)
        return await cmd_rollback(pool, args, out)
    finally:
        await pool.close()


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_amain(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
