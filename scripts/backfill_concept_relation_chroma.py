#!/usr/bin/env python3
"""One-off backfill: project every ACTIVE memory crystallization into the Chroma
collection the concept-relation writer searches (`orion_memory_crystallizations`).

Why: `fetch_similar_candidates()` (orion/memory/crystallization/candidate_retrieval.py)
queries that collection for cross-window candidates. Until PR #2600 the live projection
path had no embed host, so `publish_crystallization_to_chroma()` skipped every row with
`no_embedding`; and orion-vector-db runs without IS_PERSISTENT, so any doc that did land
was lost on the next vector-db restart. Result: 0 docs vs ~760 active crystallizations,
and no concept-relation decision since 2026-09-07.

Reuses the live path exactly -- `publish_crystallization_to_chroma()`: same doc text
(`build_chroma_upsert`), same embed endpoint/profile as the writer's query side, same bus
channel (`orion:memory:vector:upsert`) consumed by orion-vector-writer. Idempotent:
Chroma upserts by doc id `crys_<crystallization_id>`, and ids already in the collection
are skipped unless --force. Postgres is read-only here (projection_refs are NOT updated).

Step 0: orion-vector-db must already be running with IS_PERSISTENT=TRUE (this PR's
compose fix) and `chroma.sqlite3` must exist on its mount. Recreating an in-memory
vector-db wipes it, so a backfill run BEFORE that redeploy is lost and must be re-run
after it (re-running is cheap: ids already present are skipped).

Run INSIDE the orion-athena-memory-consolidation container (it has the orion package,
asyncpg, chromadb and the correct env), e.g.:

    docker cp scripts/backfill_concept_relation_chroma.py orion-athena-memory-consolidation:/tmp/bf.py
    docker exec orion-athena-memory-consolidation python /tmp/bf.py --snapshot-only
    docker exec orion-athena-memory-consolidation python /tmp/bf.py | tee /tmp/concept-relation-chroma-backfill/progress.log
    docker cp orion-athena-memory-consolidation:/tmp/concept-relation-chroma-backfill/. /tmp/concept-relation-chroma-backfill/

Section-14 artifacts written to --out-dir: targets.jsonl, chroma_before.json,
chroma_after.json, before_after.csv, report.md; progress lines go to stdout and
progress.log.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

TITLE = "concept-relation-chroma-backfill"
DEFAULT_OUT = "/tmp/concept-relation-chroma-backfill"
DEFAULT_COLLECTION = "orion_memory_crystallizations"
MAX_ROWS = 100_000
MAX_BYTES = 100 * 1024 * 1024


def doc_id_for(crystallization_id: str) -> str:
    # Must match projection_chroma.build_chroma_upsert's doc id.
    return f"crys_{crystallization_id}"


@dataclass
class Progress:
    total: int
    started: float = field(default_factory=time.monotonic)
    processed: int = 0
    published: int = 0
    skipped_existing: int = 0
    errors: int = 0
    anomalies: list[str] = field(default_factory=list)

    def line(self, now: float | None = None) -> str:
        now = time.monotonic() if now is None else now
        elapsed = max(now - self.started, 1e-9)
        rate = self.processed / elapsed
        pct = 100.0 * self.processed / self.total if self.total else 100.0
        remaining = self.total - self.processed
        eta = f"{remaining / rate:.0f}s" if rate > 0 else "unknown"
        anomalies = ";".join(self.anomalies[-3:]) or "none"
        return (
            f"{TITLE} | {pct:5.1f}% | ETA {eta} | {self.processed}/{self.total} | "
            f"{rate:.2f} rows/s | published={self.published} skipped_existing={self.skipped_existing} | "
            f"errors={self.errors} | anomalies={anomalies}"
        )


def build_before_after_rows(
    target_ids: Iterable[str],
    before_ids: set[str],
    after_ids: set[str],
    results: dict[str, str],
) -> list[dict[str, str]]:
    rows = []
    for cid in target_ids:
        did = doc_id_for(cid)
        rows.append(
            {
                "crystallization_id": cid,
                "doc_id": did,
                "in_chroma_before": str(did in before_ids).lower(),
                "in_chroma_after": str(did in after_ids).lower(),
                "result": results.get(cid, "not_attempted"),
            }
        )
    return rows


async def run_backfill(
    targets: list[Any],
    *,
    existing_ids: set[str],
    publish: Callable[[Any], Awaitable[dict]],
    emit: Callable[[str], None],
    sleep_sec: float = 0.2,
    force: bool = False,
    report_every: int = 25,
) -> tuple[Progress, dict[str, str]]:
    """Publish each target via `publish` (the live projection call). Never raises per row."""
    prog = Progress(total=len(targets))
    results: dict[str, str] = {}
    for row in targets:
        cid = row.crystallization_id
        if not force and doc_id_for(cid) in existing_ids:
            prog.skipped_existing += 1
            results[cid] = "skipped_existing"
        else:
            try:
                res = await publish(row)
                if res.get("published"):
                    prog.published += 1
                    results[cid] = "published"
                else:
                    prog.errors += 1
                    reason = res.get("reason") or "not_published"
                    results[cid] = f"error:{reason}"
                    prog.anomalies.append(f"{cid}:{reason}")
            except Exception as exc:  # one bad row must not kill the job
                prog.errors += 1
                results[cid] = f"error:{type(exc).__name__}"
                prog.anomalies.append(f"{cid}:{type(exc).__name__}")
            if sleep_sec > 0:
                await asyncio.sleep(sleep_sec)
        prog.processed += 1
        if prog.processed % report_every == 0 or prog.processed == prog.total:
            emit(prog.line())
    return prog, results


# ---------------------------------------------------------------- live I/O helpers


def _chroma_ids(host: str, port: int, collection: str) -> tuple[int, set[str], bool]:
    import chromadb  # type: ignore

    client = chromadb.HttpClient(host=host, port=port)
    client.heartbeat()  # unreachable chroma must raise, not read as "empty collection"
    try:
        coll = client.get_collection(collection)
    except Exception as exc:
        # chromadb 0.4.x HttpClient raises a bare Exception("Collection X does not exist.")
        if "does not exist" not in str(exc):
            raise
        return 0, set(), False
    got = coll.get(include=[])
    ids = set(got.get("ids") or [])
    return len(ids), ids, True


async def _load_targets(pool) -> list[Any]:
    from orion.memory.crystallization.repository import get_crystallization

    rows = await pool.fetch(
        "SELECT crystallization_id::text AS id FROM memory_crystallizations "
        "WHERE status = 'active' ORDER BY created_at"
    )
    out = []
    for r in rows:
        c = await get_crystallization(pool, r["id"])
        if c is not None and c.status == "active":
            out.append(c)
    return out


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


async def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--dsn", default=os.getenv("POSTGRES_URI", ""))
    ap.add_argument("--embed-url", default=os.getenv("CRYSTALLIZER_EMBED_HOST_URL", ""))
    ap.add_argument("--chroma-host", default=os.getenv("CHROMA_HOST", ""))
    ap.add_argument("--chroma-port", type=int, default=int(os.getenv("CHROMA_PORT", "8000")))
    ap.add_argument("--collection", default=os.getenv("CRYSTALLIZER_VECTOR_COLLECTION", DEFAULT_COLLECTION))
    ap.add_argument("--bus-url", default=os.getenv("ORION_BUS_URL", ""))
    ap.add_argument("--sleep", type=float, default=0.2, help="pause between rows (embed-host throttle)")
    ap.add_argument("--settle-sec", type=float, default=20.0, help="wait for vector-writer before the after-snapshot")
    ap.add_argument("--force", action="store_true", help="re-upsert ids already present")
    ap.add_argument("--snapshot-only", action="store_true")
    args = ap.parse_args(argv)

    for name in ("dsn", "embed_url", "chroma_host", "bus_url"):
        if not getattr(args, name).strip():
            print(f"missing --{name.replace('_', '-')}", file=sys.stderr)
            return 2

    import asyncpg

    from orion.core.bus.async_service import OrionBusAsync
    from orion.memory.crystallization.chroma_publish import publish_crystallization_to_chroma

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "progress.log"

    def emit(line: str) -> None:
        stamped = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {line}"
        print(stamped, flush=True)
        with log_path.open("a") as fh:
            fh.write(stamped + "\n")

    pool = await asyncpg.create_pool(dsn=args.dsn, min_size=1, max_size=2)
    try:
        targets = await _load_targets(pool)
    finally:
        await pool.close()
    if len(targets) > MAX_ROWS:
        emit(f"{TITLE} | ABORT target rows {len(targets)} > {MAX_ROWS}; ask Juniper")
        return 3

    with (out / "targets.jsonl").open("w") as fh:
        for c in targets:
            fh.write(json.dumps({
                "crystallization_id": c.crystallization_id, "kind": c.kind, "subject": c.subject,
                "projection_refs_chroma_doc_ids": list(c.projection_refs.chroma_doc_ids),
            }) + "\n")
    before_count, before_ids, before_exists = _chroma_ids(args.chroma_host, args.chroma_port, args.collection)
    snapshot_path = out / "chroma_before.json"
    if not snapshot_path.exists() or args.snapshot_only:
        _write_json(snapshot_path, {
            "taken_at": datetime.now(timezone.utc).isoformat(), "collection": args.collection,
            "exists": before_exists, "count": before_count, "ids": sorted(before_ids),
        })
    else:
        # Keep the first (pre-job) snapshot authoritative across re-runs.
        prior = json.loads(snapshot_path.read_text())
        before_ids = set(prior.get("ids") or [])
        before_count = int(prior.get("count") or 0)
    size = _dir_bytes(out)
    emit(f"{TITLE} | snapshot targets={len(targets)} chroma_before={before_count} bytes={size}")
    if size > MAX_BYTES:
        emit(f"{TITLE} | ABORT snapshot {size} bytes > {MAX_BYTES}; ask Juniper")
        return 3
    if args.snapshot_only:
        return 0

    bus = OrionBusAsync(url=args.bus_url, enabled=True)
    await bus.connect()
    _, live_ids, _ = _chroma_ids(args.chroma_host, args.chroma_port, args.collection)

    async def publish(row):
        _, res = await publish_crystallization_to_chroma(
            row, bus, collection=args.collection, embed_host_url=args.embed_url,
            embed_mode="http", service_name="orion-memory-consolidation",
        )
        return res

    try:
        emit(Progress(total=len(targets)).line())
        prog, results = await run_backfill(
            targets, existing_ids=live_ids, publish=publish, emit=emit,
            sleep_sec=args.sleep, force=args.force,
        )
    finally:
        await bus.close()

    await asyncio.sleep(args.settle_sec)
    after_count, after_ids, _ = _chroma_ids(args.chroma_host, args.chroma_port, args.collection)
    _write_json(out / "chroma_after.json", {
        "taken_at": datetime.now(timezone.utc).isoformat(), "collection": args.collection,
        "count": after_count, "ids": sorted(after_ids),
    })
    target_ids = [c.crystallization_id for c in targets]
    rows = build_before_after_rows(target_ids, before_ids, after_ids, results)
    with (out / "before_after.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["crystallization_id"])
        w.writeheader()
        w.writerows(rows)
    missing = [r["crystallization_id"] for r in rows if r["in_chroma_after"] != "true"]
    verdict = "OK" if not missing and prog.errors == 0 else "NEEDS_ANOTHER_PASS"
    examples = "\n".join(f"- `{r['doc_id']}`: before={r['in_chroma_before']} after={r['in_chroma_after']} ({r['result']})" for r in rows[:5])
    (out / "report.md").write_text(
        f"# {TITLE} report\n\n"
        f"- verdict: **{verdict}**\n"
        f"- collection: `{args.collection}` on {args.chroma_host}:{args.chroma_port}\n"
        f"- active crystallizations targeted: {len(targets)}\n"
        f"- chroma docs before: {before_count}; after: {after_count}\n"
        f"- published this run: {prog.published}; skipped (already present): {prog.skipped_existing}\n"
        f"- errors: {prog.errors}; targets still missing from chroma: {len(missing)}\n"
        f"- anomalies: {', '.join(prog.anomalies[:20]) or 'none'}\n"
        f"- needs another pass: {'yes' if verdict != 'OK' else 'no'}\n\n"
        f"## Before/after examples\n\n{examples}\n\n"
        f"## Files\n\n- {out}/targets.jsonl\n- {out}/chroma_before.json\n- {out}/chroma_after.json\n"
        f"- {out}/before_after.csv\n- {out}/progress.log\n- {out}/report.md\n"
    )
    emit(f"{TITLE} | DONE verdict={verdict} chroma_after={after_count} missing={len(missing)} errors={prog.errors}")
    return 0 if verdict == "OK" else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
