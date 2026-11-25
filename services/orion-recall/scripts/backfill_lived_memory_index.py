"""Backfill the ``orion_lived_memory`` Chroma collection (issue #3533). CLAUDE.md s14 protocol.

Read-only on Postgres (SELECTs only, via ``orion/memory/lived_memory.py``). Writes go only to the
new Chroma collection, through the normal path: vector-host ``/embedding`` embeds, the upsert is
published on ``orion:vector:semantic:upsert``, orion-vector-writer stores it. The same code runs
every ``RECALL_LIVED_MEMORY_INDEX_INTERVAL_SEC`` inside orion-recall once deployed; this script
only drains the first backlog ahead of that.

Writes under ``/tmp/<job>/`` (ids only, never memory text -- it is private):
  targets.csv        every doc id the pass will make current (the snapshot of what changes)
  progress.log       one line per batch (percent, ETA, rate, errors)
  report.md          verdict and counts
  before_after.csv   doc_id, before (absent|stale|current), after

Usage (host, repo root):
  ORION_BUS_URL=redis://<tailscale-ip>:6379/0 \\
  python services/orion-recall/scripts/backfill_lived_memory_index.py \\
      --dsn postgresql://...@127.0.0.1:55432/conjourney \\
      --chroma-url http://127.0.0.1:8500 --embed-url http://127.0.0.1:8320/embedding
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import asyncpg  # noqa: E402
import httpx  # noqa: E402

from orion.core.bus.async_service import OrionBusAsync  # noqa: E402
from orion.core.bus.bus_schemas import ServiceRef  # noqa: E402
from orion.introspect.semantic_index import SearchConfig, doc_hash, index_docs, stored_hashes  # noqa: E402
from orion.memory.lived_memory import COLLECTION, fetch_live_records, index_doc  # noqa: E402

HASH_KEYS = ("occurred_ts",)


async def _states(client, cfg, docs) -> dict[str, str]:
    stored = await stored_hashes(client, cfg, [d[0] for d in docs])
    out = {}
    for did, text, meta in docs:
        have = stored.get(did)
        out[did] = "absent" if have is None else ("current" if have == doc_hash(text, meta, HASH_KEYS) else "stale")
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--chroma-url", default="http://127.0.0.1:8500")
    ap.add_argument("--embed-url", default="http://127.0.0.1:8320/embedding")
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--job", default="lived-memory-index-backfill")
    ap.add_argument("--max-rows", type=int, default=100_000)
    args = ap.parse_args()

    job = Path("/tmp") / args.job
    job.mkdir(parents=True, exist_ok=True)
    progress = (job / "progress.log").open("a", encoding="utf-8")

    def log(line: str) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        progress.write(f"{stamp} {line}\n")
        progress.flush()
        print(line, flush=True)

    cfg = SearchConfig(chroma_url=args.chroma_url, embed_url=args.embed_url, collection=args.collection,
                       min_similarity=0.0, index_batch=args.batch)
    conn = await asyncpg.connect(args.dsn)
    try:
        await conn.execute("SET default_transaction_read_only = on")
        recs = await fetch_live_records(conn, include_ended=True)  # what the index loop mirrors
    finally:
        await conn.close()
    docs = [index_doc(r) for r in recs]
    if len(docs) > args.max_rows:
        log(f"STOP: {len(docs)} rows exceeds --max-rows {args.max_rows}; ask Juniper (s14)")
        return 2
    with (job / "targets.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["doc_id"])
        w.writerows([[d[0]] for d in docs])

    bus = OrionBusAsync(os.environ["ORION_BUS_URL"])
    await bus.connect()
    source = ServiceRef(name="orion-recall", version="backfill", node=os.uname().nodename)
    errors = 0
    async with httpx.AsyncClient(timeout=30.0) as client:
        before = await _states(client, cfg, docs)
        total = sum(1 for s in before.values() if s != "current")
        log(f"event=start total_docs={len(docs)} to_index={total} collection={cfg.collection}")
        done, started = 0, time.monotonic()
        while True:
            try:
                result = await index_docs(docs, cfg, client=client, bus=bus, source=source,
                                          doc_prefix="lived-memory", batch=args.batch, hash_keys=HASH_KEYS)
            except Exception as exc:  # noqa: BLE001
                errors += 1
                log(f"event=batch_error errors={errors} error={type(exc).__name__}: {str(exc)[:160]}")
                if errors >= 5:
                    break
                await asyncio.sleep(5)
                continue
            done += result.indexed
            rate = done / max(1e-6, time.monotonic() - started)
            pct = 100.0 * done / total if total else 100.0
            eta = (result.pending / rate) if rate > 0 else 0
            log(f"event=batch pct={pct:.1f} rows={done}/{total} rate={rate:.2f}/s eta_s={eta:.0f} "
                f"errors={errors} pending={result.pending}")
            if result.pending == 0:
                break
        # The writer stores asynchronously; wait for it before reading back.
        for _ in range(30):
            after = await _states(client, cfg, docs)
            if all(s == "current" for s in after.values()):
                break
            await asyncio.sleep(2)
    await bus.close()

    current = sum(1 for s in after.values() if s == "current")
    with (job / "before_after.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["doc_id", "before", "after"])
        for did, _t, _m in docs:
            w.writerow([did, before[did], after[did]])
    verdict = "complete" if current == len(docs) and errors == 0 else "incomplete"
    (job / "report.md").write_text(
        f"# Lived memory index backfill\n\n- verdict: {verdict}\n- collection: {cfg.collection}\n"
        f"- live records: {len(docs)}\n- indexed this run: {done}\n- current after: {current}\n"
        f"- errors: {errors}\n- needs another pass: {'no' if verdict == 'complete' else 'yes'}\n"
        f"- files: {job}/targets.csv, progress.log, before_after.csv, report.md\n",
        encoding="utf-8")
    log(f"event=done verdict={verdict} current={current}/{len(docs)} errors={errors}")
    return 0 if verdict == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
