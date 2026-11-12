#!/usr/bin/env python3
"""Prune durable-runs' LangGraph checkpoint history to what each thread needs to resume.

DRY RUN BY DEFAULT: prints what it would delete and changes nothing.

    # inside the running container (POSTGRES_URI is already set there)
    docker exec orion-athena-durable-runs python /app/scripts/prune_checkpoints.py            # report
    docker exec orion-athena-durable-runs python /app/scripts/prune_checkpoints.py --apply    # delete

Keeps, per (thread_id, checkpoint_ns): the newest checkpoint, its pending writes, and the blob each
channel's newest version points at. Deletes older checkpoints, their writes, and channel versions
the newest checkpoint no longer names. Rule and evidence: ``app/checkpoint_prune.py``.

Refuses while a pg_dump/pg_dumpall runs in the database (``--allow-during-backup`` overrides).
Tables are the ones on the DSN's search_path (durable-runs: ``public``). Do not point this at the
GPU pool's ``gpu_pool`` schema: that service's resume rules were not checked.

Space: deleted rows become reusable after autovacuum; the files do not shrink on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) > 3 else SERVICE_ROOT
sys.path[:0] = [str(SERVICE_ROOT), str(REPO_ROOT)]

from app.checkpoint_prune import run_once  # noqa: E402


def like_prefix(raw: str) -> str:
    """A LIKE pattern matching ids that start with ``raw`` literally (``_`` and ``%`` escaped)."""
    return raw.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dsn", default=os.environ.get("POSTGRES_URI"), help="default: $POSTGRES_URI")
    p.add_argument("--apply", action="store_true", help="delete (default: dry run, report only)")
    p.add_argument("--thread-prefix", default=None,
                   help="only threads whose id starts with this, e.g. 'temporal_self:'")
    p.add_argument("--batch-threads", type=int, default=200, help="threads per delete transaction")
    p.add_argument("--allow-during-backup", action="store_true")
    p.add_argument("--json", action="store_true", help="one JSON object instead of text")
    return p.parse_args(argv)


def _print_text(report: dict) -> None:
    t = report["totals"]
    print(f"mode: {report['mode']}")
    print(f"threads: {t['threads']} (to prune: {t['threads_to_prune']})")
    print(f"checkpoints: {t['ckpt_total']} total, {t['ckpt_del']} older than their thread's newest")
    print(f"writes:      {t['writes_total']} total, {t['writes_del']} on those older checkpoints")
    print(f"blobs:       {t['blobs_total']} total, {t['blobs_del']} superseded "
          f"({t['blob_bytes_del'] / 1e6:.1f} MB of blob payload)")
    print("by thread family (threads, checkpoints del/total, writes del, blobs del/total):")
    for f in report["by_family"]:
        print(f"  {f['family']:<45} {f['threads']:>6}  {f['ckpt_del']:>7}/{f['ckpt_total']:<7} "
              f"{f['writes_del']:>7}  {f['blobs_del']:>7}/{f['blobs_total']:<7}")
    print("largest threads by blob rows:")
    for r in report["largest_threads"]:
        print(f"  {r['thread_id']:<55} blobs {r['blobs_total']:>6} -> {r['blobs_total'] - r['blobs_del']:<4} "
              f"checkpoints {r['ckpt_total']:>5} -> {r['ckpt_total'] - r['ckpt_del']}")
    if report.get("deleted"):
        print(f"deleted: {report['deleted']} in {report['apply_sec']:.1f} s")
    elif report["mode"] == "dry_run":
        print("dry run: nothing deleted. Re-run with --apply to delete.")


def main(argv=None) -> int:
    a = _args(argv)
    if not a.dsn:
        print("no DSN: pass --dsn or set POSTGRES_URI", file=sys.stderr)
        return 2
    prefix = like_prefix(a.thread_prefix) if a.thread_prefix else None
    report = run_once(a.dsn, apply=a.apply, prefix=prefix, batch_threads=a.batch_threads,
                      allow_during_backup=a.allow_during_backup)
    if report["status"] == "skipped_backup":
        print(f"refusing: backup running {report['backups']}; re-run after it ends", file=sys.stderr)
        return 3
    if a.json:
        print(json.dumps(report, default=str))
    else:
        _print_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
