#!/usr/bin/env python3
"""Batch-export GraphDB repository statements into Fuseki /orion dataset.

Prefer the offline path on this host (GraphDB HTTP export hangs/ignores limits):
  scripts/export_graphdb_collapse_offline.sh
  scripts/complete_fuseki_graphdb_migration.sh

This HTTP helper remains for small repos or dry-run checks only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def _request(
    method: str,
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 600.0,
) -> bytes:
    req = urllib.request.Request(url, data=data, method=method)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _count_triples(graphdb_url: str, repo: str) -> int:
    url = f"{graphdb_url.rstrip('/')}/repositories/{repo}/size"
    raw = _request("GET", url).decode().strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return int(raw)
    if isinstance(payload, int):
        return payload
    return int(payload.get("explicit") or payload.get("total") or 0)


def _export_batch(
    graphdb_url: str,
    repo: str,
    *,
    offset: int,
    limit: int,
) -> bytes:
    params = urllib.parse.urlencode({"infer": "false", "offset": offset, "limit": limit})
    url = f"{graphdb_url.rstrip('/')}/repositories/{repo}/statements?{params}"
    return _request("GET", url, headers={"Accept": "application/n-triples"})


def _import_batch(fuseki_data_url: str, body: bytes) -> None:
    if not body.strip():
        return
    _request(
        "POST",
        fuseki_data_url,
        data=body,
        headers={"Content-Type": "application/n-triples"},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate GraphDB repo statements into Fuseki")
    parser.add_argument("--graphdb-url", default="http://localhost:7200")
    parser.add_argument("--graphdb-repo", default="collapse")
    parser.add_argument("--fuseki-data-url", default="http://localhost:3030/orion/data")
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--max-batches", type=int, default=0, help="0 = all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        total = _count_triples(args.graphdb_url, args.graphdb_repo)
    except urllib.error.URLError as exc:
        print(f"graphdb_unreachable: {exc}", file=sys.stderr)
        return 1

    print(f"graphdb_repo={args.graphdb_repo} explicit_triples={total} batch_size={args.batch_size}")

    if args.dry_run:
        sample = _export_batch(args.graphdb_url, args.graphdb_repo, offset=0, limit=min(10, args.batch_size))
        print(f"dry_run sample_bytes={len(sample)}")
        return 0

    offset = 0
    batch_num = 0
    imported_lines = 0
    started = time.time()

    while offset < total:
        if args.max_batches and batch_num >= args.max_batches:
            break
        try:
            body = _export_batch(
                args.graphdb_url,
                args.graphdb_repo,
                offset=offset,
                limit=args.batch_size,
            )
            _import_batch(args.fuseki_data_url, body)
        except urllib.error.HTTPError as exc:
            print(f"http_error batch={batch_num + 1} offset={offset} code={exc.code}", file=sys.stderr)
            return 1
        except urllib.error.URLError as exc:
            print(f"url_error batch={batch_num + 1} offset={offset} err={exc}", file=sys.stderr)
            return 1

        line_count = body.count(b"\n") if body else 0
        offset += args.batch_size
        batch_num += 1
        imported_lines += line_count
        elapsed = time.time() - started
        print(
            f"batch={batch_num} offset={offset} lines={line_count} "
            f"cumulative_lines={imported_lines} elapsed_sec={elapsed:.1f}",
            flush=True,
        )
        if not body.strip():
            break

    print(
        f"done batches={batch_num} cumulative_lines={imported_lines} "
        f"elapsed_sec={time.time() - started:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
