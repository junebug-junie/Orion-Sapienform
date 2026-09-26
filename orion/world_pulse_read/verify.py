"""Read-only acceptance check for a real reading, including stored journal bodies.

Run inside Hub: python3 -m orion.world_pulse_read.verify --url https://...
Exit 0 means verified complete, 2 means incomplete, 1 means check unavailable.
No inference, retry, queue mutation, or journal repair is performed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any

from orion.schemas.world_pulse_read import WorldPulseReadHandoffV1, WorldPulseReadStage2ResultV1
from orion.world_pulse_read.read_evidence import source_read_evidence
from orion.world_pulse_read.urls import normalize_source_url


def completion_gaps(row: dict[str, Any], journal_bodies: dict[str, str]) -> list[str]:
    gaps = []
    if row["status"] != "done":
        gaps.append("stage1_not_done")
    if row["stage2_status"] != "done":
        gaps.append("stage2_not_done")
    try:
        raw = row.get("handoff_json")
        handoff = WorldPulseReadHandoffV1.model_validate_json(raw) if isinstance(raw, str) else WorldPulseReadHandoffV1.model_validate(raw)
        if not source_read_evidence(row["url"], handoff.read_evidence):
            gaps.append("missing_source_fetch_evidence")
        if handoff.trace_id != row.get("trace_id") or handoff.seed_ref.seed_id != row["seed_id"]:
            gaps.append("stage1_provenance_mismatch")
    except ValueError:
        gaps.append("missing_or_invalid_handoff")
    try:
        raw = row.get("stage2_result_json")
        result = WorldPulseReadStage2ResultV1.model_validate_json(raw) if isinstance(raw, str) else WorldPulseReadStage2ResultV1.model_validate(raw)
        if result.trace_id != row.get("stage2_trace_id") or result.seed_id != row["seed_id"]:
            gaps.append("stage2_provenance_mismatch")
    except ValueError:
        gaps.append("missing_or_invalid_stage2_result")
    for prefix, key in (("world_pulse_read", "trace_id"), ("world_pulse_read_stage2", "stage2_trace_id")):
        ref = f"{prefix}:{row.get(key)}"
        if not str(journal_bodies.get(ref) or "").strip():
            gaps.append(f"missing_{prefix}_journal_body")
    if row.get("landing_at") is None:
        gaps.append("landing_not_confirmed")
    return gaps


async def inspect_reading(conn: Any, url: str) -> dict[str, Any]:
    url = normalize_source_url(url)
    async with conn.transaction(readonly=True, isolation="repeatable_read"):
        selected = await conn.fetchrow(
            "SELECT * FROM world_pulse_read_seed WHERE url=$1 ORDER BY created_at DESC, seed_id DESC LIMIT 1", url,
        )
        if selected is None:
            return {"url": url, "verified_complete": False, "gaps": ["not_found"]}
        row = selected
        if row["duplicate_of"]:
            row = await conn.fetchrow("SELECT * FROM world_pulse_read_seed WHERE seed_id=$1", row["duplicate_of"])
            if row is None:
                return {"url": url, "verified_complete": False, "gaps": ["alias_target_missing"]}
        refs = [f"world_pulse_read:{row['trace_id']}", f"world_pulse_read_stage2:{row['stage2_trace_id']}"]
        entries = await conn.fetch("SELECT source_ref, body FROM journal_entries WHERE source_ref = ANY($1::text[])", refs)
        gaps = completion_gaps(dict(row), {e["source_ref"]: e["body"] for e in entries})
        return {
            "url": url, "request_id": str(selected["request_id"]) if selected["request_id"] else None,
            "seed_id": row["seed_id"], "stage1_status": row["status"], "stage2_status": row["stage2_status"],
            "stage1_error": row["last_error"], "stage2_error": row["stage2_error"],
            "stage1_trace_id": row["trace_id"], "stage2_trace_id": row["stage2_trace_id"],
            "verified_complete": not gaps, "gaps": gaps,
        }


async def _run(args: argparse.Namespace) -> int:
    import asyncpg
    from sqlalchemy.engine import make_url

    uri = make_url(os.environ["POSTGRES_URI"]).set(drivername="postgresql")
    conn = await asyncpg.connect(uri.render_as_string(hide_password=False), command_timeout=15)
    deadline = time.monotonic() + max(0, args.watch_seconds)
    try:
        while True:
            report = await inspect_reading(conn, args.url)
            print(json.dumps(report, default=str), flush=True)
            if report["verified_complete"]:
                return 0
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return 2
            await asyncio.sleep(min(10, remaining))
    finally:
        await conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--watch-seconds", type=float, default=0)
    args = parser.parse_args()
    try:
        return asyncio.run(_run(args))
    except Exception as exc:
        print(json.dumps({"verified_complete": False, "error": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
