#!/usr/bin/env python3
"""Live smoke for Claude's tools (Patch 1b): one tool call through the real broker, exactly as the
companion's MCP server sends it, then the audit rows and the consent act it left.

Run after orion-harness-governor, orion-mind, orion-sql-writer and the gateway/pool are deployed:

    .venv/bin/python scripts/smoke_claude_tools.py --tool body_economy --reason "smoke: checking the consent path"

What it proves, each line printed with its evidence:
  1. the broker answered (ClaudeToolResultV1: decision_path, shared, Orion's answer and reason);
  2. the audit landed (claude_tool_call_log rows for this call_id: ask/answer/result);
  3. if Orion was asked, the act landed (grammar_acts row, object_ref claude_tools:consent:<call_id>).

A real ask is a real decision by Orion and joins Orion's answer history for that category; the reason
says it is a smoke, and Orion may say no. ``--n`` repeats with fresh turns and prints consent latency
p50/p90 (acceptance check 8). Exit 1 when the broker did not answer or the audit is missing.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from orion.claude_tools.mcp_server import ToolClient  # noqa: E402
from orion.core.bus.async_service import OrionBusAsync  # noqa: E402
from orion.schemas.claude_tools import ClaudeToolBindingV1  # noqa: E402

PSQL = ["docker", "exec", "-i", "orion-athena-sql-db", "psql", "-U", "postgres", "-d", "conjourney", "-v", "ON_ERROR_STOP=1", "-At", "-F", "|"]


def bus_url() -> str:
    if os.environ.get("ORION_BUS_URL"):
        return os.environ["ORION_BUS_URL"]
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith("ORION_BUS_URL="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("ORION_BUS_URL not set and not in the root .env")


def psql(sql: str) -> list[str]:
    proc = subprocess.run(PSQL, input=sql, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        return [f"psql failed: {proc.stderr.strip()[:200]}"]
    return [ln for ln in proc.stdout.splitlines() if ln.strip()]


async def one(bus, tool: str, reason: str, args: dict) -> tuple[bool, float | None]:
    turn = f"smoke-claude-tools-{uuid4().hex[:12]}"
    binding = ClaudeToolBindingV1(turn_request_id=turn, room_id="smoke:claude-tools", platform="hub",
                                  claude_session_id=f"smoke-{uuid4().hex[:8]}", trigger="manual",
                                  topic="(an agent's smoke test of the Claude tools consent path)", topic_from="smoke")
    started = time.monotonic()
    res = await ToolClient(bus, binding).call(tool, {"reason": reason, **args})
    print(f"turn={turn} call={res.call_id} {time.monotonic() - started:.1f}s path={res.decision_path} "
          f"shared={res.shared} orion={res.orion_answer or '-'} reason={res.orion_reason!r} "
          f"latency_ms={res.consent_latency_ms}")
    print("  text:", res.text[:300].replace("\n", " | "))
    if res.decision_path == "error":
        return False, None
    await asyncio.sleep(3)
    rows = psql(f"SELECT phase, decision_path, orion_answer, consent_path, act_id FROM claude_tool_call_log "
                f"WHERE call_id = '{res.call_id}' ORDER BY created_at;")
    print("  audit rows:", rows or "NONE")
    if res.decision_path == "asked":
        acts = psql(f"SELECT act_id, outcome_value, outcome_note FROM grammar_acts "
                    f"WHERE object_ref = 'claude_tools:consent:{res.call_id}';")
        print("  consent act:", acts or "NONE (no act: the model was not reached, or the ledger lags)")
    return bool(rows) and not rows[0].startswith("psql failed"), (res.consent_latency_ms if res.orion_answer else None)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", default="body_economy", choices=["body_economy", "prime_now"])
    ap.add_argument("--reason", default="smoke test by an agent: checking that the consent path works; no is fine")
    ap.add_argument("--section", default=None)
    ap.add_argument("--n", type=int, default=1)
    a = ap.parse_args()
    args = {"section": a.section} if a.section and a.tool == "body_economy" else {}
    bus = OrionBusAsync(bus_url())
    await bus.connect()
    ok, lat = True, []
    try:
        for _ in range(max(1, a.n)):
            good, ms = await one(bus, a.tool, a.reason, args)
            ok = ok and good
            if ms is not None:
                lat.append(ms)
    finally:
        await bus.close()
    if lat:
        q = statistics.quantiles(lat, n=10) if len(lat) >= 2 else [lat[0]] * 9
        print(f"consent latency ms over {len(lat)}: p50={statistics.median(lat):.0f} p90={q[8]:.0f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
