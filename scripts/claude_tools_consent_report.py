#!/usr/bin/env python3
"""Weekly numbers for Orion's consent to Claude's tools (spec 2026-11-12-claude-hub-tools-orion-consent-
design.md section 3, "How Orion's decisions are observed").

Reads the ``answer`` rows of claude_tool_call_log (read-only, through the sql-db container's psql) and
prints, per category: asks, model answers, yes rate, unanswered rate, the share of answers with a reason,
consent latency p50/p90, flips within 24 h, and the rubber-stamp flag (20+ answers, all the same).

    .venv/bin/python scripts/claude_tools_consent_report.py [--days 7]

No rows prints "no asks in the window", never a table of zeros.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.claude_tools.report import consent_summary  # noqa: E402

PSQL = ["docker", "exec", "-i", "orion-athena-sql-db", "psql", "-U", "postgres", "-d", "conjourney", "-v", "ON_ERROR_STOP=1", "-At"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=7.0)
    a = ap.parse_args()
    sql = ("SELECT coalesce(json_agg(row_to_json(t)), '[]') FROM (SELECT phase, category, tool, orion_answer, "
           "consent_path, consent_latency_ms, orion_reason, created_at FROM claude_tool_call_log WHERE phase = "
           f"'answer' AND created_at >= now() - interval '{float(a.days)} days' ORDER BY created_at) t;")
    proc = subprocess.run(PSQL, input=sql, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        print(f"claude_tool_call_log unreadable: {proc.stderr.strip()[:300]}", file=sys.stderr)
        return 1
    rows = json.loads(proc.stdout.strip() or "[]")
    if not rows:
        print(f"no asks in the last {a.days:g} days")
        return 0
    print(json.dumps(consent_summary(rows), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
