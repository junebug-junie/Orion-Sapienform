#!/usr/bin/env python3
"""Replay stored metacog triggers through orion.metacog.evidence_map (read-only).

Spec: docs/superpowers/specs/2026-09-24-metacog-capture-and-transport-ewma-baseline-design.md,
section C / acceptance checks 4 and 5.

Reads the last N days of `metacog_trigger` (the raw, honest trigger table) and,
where a row can be joined on correlation_id, the severity/causal_density the
OLD pipeline actually published to `orion_metacog`. Maps every trigger through
the new deterministic mapper and writes a markdown report.

Writes nothing to the database (session is forced read-only by
orion.db_readonly). Output: /tmp/metacog-capture-replay/report.md and
summary.json.

    POSTGRES_URI=postgresql://postgres:postgres@localhost:55432/conjourney \\
      /mnt/scripts/Orion-Sapienform/.venv/bin/python scripts/analysis/replay_metacog_capture.py --days 7

    # offline, from a JSONL of {trigger_kind, reason, upstream, timestamp}:
    ... replay_metacog_capture.py --input orion/metacog/tests/fixtures/metacog_trigger_sample.jsonl

Honesty note on acceptance check 4: for kinds whose severity is a banded
function of ONE raw quantity (telemetry recon ratio, bus_synaptic ratio),
Spearman(severity, that quantity) is high by construction -- it is a regression
guard that the mapper really reads the number, not independent validation.
The transport rpc_health rho is the more informative one, because its
severity mixes timeouts, latency ratio and a thin-sample cap.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.metacog.capture_replay import acceptance_failures, analyze, render_report  # noqa: E402

DEFAULT_DSN = "postgresql://postgres:postgres@localhost:55432/conjourney"
OUT_DIR = Path("/tmp/metacog-capture-replay")


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------


_SQL = """
SELECT t.trigger_kind,
       t.reason,
       t.upstream,
       to_char(t.timestamp, 'YYYY-MM-DD"T"HH24:MI:SS') AS ts,
       m.severity AS old_severity,
       (m.causal_density->>'score')::float AS old_density_score
FROM metacog_trigger t
LEFT JOIN LATERAL (
    SELECT severity, causal_density
    FROM orion_metacog om
    WHERE om.correlation_id = t.correlation_id AND t.correlation_id IS NOT NULL
    LIMIT 1
) m ON TRUE
WHERE t.timestamp > now() - (%s || ' days')::interval
ORDER BY t.timestamp
"""


def load_rows_from_db(dsn: str, days: int) -> list[dict[str, Any]]:
    from orion.db_readonly import open_readonly_connection

    conn = open_readonly_connection(dsn, connect_timeout=10, statement_timeout_ms=300_000)
    if conn is None:
        raise SystemExit(f"could not open a read-only session on {dsn!r}")
    try:
        with conn.cursor() as cur:
            cur.execute(_SQL, (str(int(days)),))
            out = []
            for kind, reason, upstream, ts, old_sev, old_score in cur.fetchall():
                if isinstance(upstream, str):
                    try:
                        upstream = json.loads(upstream)
                    except ValueError:
                        upstream = None
                out.append(
                    {
                        "trigger_kind": kind,
                        "reason": reason,
                        "upstream": upstream,
                        "timestamp": ts,
                        "old_severity": old_sev,
                        "old_density_score": old_score,
                    }
                )
            return out
    finally:
        conn.close()


def load_rows_from_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--input", type=Path, default=None, help="JSONL instead of Postgres")
    ap.add_argument("--dsn", default=os.environ.get("POSTGRES_URI", DEFAULT_DSN))
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    if args.input:
        rows = load_rows_from_jsonl(args.input)
        source = str(args.input)
    else:
        rows = load_rows_from_db(args.dsn, args.days)
        source = f"metacog_trigger, last {args.days} days (joined to orion_metacog on correlation_id)"

    res = analyze(rows)
    fails = acceptance_failures(res)
    res["acceptance_failures"] = fails
    report = render_report(res, source=source)
    report += "\n## Acceptance verdict (checks 4-6, deterministic half)\n\n"
    report += ("PASS\n" if not fails else "FAIL\n\n" + "\n".join(f"- {f}" for f in fails) + "\n")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.md").write_text(report)
    (args.out_dir / "summary.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"rows={res['rows']} acceptance={'PASS' if not fails else 'FAIL'} report={args.out_dir / 'report.md'}")
    for f in fails:
        print(f"  FAIL {f}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
