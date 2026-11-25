#!/usr/bin/env python3
"""Live integration coverage: per registered organ, what it lived, read and used, and who used it. Read-only.

Spec: docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md, Part A.6 "The live coverage
report". For every organ in ``config/integration/organs.yaml``, over the last ``--hours`` (default 24):

* experiences recorded (``temporal_self_event`` rows of its kinds, by when they happened);
* now reads (``experience_shown`` rows it was shown), split (#3498): closed with no cite (its model ran and cited
  nothing) and never closed (its model never saw the read);
* valid cites it made (``experience_use`` by ``citing_organ``);
* valid cites it received from OTHER organs (distinct experiences, within 24 h of each);
* closed: yes when another organ used at least one of its experiences.

Then the experience use rate and the closed organs, by the same SQL as orion-durable-runs
``GET /temporal-self/integration`` (``orion/integration/coverage.py``) and the same arithmetic
(``orion.temporal_self.now.use_rate``). A missing ``experience_*`` table (the migration not applied yet) reads
as no reads and no cites, and is named. The gate (scripts/check_integration_seam.py) cannot see a producer
that writes a new table with no family, workflow, adapter or channel; kinds that land in
``temporal_self_event`` without an organ are listed here as unregistered.

    ORION_LIVE_DSN=postgresql://... python3 scripts/report_integration_coverage.py [--hours 24] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.integration.coverage import (  # noqa: E402
    NOW_READS_SQL,
    RATE_CITES_MADE_SQL,
    RATE_KINDS_SQL,
    RATE_USES_SQL,
)
from orion.integration.organs import load_organs  # noqa: E402
from orion.temporal_self.now import OrganRegistry, use_rate  # noqa: E402


def build_report(*, kind_counts: Mapping[str, int], uses: list[tuple[str, str, str]], cites_made: Mapping[str, int],
                 reads: Mapping[str, tuple[int, ...]], lo: datetime, hi: datetime, registry: OrganRegistry,
                 missing_tables: list[str]) -> dict[str, Any]:
    """Pure: the report from the four reads. ``reads``: organ -> (now reads, closed reads with no cite, reads never
    closed); a missing count reads as 0."""
    rate = use_rate(kind_counts, uses, window_start=lo, window_end=hi, registry=registry, cites_made=cites_made)
    organs = []
    for o in rate.organs:
        r, r0, ru = (tuple(reads.get(o.organ, ())) + (0, 0, 0))[:3]
        organs.append({"organ": o.organ, "experiences": o.experiences, "now_reads": r, "reads_no_cite": r0,
                       "reads_unclosed": ru,
                       "cites_made": o.cites_made, "cites_received": o.used_by_other,
                       "closed": o.used_by_other > 0, "rate": o.rate})
    return {"window_start": lo.isoformat(), "window_end": hi.isoformat(), "experiences": rate.experiences,
            "used_by_other": rate.used_by_other, "experience_use_rate": rate.rate,
            "closed_organs": rate.closed_organs, "closed_count": len(rate.closed_organs),
            "registered_organs": rate.registered_organs, "organs": organs,
            "unregistered_kinds": rate.unregistered_kinds, "missing_tables": missing_tables}


def render(rep: Mapping[str, Any]) -> str:
    lines = [f"Integration coverage {rep['window_start'][:16]}Z .. {rep['window_end'][:16]}Z"]
    lines.append(f"{'organ':<14}{'lived':>8}{'reads':>7}{'no-cite':>9}{'unclosed':>10}{'cites':>7}{'used-by':>9}  closed")
    for o in rep["organs"]:
        lines.append(f"{o['organ']:<14}{o['experiences']:>8}{o['now_reads']:>7}{o['reads_no_cite']:>9}"
                     f"{o['reads_unclosed']:>10}"
                     f"{o['cites_made']:>7}{o['cites_received']:>9}  {'yes' if o['closed'] else 'no'}")
    rate = rep["experience_use_rate"]
    lines.append(f"experience use rate: {rep['used_by_other']} / {rep['experiences']} = "
                 f"{'n/a' if rate is None else f'{rate:.2%}'}; closed organs: {rep['closed_count']} of "
                 f"{rep['registered_organs']} ({', '.join(rep['closed_organs']) or 'none'})")
    if rep["unregistered_kinds"]:
        lines.append(f"unregistered kinds (no organ claims them): {rep['unregistered_kinds']}")
    if rep["missing_tables"]:
        lines.append(f"missing tables (migration manual_migration_experience_now_v1.sql not applied): "
                     f"{', '.join(rep['missing_tables'])} -- read as no reads and no cites")
    return "\n".join(lines)


def read_live(dsn: str, hours: float, now: Optional[datetime] = None) -> dict[str, Any]:
    import psycopg
    from psycopg.rows import dict_row

    hi = now or datetime.now(timezone.utc)
    lo = hi - timedelta(hours=hours)
    params = {"lo": lo, "hi": hi}
    with psycopg.connect(dsn, row_factory=dict_row, autocommit=True) as conn:
        with conn.transaction():
            conn.execute("SET TRANSACTION READ ONLY")
            conn.execute("SET LOCAL statement_timeout = '60s'")
            missing = [t for t in ("experience_shown", "experience_use")
                       if not conn.execute("SELECT to_regclass(%s) IS NOT NULL AS p", (t,)).fetchone()["p"]]
            kinds = {r["source_kind"]: int(r["n"]) for r in conn.execute(RATE_KINDS_SQL, params).fetchall()}
            uses, made, reads = [], {}, {}
            if "experience_use" not in missing:
                uses = [(r["event_id"], r["source_kind"], r["citing_organ"])
                        for r in conn.execute(RATE_USES_SQL, params).fetchall()]
                made = {r["citing_organ"]: int(r["n"]) for r in conn.execute(RATE_CITES_MADE_SQL, params).fetchall()}
            if not missing:
                reads = {r["reader_organ"]: (int(r["reads"]), int(r["reads_no_cite"]), int(r["reads_unclosed"]))
                         for r in conn.execute(NOW_READS_SQL, params).fetchall()}
    return build_report(kind_counts=kinds, uses=uses, cites_made=made, reads=reads, lo=lo, hi=hi,
                        registry=load_organs(), missing_tables=missing)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dsn", default=os.getenv("ORION_LIVE_DSN") or os.getenv("POSTGRES_URI"))
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if not args.dsn:
        print("set ORION_LIVE_DSN (or POSTGRES_URI, or --dsn); the report only reads", file=sys.stderr)
        return 2
    rep = read_live(args.dsn, args.hours)
    print(json.dumps(rep, indent=2) if args.json else render(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
