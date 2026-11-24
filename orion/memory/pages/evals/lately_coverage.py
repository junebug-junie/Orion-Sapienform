"""page:lately coverage eval (issue #3463): does the page talk about last week's real events?

Reads a dry-run or folded page (``scripts/core_page_dry_run.py --page lately --out <json>``) and the
live ``episode_memory`` rows of the last ``--days`` days, through the same serving filter the fold
uses (``orion/memory/pages/lately.py`` EPISODE_SQL). Prints COUNTS ONLY, never row text, so the output
is safe to paste into a public PR:

- cited records, and how many are from the last ``--days`` days;
- days with memories vs days the page cites;
- episodes with memories vs episodes the page cites;
- high-stakes rows, open follow-ups and about_juniper rows: present vs cited.

``coverage`` is pure and unit-tested; ``main`` does the read-only DB pass.

    .venv/bin/python -m orion.memory.pages.evals.lately_coverage --page-json ~/.orion/core_pages/lately_dry_run.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[4]
ZONE = ZoneInfo("America/Denver")


def _local_day(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.astimezone(ZONE).date() if value.tzinfo else value.date()
    return value if isinstance(value, date) else None


def coverage(refs: Iterable[str], rows: Iterable[Mapping[str, Any]], *, since: date) -> dict[str, Any]:
    """``refs``: the page's stored refs (``episode_memory:<id>``, ...). ``rows``: served episode rows
    (EPISODE_SQL shape plus ``episode_id``). Counts only rows dated on or after ``since``."""
    refs = list(dict.fromkeys(refs))
    cited = {r.split(":", 1)[1] for r in refs if r.startswith("episode_memory:")}
    week = [r for r in rows if (_local_day(r.get("occurred_at")) or date.min) >= since]
    hit = [r for r in week if r["record_id"] in cited]

    def pair(pred) -> list[int]:
        return [sum(1 for r in hit if pred(r)), sum(1 for r in week if pred(r))]

    return {
        "cited_refs": len(refs),
        "cited_refs_from_window": len(hit) + sum(1 for r in refs if r.startswith("chat_history_log:")),
        "days_cited_of_days_with_memories": [len({_local_day(r.get("occurred_at")) for r in hit}),
                                             len({_local_day(r.get("occurred_at")) for r in week})],
        "episodes_cited_of_episodes": [len({r.get("episode_id") for r in hit if r.get("episode_id")}),
                                       len({r.get("episode_id") for r in week if r.get("episode_id")})],
        "high_stakes_cited_of_present": pair(lambda r: str(r.get("stakes") or "").lower() == "high"),
        "open_follow_ups_cited_of_present": pair(lambda r: r.get("purpose") == "follow_up"),
        "about_juniper_cited_of_present": pair(lambda r: r.get("purpose") == "about_juniper"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--page-json", required=True, help="dry-run JSON with 'refs'")
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args(argv)
    sys.path.insert(0, str(REPO))
    import psycopg
    from psycopg.rows import dict_row

    import importlib.util

    from orion.memory.pages.lately import EPISODE_SQL

    spec = importlib.util.spec_from_file_location("core_page_dry_run", REPO / "scripts" / "core_page_dry_run.py")
    dry_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dry_run)
    default_dsn = dry_run.default_dsn

    refs = json.loads(Path(args.page_json).expanduser().read_text())["refs"]
    sql = f"SELECT x.*, e.episode_id FROM ({EPISODE_SQL}) x JOIN episode_memory e ON e.memory_id::text = x.record_id"
    with psycopg.connect(default_dsn(), row_factory=dict_row) as conn:
        conn.read_only = True
        rows = [dict(r) for r in conn.execute(sql).fetchall()]
    since = datetime.now(ZONE).date() - timedelta(days=args.days - 1)
    print(json.dumps(coverage(refs, rows, since=since), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
