"""World-pulse source-quality eval: are Orion's reading seeds fresh, readable articles?

Two lanes, both read-only:

  --replay N   Last N real digests from live Postgres (world_pulse_read_seed,
               the seeds Orion's reader actually received). Per section: seeds,
               distinct URLs, and "new readable" URLs = first time seen in the
               window, not a listing/data page, not on a site that blocks the
               reader. This is the BEFORE baseline.
  --live       Today's digest built twice through the real pipeline (network,
               dry_run=True, no SQL/bus/curiosity): once with the origin/main
               sources config and no freshness window (old behaviour), once with
               this branch's config + window. Per section: seeds, and how many
               are readable AND published within the window.

Usage (repo root):
  PYTHONPATH=$PWD:$PWD/services/orion-world-pulse \
    .venv/bin/python services/orion-world-pulse/evals/run_source_quality_eval.py \
    --replay 10 --live --old-config /tmp/old_sources.yaml

Exit 1 when the --live AFTER run has a section with seeds but zero fresh
readable ones, or fewer fresh readable seeds in total than BEFORE.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import os

REPO = Path(__file__).resolve().parents[3]
# Code under test: this tree by default; an extracted origin/main tree for the
# BEFORE live run (set by main() for its subprocess).
CODE_ROOT = Path(os.environ.get("WP_EVAL_CODE_ROOT") or REPO)
SERVICE = CODE_ROOT / "services" / "orion-world-pulse"
for p in (str(SERVICE), str(CODE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from orion.world_pulse_read.seeds import seeds_from_digest_payload  # noqa: E402
from orion.world_pulse_read.url_filters import url_looks_like_section_index  # noqa: E402

# Sites whose refusal of the reading harness is recorded with live evidence in
# PR #2634 (orion/world_pulse_read/reader_blocks.py once merged). Mirrored here
# so this eval runs on main before that lands; prefer the module when present.
_FALLBACK_BLOCKED = ("bbc.co.uk", "bbc.com", "bbci.co.uk", "npr.org")


def reader_blocked(url: str) -> bool:
    try:
        from orion.world_pulse_read.reader_blocks import reader_block_reason  # type: ignore

        return bool(reader_block_reason(url))
    except Exception:
        host = (urlparse(url).hostname or "").lower()
        return any(host == d or host.endswith("." + d) for d in _FALLBACK_BLOCKED)


def readable(url: str) -> bool:
    return not url_looks_like_section_index(url) and not reader_blocked(url)


# --- replay (BEFORE, from live seeds) --------------------------------------


def replay(n: int) -> dict:
    sql = f"""
    WITH runs AS (
      SELECT run_id, created_at FROM world_pulse_run
      WHERE lower(dry_run) = 'false' ORDER BY created_at DESC LIMIT {int(n)}
    )
    SELECT json_agg(json_build_object('run_id', s.run_id, 'at', r.created_at,
                    'section', s.section, 'url', s.url, 'kind', s.kind) ORDER BY r.created_at)
    FROM world_pulse_read_seed s JOIN runs r USING (run_id)
    WHERE s.kind IN ('digest_item', 'finding')
    """
    out = subprocess.run(
        ["docker", "exec", "orion-athena-sql-db", "psql", "-U", "postgres", "-d", "conjourney", "-Atc", sql],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    rows = json.loads(out or "[]") or []
    seen: set[str] = set()
    per = defaultdict(lambda: {"seeds": 0, "distinct": set(), "new_readable": 0, "hosts": defaultdict(int)})
    for row in rows:
        sec = row["section"] or "?"
        url = row["url"]
        d = per[sec]
        d["seeds"] += 1
        d["distinct"].add(url)
        d["hosts"][urlparse(url).hostname or "?"] += 1
        if url not in seen and readable(url):
            d["new_readable"] += 1
        seen.add(url)
    runs = len({r["run_id"] for r in rows})
    return {
        "runs": runs,
        "stuck_replay": _replay_stuck_detector(rows),
        "sections": {
            s: {
                "seeds": d["seeds"],
                "distinct_urls": len(d["distinct"]),
                "new_readable": d["new_readable"],
                "hosts": dict(d["hosts"]),
            }
            for s, d in sorted(per.items())
        },
    }


def _replay_stuck_detector(rows: list[dict]) -> dict[str, int | None]:
    """Run the real detector over the replayed digests in order, one pseudo
    source per (section, host) -- seeds carry no source_id. Returns, per pseudo
    source, the 1-based digest index where it was first flagged stuck (None =
    never)."""
    from app.services.source_health import assess_source_health
    from app.settings import settings

    k = int(getattr(settings, "world_pulse_stuck_source_digests", 3))
    order: list[str] = []
    by_run: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["run_id"] not in by_run:
            order.append(row["run_id"])
        key = f"{row['section']}@{urlparse(row['url']).hostname}"
        by_run[row["run_id"]][key].append(row["url"])
    keys = sorted({k2 for r in by_run.values() for k2 in r})
    previous = None
    first_stuck: dict[str, int | None] = {k2: None for k2 in keys}
    for idx, run_id in enumerate(order, start=1):
        current = {k2: {"urls": by_run[run_id].get(k2, []), "fetched": 1} for k2 in keys}
        health = assess_source_health(current, previous, stuck_after=k)
        for k2, entry in health.items():
            if entry["stuck"] and first_stuck[k2] is None:
                first_stuck[k2] = idx
        previous = health
    return first_stuck


# --- live (today, old vs new) ----------------------------------------------


def live_digest(config_path: str, max_age_days: float) -> dict:
    from app.services import pipeline
    from app.settings import settings

    settings.world_pulse_sources_config_path = config_path
    if hasattr(settings, "world_pulse_max_item_age_days"):
        settings.world_pulse_max_item_age_days = max_age_days
    if hasattr(settings, "world_pulse_source_health_db_url"):
        settings.world_pulse_source_health_db_url = ""
    settings.world_pulse_fetch_enabled = True
    result = pipeline.run_world_pulse(requested_by="test", dry_run=True)
    now = datetime.now(timezone.utc)
    window = timedelta(days=3)
    articles = result.publish_status.get("articles") or []
    url_by_id = {a["article_id"]: a["url"] for a in articles}
    published = {}
    for a in articles:
        p = a.get("published_at")
        published[a["url"]] = datetime.fromisoformat(p.replace("Z", "+00:00")) if p else None
    payload = result.digest.model_dump(mode="json") if result.digest else {"items": []}
    payload["run_id"] = result.run.run_id
    seeds = seeds_from_digest_payload(payload, article_urls=url_by_id)
    per = defaultdict(lambda: {"seeds": 0, "fresh_readable": 0, "picks": []})
    for seed in seeds:
        d = per[seed.section or "?"]
        d["seeds"] += 1
        pub = published.get(seed.url)
        fresh = pub is not None and now - pub <= window
        ok = fresh and readable(seed.url)
        d["fresh_readable"] += int(ok)
        d["picks"].append(
            {"url": seed.url, "published": pub.isoformat() if pub else None, "fresh": fresh, "readable": readable(seed.url)}
        )
    return {
        "status": result.run.status,
        "warnings": [w for w in result.run.warnings if w.startswith(("source_stuck", "optional_source_fetch_error"))],
        "stale_dropped": result.run.metrics.get("stale_items_dropped"),
        "undated_dropped": result.run.metrics.get("undated_items_dropped"),
        "sections": dict(sorted(per.items())),
    }


def _live_subprocess(code_root: Path, config_path: str) -> dict:
    env = dict(os.environ, WP_EVAL_CODE_ROOT=str(code_root), PYTHONPATH=f"{code_root}/services/orion-world-pulse:{code_root}")
    out = subprocess.run(
        [sys.executable, __file__, "--_live-one", config_path],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(code_root),
        check=True,
    ).stdout.strip().splitlines()[-1]
    return json.loads(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=int, default=0)
    ap.add_argument("--live", action="store_true")
    ap.add_argument(
        "--old-root",
        default="",
        help="extracted origin/main tree (git archive origin/main orion services/orion-world-pulse config | tar -x -C DIR)",
    )
    ap.add_argument("--_live-one", dest="live_one", default="", help=argparse.SUPPRESS)
    ap.add_argument("--json", default="", help="write full report here")
    args = ap.parse_args()
    if args.live_one:
        print(json.dumps(live_digest(args.live_one, 3), default=str))
        return 0
    report: dict = {}
    failed = False
    if args.replay:
        report["replay_before"] = replay(args.replay)
        print(f"== BEFORE (replay of last {report['replay_before']['runs']} real digests, live seeds)")
        for s, d in report["replay_before"]["sections"].items():
            print(f"  {s:34s} seeds={d['seeds']:3d} distinct={d['distinct_urls']:3d} new_readable={d['new_readable']:3d}  {d['hosts']}")
        print("  stuck-source detector replayed over these digests (first digest flagged, of the window):")
        for key, idx in report["replay_before"]["stuck_replay"].items():
            print(f"    {key:60s} {'stuck from digest ' + str(idx) if idx else 'never stuck'}")
    if args.live:
        new_cfg = str(REPO / "config" / "world_pulse" / "sources.yaml")
        if args.old_root:
            old_cfg = str(Path(args.old_root) / "config" / "world_pulse" / "sources.yaml")
            report["live_old"] = _live_subprocess(Path(args.old_root), old_cfg)
        report["live_new"] = _live_subprocess(REPO, new_cfg)
        for key in ("live_old", "live_new"):
            if key not in report:
                continue
            r = report[key]
            print(f"== {key} status={r['status']} stale_dropped={r['stale_dropped']} undated_dropped={r['undated_dropped']}")
            for s, d in r["sections"].items():
                print(f"  {s:34s} seeds={d['seeds']} fresh_readable={d['fresh_readable']}")
                for p in d["picks"]:
                    print(f"      {'OK ' if p['fresh'] and p['readable'] else 'BAD'} {p['published']} {p['url']}")
            if r["warnings"]:
                print("  warnings:", r["warnings"])
        new = report["live_new"]["sections"]
        if any(d["seeds"] and not d["fresh_readable"] for d in new.values()):
            failed = True
        if "live_old" in report:
            old_total = sum(d["fresh_readable"] for d in report["live_old"]["sections"].values())
            new_total = sum(d["fresh_readable"] for d in new.values())
            print(f"== fresh readable seeds today: before={old_total} after={new_total}")
            if new_total < old_total:
                failed = True
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
