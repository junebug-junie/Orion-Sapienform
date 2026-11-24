#!/usr/bin/env python3
"""Replay past Prime now reads with today's ranking (#3783). Read-only.

For each stored ``experience_shown`` row of the ``prime`` organ, this reads the same candidate rows the endpoint
reads (the store's own SQL), builds the now again with ``orion.temporal_self.now.build_now`` and prints what was
shown then next to what would be shown now. ``--held`` gives the refs Prime held at the time (a concern's
event_ref, ``arc:<id>``, ``prime_concern:<id>``); without it, the replay shows the fair ranking alone.

It never writes: it runs SELECTs through ``docker exec <db container> psql`` (no experience_shown row, unlike a GET).

    python3 scripts/replay_prime_now.py --shown 8151e652a25044969bd71fcf0737edae \\
        --held world_pulse_read_seed:reading:50753c3f-d556-5a6f-bfa7-b02b95254905 --held prime_concern:520a5a9f8d084495
    python3 scripts/replay_prime_now.py --last 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "services" / "orion-durable-runs")]

from app.temporal_self_store import NOW_ARC_SUBJECTS_SQL, NOW_CANDIDATES_SQL, NOW_HELD_SQL  # noqa: E402
from orion.integration.organs import load_organs  # noqa: E402
from orion.schemas.temporal_self import NOW_MAX_PER_ORGAN  # noqa: E402
from orion.temporal_self.now import ARC_REF_PREFIX, build_now, held_lookup  # noqa: E402

HELD_LOOKBACK = timedelta(hours=24)  # services/orion-durable-runs/app/main.py NOW_HELD_LOOKBACK_HOURS


def _lit(v: Any) -> str:
    if isinstance(v, datetime):
        return f"'{v.isoformat()}'::timestamptz"
    if isinstance(v, (list, tuple)):
        return "ARRAY[" + ",".join(_lit(x) for x in v) + "]::text[]" if v else "ARRAY[]::text[]"
    if isinstance(v, int):
        return str(v)
    return "'" + str(v).replace("'", "''") + "'"


def query(container: str, sql: str, params: dict[str, Any]) -> list[dict]:
    for k, v in params.items():
        sql = sql.replace(f"%({k})s", _lit(v))
    wrapped = f"SELECT coalesce(json_agg(t), '[]'::json) FROM ({sql}) t"
    out = subprocess.run(["docker", "exec", "-i", container, "psql", "-U", "postgres", "-d", "conjourney", "-Atq",
                          "-v", "ON_ERROR_STOP=1", "-c", "SET default_transaction_read_only = on",
                          "-c", wrapped], check=True, capture_output=True, text=True).stdout
    return json.loads(out.strip())


def _rows(raw: list[dict]) -> list[dict]:
    out = []
    for r in raw:
        r = dict(r)
        r["occurred_at"] = datetime.fromisoformat(r["occurred_at"])
        r["payload"] = r.pop("payload_json", None) or {}
        r["related_refs"] = r.get("related_refs") or []
        if "kind_total" in r:
            r["kind_total"] = int(r["kind_total"])
        out.append(r)
    return out


def replay(container: str, shown: dict, held: list[str]) -> dict:
    reg = load_organs()
    lo, as_of = datetime.fromisoformat(shown["since"]), datetime.fromisoformat(shown["shown_at"])
    own = reg.kinds_of("prime")
    rows = _rows(query(container, NOW_CANDIDATES_SQL,
                       {"since": lo, "as_of": as_of, "own": sorted(own), "per_kind": NOW_MAX_PER_ORGAN}))
    arc_ids = [h[len(ARC_REF_PREFIX):] for h in held if h.startswith(ARC_REF_PREFIX)]
    arcs = {r["arc_id"]: r["subject_ref"] for r in query(container, NOW_ARC_SUBJECTS_SQL, {"ids": arc_ids})} \
        if arc_ids else {}
    if held:
        rows += _rows(query(container, NOW_HELD_SQL, {"keys": sorted(held_lookup(held, arcs)),
                                                       "lo": min(lo, as_of - HELD_LOOKBACK), "as_of": as_of,
                                                       "own": sorted(own), "limit": 50}))
    before_ids = list(shown["event_ids"])
    by_id = {r["event_id"]: r for r in rows}
    missing = [i for i in before_ids if i not in by_id]
    if missing:
        for r in _rows(query(container, "SELECT event_id, source_kind, occurred_at, label, verdict, related_refs, "
                                        "payload_json, salience, surprise_nats FROM temporal_self_event "
                                        "WHERE event_id = ANY(%(ids)s)", {"ids": missing})):
            by_id[r["event_id"]] = r
    after = build_now(rows, "prime", since=lo, as_of=as_of, shown_id="replay", registry=reg, held=held,
                      arc_subjects=arcs)
    return {"shown_id": shown["shown_id"], "since": shown["since"], "shown_at": shown["shown_at"],
            "candidates": len(rows), "held": held,
            "before": [(i, by_id.get(i, {}).get("source_kind"), by_id.get(i, {}).get("salience")) for i in before_ids],
            "after": [(i.event_id, i.source_kind, i.salience, i.held_refs) for i in after.items]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shown", action="append", default=[], help="experience_shown ids to replay")
    ap.add_argument("--last", type=int, default=0, help="replay the last N prime reads")
    ap.add_argument("--held", action="append", default=[], help="a ref Prime held (repeatable)")
    ap.add_argument("--container", default="orion-athena-sql-db")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    where = ("shown_id = ANY(%(ids)s)" if a.shown else "TRUE")
    shown = query(a.container, "SELECT shown_id, since, shown_at, event_ids FROM experience_shown WHERE "
                  f"reader_organ = 'prime' AND {where} ORDER BY shown_at DESC LIMIT %(n)s",
                  {"ids": a.shown, "n": max(a.last, len(a.shown), 1)})
    for s in shown:
        r = replay(a.container, s, a.held)
        if a.json:
            print(json.dumps(r, default=str))
            continue
        print(f"\n== {r['shown_id']} since {r['since']} shown {r['shown_at']} ({r['candidates']} candidate rows)")
        print("  BEFORE (what Prime was shown):")
        for i, k, sal in r["before"]:
            print(f"    {k or '?':<24} sal={'None' if sal is None else f'{sal:.2f}':<5} {i}")
        print("  AFTER (this ranking):")
        for i, k, sal, h in r["after"]:
            print(f"    {k:<24} sal={'None' if sal is None else f'{sal:.2f}':<5} {i}" + (f"  HELD {h}" if h else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
