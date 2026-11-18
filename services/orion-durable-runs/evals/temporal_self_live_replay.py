"""Temporal Self patch 3 eval: the LIVE code path reproduces patch 2's arc-precision numbers.

The 10-09 fixture (``orion/temporal_self/evals/fixtures/day_2026-10-09.jsonl.gz``, a read-only,
text-free export of 10-08 and 10-09) is loaded into real-typed source tables in a throwaway schema.
Then the production ``Chronicler`` runs over it with a fake clock: the real ``SourceReader`` SQL,
the real late-row probe, the real body reads, and the real ``ChronicleStore`` transaction, one
window per step. Midway it is torn down and rebuilt from the store (a restart).

Gates (exit 1 if any fails):

1. The closed day 10-09 read back from ``temporal_self_day`` equals the pure one-pass fold of the
   same rows (``run_arc_precision_eval.run``) as canonical JSON (sorted keys: jsonb reorders object
   keys), with the live body summaries removed.
2. Patch 2's ``score`` on the persisted day passes: evidence precision 1.0, context purity,
   process recall, labels, sleep hypotheses.
3. Every non-reverie arc's live body summary equals the summary computed from the fixture's body
   rows (the eval's own rule), so the body SQL (TEXT timestamps included) reads what it should.
4. Nothing was skipped or counted late (every row was visible when its window was read).

Usage (never against production; it creates and drops its own schema):
    ORION_ADMISSION_TEST_DSN=postgresql://... python services/orion-durable-runs/evals/temporal_self_live_replay.py [--step-sec 600]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

HERE = Path(__file__).resolve().parent
SERVICE = HERE.parent
REPO = SERVICE.parents[1]
sys.path[:0] = [str(REPO), str(SERVICE), str(HERE), str(REPO / "orion/temporal_self/evals")]

from psycopg import AsyncConnection  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402
from psycopg_pool import AsyncConnectionPool  # noqa: E402

import run_arc_precision_eval as pure  # noqa: E402
import temporal_self_fixture_db as fdb  # noqa: E402
from app.temporal_self_chronicle import ChronicleConfig, Chronicler  # noqa: E402
from app.temporal_self_sources import SourceReader  # noqa: E402
from app.temporal_self_store import ChronicleStore  # noqa: E402
from orion.schemas.temporal_self import TemporalSelfDayV1  # noqa: E402
from orion.temporal_self import ReducerConfig  # noqa: E402
from orion.temporal_self.body import NO_BODY_KINDS, body_window, summarize_body  # noqa: E402
from orion.temporal_self.day import as_utc, day_window  # noqa: E402

DAY = pure.DAY


def _strip_bodies(day: TemporalSelfDayV1) -> TemporalSelfDayV1:
    d = day.model_copy(deep=True)
    for a in d.arcs:
        a.body = None
    return d


async def replay(dsn: str, rows, step_sec: float, restart_at: datetime, source_dsn: str | None = None) -> dict:
    schema = f"ts_replay_{uuid4().hex[:12]}"
    async with await AsyncConnection.connect(dsn, autocommit=True) as conn:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
    try:
        async with AsyncConnectionPool(dsn, min_size=1, max_size=4, open=False, kwargs={
                "autocommit": True, "prepare_threshold": 0, "row_factory": dict_row,
                "options": f"-c search_path={schema}"}) as pool:
            async with pool.connection() as conn:
                await fdb.create_schema(conn)
                loaded = await fdb.load(conn, rows) if source_dsn is None else {}
            # --source-dsn: read the real source tables (READ ONLY transactions), write here.
            src = AsyncConnectionPool(source_dsn, min_size=1, max_size=2, open=False, kwargs={
                "autocommit": True, "prepare_threshold": 0, "row_factory": dict_row}) if source_dsn else None
            if src is not None:
                await src.open()
            rc = ReducerConfig()
            origin = day_window("2026-10-08", rc.tz_name)[0]
            end = day_window(DAY, rc.tz_name)[1]
            cfg = ChronicleConfig(reducer=rc, read_lag_sec=300.0, backfill_days=0)
            clock = {"now": origin + timedelta(seconds=cfg.read_lag_sec)}

            def build() -> Chronicler:
                return Chronicler(store=ChronicleStore(pool), reader=SourceReader(src or pool, rc.tz_name), cfg=cfg,
                                  now=lambda: clock["now"])

            ch, restarted, steps, t0 = build(), False, 0, time.time()
            while ch.watermark is None or ch.watermark < end:
                out = await ch.step()
                steps += 1
                if out["error"]:
                    raise RuntimeError(f"step failed at {clock['now']}: {out['error']}")
                if not restarted and ch.watermark >= restart_at:
                    ch, restarted = build(), True      # restart: state comes back from temporal_self_state
                clock["now"] += timedelta(seconds=step_sec)
            store = ChronicleStore(pool)
            day_json = await store.day(DAY)
            async with pool.connection() as conn:
                late = (await (await conn.execute("SELECT count(*) AS n FROM temporal_self_event WHERE late_unfolded")).fetchone())["n"]
                events = (await (await conn.execute("SELECT count(*) AS n FROM temporal_self_event")).fetchone())["n"]
                arcs_n = (await (await conn.execute("SELECT count(*) AS n FROM temporal_self_arc WHERE day_id = %s", (DAY,))).fetchone())["n"]
                cursors = await store.cursors()
            if src is not None:
                await src.close()
            return {"day": TemporalSelfDayV1.model_validate(day_json) if day_json else None, "late": late,
                    "events": events, "arc_rows": arcs_n, "steps": steps, "seconds": round(time.time() - t0, 1),
                    "loaded": loaded, "cursors": len(cursors), "origin": origin, "origin_check": ch.health()["origin"]}
    finally:
        async with await AsyncConnection.connect(dsn, autocommit=True) as conn:
            await conn.execute(f'DROP SCHEMA "{schema}" CASCADE')


def _canon(model) -> str:
    return json.dumps(json.loads(model.model_dump_json()), sort_keys=True, separators=(",", ":"))


def json_diff(a, b, path: str = "", out=None, limit: int = 8) -> list[str]:
    """First few paths where two JSON values differ (lists of arcs are keyed by arc_id)."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                json_diff(a.get(k), b.get(k), f"{path}.{k}", out, limit)
    elif isinstance(a, list) and isinstance(b, list) and all(isinstance(x, dict) and "arc_id" in x for x in a + b):
        json_diff({x["arc_id"]: x for x in a}, {x["arc_id"]: x for x in b}, path, out, limit)
    elif a != b:
        out.append(f"{path}: live={json.dumps(a, default=str)[:160]} pure={json.dumps(b, default=str)[:160]}")
    return out


def expected_bodies(rows, day: TemporalSelfDayV1) -> dict:
    body_rows = {k: [dict(r, _t=as_utc(r.get("observed_at") or r.get("timestamp"))) for r in rows[k]]
                 for k in ("body_cluster", "body_cabinet", "body_spike")}
    out = {}
    for a in day.arcs:
        if a.kind in NO_BODY_KINDS:
            continue
        b0, b1 = body_window(a)
        pick = {k: [r for r in v if b0 <= r["_t"] <= b1] for k, v in body_rows.items()}
        out[a.arc_id] = summarize_body(pick["body_cluster"], pick["body_cabinet"], pick["body_spike"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.getenv("ORION_ADMISSION_TEST_DSN"))
    ap.add_argument("--step-sec", type=float, default=600.0, help="fake-clock step (live tick is 120 s)")
    ap.add_argument("--source-dsn", default=None,
                    help="read sources from this database instead of the fixture (READ ONLY transactions; "
                         "the chronicle's own writes still go to the disposable --dsn). Reports drift since the export.")
    args = ap.parse_args()
    if not args.dsn:
        print("ORION_ADMISSION_TEST_DSN (a disposable database) is required", file=sys.stderr)
        return 2
    rows = pure.load()
    ticks, events = pure.build_inputs(rows)
    rc = ReducerConfig()
    pure_day, _ = pure.run(rc, rows, ticks, events)
    restart_at = datetime(2026, 10, 9, 19, 3, tzinfo=timezone.utc)   # mid-afternoon, mid-arc
    live = asyncio.run(replay(args.dsn, rows, args.step_sec, restart_at, args.source_dsn))
    day = live["day"]
    if day is None:
        print(json.dumps({"passed": False, "error": "day 2026-10-09 was never closed"}))
        return 1

    # Canonical JSON: temporal_self_day.day_json is jsonb, which stores object keys in its own order,
    # so dict-valued fields (rows_by_lane, source_cursors, ...) come back reordered. Same content.
    identical = _canon(_strip_bodies(day)) == _canon(pure_day)
    report = pure.score(rc, rows, ticks, day, identical)
    want = expected_bodies(rows, day)
    outside = []
    if args.source_dsn:
        # The export holds body rows for [10-09 00:00 local, 10-10 00:00 local) only; the live
        # tables also hold the earlier ones, so arcs reaching before that cannot be compared.
        d0, d1 = day_window(DAY, rc.tz_name)
        outside = [a.arc_id for a in day.arcs if a.arc_id in want and not (d0 <= body_window(a)[0] and body_window(a)[1] < d1)]
        for k in outside:
            want.pop(k)
    body_mismatch = [a.arc_id for a in day.arcs if a.arc_id in want and a.body != want[a.arc_id]]
    body_missing = [a.arc_id for a in day.arcs if a.kind not in NO_BODY_KINDS and a.body is None]
    pure_report = pure.score(rc, rows, ticks, pure_day, True)
    same_numbers = all(report[k] == pure_report[k] for k in ("arcs_by_kind", "evidence", "process_recall", "returns",
                                                             "dwell_sec", "rest", "frame_at_close", "label_failures"))
    summary = {
        "step_sec": args.step_sec, "steps": live["steps"], "seconds": live["seconds"],
        "rows_loaded": sum(live["loaded"].values()), "events_stored": live["events"], "late_rows": live["late"],
        "arc_rows_10_09": live["arc_rows"], "day_arcs": len(day.arcs),
        "closed_day_equals_pure_one_pass": identical,
        "closed_day_diff": [] if identical else json_diff(json.loads(_strip_bodies(day).model_dump_json()),
                                                         json.loads(pure_day.model_dump_json())),
        "score_equals_patch2": same_numbers,
        "arcs_by_kind": report["arcs_by_kind"],
        "evidence": {k: report["evidence"][k] for k in ("total", "matched", "unresolved", "precision", "by_kind")},
        "process_recall": report["process_recall"],
        "returns": report["returns"],
        "rest": report["rest"],
        "label_failures": report["label_failures"],
        "bodies_checked": len(want), "body_mismatch": body_mismatch[:5], "body_missing": body_missing[:5],
        "bodies_outside_export_window": outside,
        "skipped_at_or_before_watermark": day.frame.skipped_at_or_before_watermark,
        "patch2_passed": report["passed"],
    }
    summary["source"] = "live database (read-only)" if args.source_dsn else "fixture"
    summary["passed"] = bool(report["passed"] and identical and same_numbers and not body_mismatch and not body_missing
                             and live["late"] == 0 and day.frame.skipped_at_or_before_watermark == 0
                             and live["arc_rows"] == len(day.arcs))
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
