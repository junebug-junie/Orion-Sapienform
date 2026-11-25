#!/usr/bin/env python3
"""Exact-match eval: the heartbeat seat rollup equals the replay's grammar_events recompute.

Issue #3886. scripts/demon_network_replay.py reads ``heartbeat_seat_rollup_5m`` first and falls back
to ``grammar_events``. That only works if both give the same number for the same bucket. This eval
checks it, cell by cell (bucket x seat), with ``==`` on both the summed level and the atom count.

Two modes, both read-only against the live database (``docker exec orion-athena-sql-db psql``):

``--mode replay`` (works before deploy): pulls every routed-service atom of a 1-hour window from
``grammar_events`` with its event_id and emitted_at, and pushes each one through the real
``HeartbeatService._handle_grammar_message`` path (with a clock that reads the atom's emitted_at
plus a small receipt delay). The service's own ``SeatRollupAccumulator`` then flushes the buckets.
The other side is the replay's own query and ``aggregate_grammar_rows``.

``--mode table`` (after deploy): compares the rows orion-sql-writer stored in
``heartbeat_seat_rollup_5m`` for the window with the replay recompute. Partial rows are left out
(they are an undercount by design). A small mismatch here can be real: the two consumers read the
same pub/sub channel, but a sql-writer persist timeout drops a ledger row the heartbeat counted.

Exit 0 when every compared cell matches exactly and at least one cell was compared; 1 otherwise.

    /mnt/scripts/Orion-Sapienform/.venv/bin/python services/orion-heartbeat/evals/seat_rollup_match.py --mode replay
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[3]
SERVICE = REPO / "services" / "orion-heartbeat"
for p in (str(SERVICE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

BUCKET = 300
RECEIPT_DELAY_SEC = 0.05


def _load_replay():
    spec = importlib.util.spec_from_file_location("demon_network_replay", REPO / "scripts" / "demon_network_replay.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _window(end: str | None, hours: float) -> tuple[int, int]:
    if end:
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    else:
        end_dt = datetime.now(timezone.utc) - timedelta(minutes=15)
    e = int(end_dt.timestamp()) // BUCKET * BUCKET
    return e - int(hours * 3600) // BUCKET * BUCKET, e


def _where(start: int, end: int) -> str:
    return f" and emitted_at >= to_timestamp({start}) and emitted_at < to_timestamp({end})"


def recompute(replay, start: int, end: int) -> dict[tuple[int, int], tuple[float, int]]:
    agg, _stats = replay.aggregate_grammar_rows(replay._psql(replay.grammar_sql(_where(start, end))))
    return agg


def via_service(replay, start: int, end: int) -> tuple[dict[tuple[int, int], tuple[float, int]], dict]:
    """Feed the window's ledger atoms through the real service path; return the flushed rollup."""
    from app.seat_rollup import SeatRollupAccumulator
    from app.service import HeartbeatService
    from app.substrate.routing import ORGAN_SITE_MAP, SEAT_ORDER

    services = ",".join(repr(s) for s in ORGAN_SITE_MAP)
    rows = replay._psql(
        "select event_id, extract(epoch from emitted_at), source_service, event_json->'atom' "
        f"from grammar_events where event_kind = 'atom_emitted' and source_service in ({services})"
        + _where(start, end) + " order by emitted_at, event_id"
    )
    svc = HeartbeatService()
    # Started exactly on the window's first boundary, so the first bucket is complete.
    svc.seat_rollup = SeatRollupAccumulator(node="eval", seats=SEAT_ORDER, started_at_epoch=float(start))
    clock = {"now": float(start)}

    async def feed() -> None:
        for event_id, epoch, source_service, atom_json in rows:
            emitted = float(epoch)
            clock["now"] = max(clock["now"], emitted + RECEIPT_DELAY_SEC)
            payload = {
                "event_kind": "atom_emitted",
                "event_id": event_id,
                "emitted_at": datetime.fromtimestamp(emitted, timezone.utc).isoformat(),
                "provenance": {"source_service": source_service},
                "atom": json.loads(atom_json),
            }
            await svc._handle_grammar_message(payload)
            while not svc._absorb_queue.empty():  # the eval does not absorb; keep the queue empty
                svc._absorb_queue.get_nowait()
                svc._absorb_queue.task_done()

    with mock.patch("app.service.time.time", lambda: clock["now"]):
        asyncio.run(feed())
    events = svc.seat_rollup.flush_closed(float(end) + 3600.0)
    out: dict[tuple[int, int], tuple[float, int]] = {}
    partial = 0
    for ev in events:
        b = int(ev.bucket_start.timestamp()) // BUCKET
        if ev.partial:
            partial += 1
            continue
        if start // BUCKET <= b < end // BUCKET:
            out[(b, ev.site_index)] = (ev.level_sum, ev.atom_count)
    info = {"atoms_fed": len(rows), "partial_rows": partial, "rollup": svc._seat_rollup_stats()}
    return out, info


def from_table(replay, start: int, end: int) -> tuple[dict[tuple[int, int], tuple[float, int]], dict]:
    rows = replay._psql(
        "select floor(extract(epoch from bucket_start) / 300)::bigint, site_index, level_sum, atom_count, partial "
        f"from {replay.ROLLUP_TABLE} where bucket_start >= to_timestamp({start}) and bucket_start < to_timestamp({end})"
    )
    out, partial = {}, 0
    for b, site, level_sum, atom_count, p in rows:
        if p == "t":
            partial += 1
            continue
        out[(int(b), int(site))] = (float(level_sum), int(atom_count))
    return out, {"rows_read": len(rows), "partial_rows": partial}


def compare(rollup, ledger, start: int, end: int, n_seats: int) -> dict:
    """Every cell the rollup holds a complete row for, against the ledger (0, 0 when it has none)."""
    mismatches = []
    for (b, site), got in sorted(rollup.items()):
        want = ledger.get((b, site), (0.0, 0))
        if got != want:
            mismatches.append({"bucket": datetime.fromtimestamp(b * BUCKET, timezone.utc).isoformat(), "site": site,
                               "rollup": list(got), "grammar_events": list(want)})
    expected = (end - start) // BUCKET * n_seats
    return {"cells_compared": len(rollup), "cells_in_window": expected, "mismatches": len(mismatches),
            "first_mismatches": mismatches[:10],
            "ledger_cells_absent_from_rollup": sum(1 for k in ledger if k not in rollup)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("replay", "table"), default="replay")
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--end", help="window end (ISO, UTC); default now - 15 min, aligned to 5 min")
    args = ap.parse_args()
    replay = _load_replay()
    from app.substrate.routing import SEAT_ORDER

    start, end = _window(args.end, args.hours)
    t0 = time.time()
    ledger = recompute(replay, start, end)
    rollup, info = via_service(replay, start, end) if args.mode == "replay" else from_table(replay, start, end)
    result = {
        "mode": args.mode,
        "window": [datetime.fromtimestamp(start, timezone.utc).isoformat(), datetime.fromtimestamp(end, timezone.utc).isoformat()],
        **info,
        **compare(rollup, ledger, start, end, len(SEAT_ORDER)),
        "level_sum_by_seat": {
            seat: round(sum(v[0] for (b, s), v in rollup.items() if s == i), 6) for i, seat in enumerate(SEAT_ORDER)
        },
        "atom_count_by_seat": {
            seat: sum(v[1] for (b, s), v in rollup.items() if s == i) for i, seat in enumerate(SEAT_ORDER)
        },
        "runtime_s": round(time.time() - t0, 1),
    }
    print(json.dumps(result, indent=2))
    ok = result["cells_compared"] > 0 and result["mismatches"] == 0
    print("EXACT MATCH" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
