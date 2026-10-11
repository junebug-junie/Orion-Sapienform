#!/usr/bin/env python3
"""One-off repair: replace model-written times on world-pulse reading data
with the server's own write time (approved by Juniper 2026-10-11).

Before PR #2599 the Stage 1 handoff / Stage 2 result ``created_at`` came from
the model's JSON. That value became:

* ``observed_at`` on wp-read concept nodes in FalkorDB (graph
  ``orion_substrate``, node_id ``sub-concept-wp-read-*``) -- and, where the
  store back-filled the decay stamp from it, ``activation_decayed_at`` too;
* ``created_at`` on reading journal rows (``journal_entries`` and its copy
  ``journal_entry_index``, source_ref ``world_pulse_read[_stage2]:<trace>``).

The server time is ``world_pulse_read_seed.handoff_at`` (Stage 1, matched on
``trace_id``) and ``stage2_completed_at`` (Stage 2, matched on
``stage2_trace_id`` with ``stage2_status = 'done'``). Rows with no matching
server time are left alone and listed.

Subcommands (all write under --out, default /tmp/wp-read-timestamp-repair):

  plan      read-only; writes snapshot.json + plan summary
  apply     applies snapshot.json (guarded: only rows still holding the
            snapshotted old value change), logs progress.log
  verify    checks invariants against live data, writes verify.json
  rollback  restores old values from snapshot.json (guarded the same way)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

NODE_PREFIX = "sub-concept-wp-read-"
STAGE1_PREFIX = "world_pulse_read:"
STAGE2_PREFIX = "world_pulse_read_stage2:"
# A value at or before the server time and within this many seconds of it is
# already a server stamp (post-#2599 rows are stamped on receipt, a few
# seconds before the DB write) and is left alone. Anything LATER than the
# server time is always rewritten, however small the gap.
TOLERANCE_S = 120.0
SNAPSHOT_LIMIT_ROWS = 100_000
DECAY_STAMP = "activation_decayed_at"


def parse_ts(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def needs_change(current: datetime | None, server: datetime) -> bool:
    if current is None:
        return True
    delta = (current - server).total_seconds()
    return delta > 0 or delta < -TOLERANCE_S


@dataclass
class Change:
    store: str  # falkor.observed_at | falkor.activation_decayed_at | journal_entries | journal_entry_index
    key: str  # node_id or entry_id
    old: str | None
    new: str
    match_ref: str  # trace / seed the server time came from
    delta_s: float | None


def plan_node_changes(
    nodes: Iterable[dict[str, Any]], handoff_at_by_trace: dict[str, datetime]
) -> tuple[list[Change], list[dict[str, Any]]]:
    changes: list[Change] = []
    unmatched: list[dict[str, Any]] = []
    for node in nodes:
        trace = node.get("trace_id") or ""
        server = handoff_at_by_trace.get(trace)
        if server is None:
            unmatched.append(node)
            continue
        old_obs = parse_ts(node.get("observed_at"))
        if not needs_change(old_obs, server):
            continue
        new = server.isoformat()
        delta = (old_obs - server).total_seconds() if old_obs else None
        changes.append(Change("falkor.observed_at", node["node_id"], node.get("observed_at"), new, trace, delta))
        # The store fills the decay stamp from observed_at when no decay has
        # run yet; a stamp still equal to the bad observed_at (or ahead of the
        # wall clock) would keep decay frozen at the model's time.
        stamp_raw = node.get(DECAY_STAMP)
        stamp = parse_ts(stamp_raw)
        if stamp is not None and (stamp == old_obs or stamp > node["_now"]) and stamp > server:
            changes.append(Change(f"falkor.{DECAY_STAMP}", node["node_id"], stamp_raw, new, trace,
                                  (stamp - server).total_seconds()))
    return changes, unmatched


def plan_journal_changes(
    rows: Iterable[dict[str, Any]],
    *,
    table: str,
    handoff_at_by_trace: dict[str, datetime],
    stage2_at_by_trace: dict[str, datetime],
) -> tuple[list[Change], list[dict[str, Any]]]:
    changes: list[Change] = []
    unmatched: list[dict[str, Any]] = []
    for row in rows:
        ref = row["source_ref"]
        if ref.startswith(STAGE2_PREFIX):
            trace = ref[len(STAGE2_PREFIX):]
            server = stage2_at_by_trace.get(trace)
        elif ref.startswith(STAGE1_PREFIX):
            trace = ref[len(STAGE1_PREFIX):]
            server = handoff_at_by_trace.get(trace)
        else:
            continue
        if server is None:
            unmatched.append(row)
            continue
        old = parse_ts(row["created_at"])
        if not needs_change(old, server):
            continue
        changes.append(Change(table, row["entry_id"], old.isoformat() if old else None, server.isoformat(), ref,
                              (old - server).total_seconds() if old else None))
    return changes, unmatched


# ---------------------------------------------------------------- live I/O


class Progress:
    def __init__(self, path: Path, title: str, total: int) -> None:
        self.path, self.title, self.total = path, title, max(total, 1)
        self.start = time.monotonic()
        self.done = self.errors = 0
        self.anomalies: list[str] = []

    def tick(self, *, error: bool = False, anomaly: str | None = None) -> None:
        self.done += 1
        self.errors += int(error)
        if anomaly:
            self.anomalies.append(anomaly)
        elapsed = time.monotonic() - self.start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else 0.0
        line = (f"{datetime.now(timezone.utc).isoformat()} {self.title} "
                f"{100.0 * self.done / self.total:.1f}% eta={eta:.1f}s "
                f"{self.done}/{self.total} rate={rate:.1f}/s errors={self.errors} "
                f"anomalies={len(self.anomalies)}{(' last=' + anomaly) if anomaly else ''}")
        with self.path.open("a") as fh:
            fh.write(line + "\n")


def _connect(args):
    import psycopg2
    from orion.graph.falkor_client import RedisGraphQueryClient

    pg = psycopg2.connect(args.pg_dsn)
    falkor = RedisGraphQueryClient(uri=args.falkor_uri, graph_name=args.graph)
    return pg, falkor


def _server_times(cur) -> tuple[dict[str, datetime], dict[str, datetime]]:
    cur.execute("SELECT trace_id, handoff_at FROM world_pulse_read_seed "
                "WHERE trace_id IS NOT NULL AND handoff_at IS NOT NULL")
    s1 = {t: parse_ts(v) for t, v in cur.fetchall()}
    cur.execute("SELECT stage2_trace_id, stage2_completed_at FROM world_pulse_read_seed "
                "WHERE stage2_status = 'done' AND stage2_trace_id IS NOT NULL AND stage2_completed_at IS NOT NULL")
    s2 = {t: parse_ts(v) for t, v in cur.fetchall()}
    return s1, s2


def _fetch_nodes(falkor) -> list[dict[str, Any]]:
    return falkor.graph_query(
        "MATCH (n:SubstrateNode) WHERE n.node_id STARTS WITH $p RETURN n.node_id AS node_id, "
        "n.provenance_trace_id AS trace_id, n.observed_at AS observed_at, "
        f"n.{DECAY_STAMP} AS {DECAY_STAMP}, n.recency_score AS recency_score, n.activation AS activation",
        {"p": NODE_PREFIX},
    )


def _fetch_journal(cur, table: str) -> list[dict[str, Any]]:
    cur.execute(f"SELECT entry_id, source_ref, created_at FROM {table} "
                "WHERE source_ref LIKE 'world_pulse_read%%'")
    return [{"entry_id": e, "source_ref": s, "created_at": c} for e, s, c in cur.fetchall()]


def cmd_plan(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pg, falkor = _connect(args)
    cur = pg.cursor()
    s1, s2 = _server_times(cur)
    now = datetime.now(timezone.utc)
    nodes = _fetch_nodes(falkor)
    for n in nodes:
        n["_now"] = now
    node_changes, node_unmatched = plan_node_changes(nodes, s1)
    changes = list(node_changes)
    unmatched: dict[str, Any] = {"falkor": [{k: v for k, v in n.items() if k != "_now"} for n in node_unmatched]}
    journal_raw: dict[str, Any] = {}
    for table in ("journal_entries", "journal_entry_index"):
        rows = _fetch_journal(cur, table)
        journal_raw[table] = len(rows)
        jc, ju = plan_journal_changes(rows, table=table, handoff_at_by_trace=s1, stage2_at_by_trace=s2)
        changes += jc
        unmatched[table] = [{**r, "created_at": r["created_at"].isoformat()} for r in ju]
    if len(changes) > SNAPSHOT_LIMIT_ROWS:
        print(f"STOP: {len(changes)} changes exceeds {SNAPSHOT_LIMIT_ROWS}", file=sys.stderr)
        return 2
    snapshot = {
        "taken_at": now.isoformat(),
        "graph": args.graph,
        "scanned": {"falkor_nodes": len(nodes), **journal_raw},
        "changes": [asdict(c) for c in changes],
        "unmatched": unmatched,
        "full_nodes_before": [{k: v for k, v in n.items() if k != "_now"} for n in nodes],
    }
    (out / "snapshot.json").write_text(json.dumps(snapshot, indent=1, default=str))
    counts: dict[str, int] = {}
    for c in changes:
        counts[c.store] = counts.get(c.store, 0) + 1
    print(json.dumps({"scanned": snapshot["scanned"], "changes": counts,
                      "unmatched": {k: len(v) for k, v in unmatched.items()}}, indent=1))
    return 0


def _apply_one(cur, falkor, store: str, key: str, expect: str | None, value: str) -> bool:
    if store.startswith("falkor."):
        prop = store.split(".", 1)[1]
        rows = falkor.graph_query(
            f"MATCH (n:SubstrateNode {{node_id: $id}}) WHERE n.{prop} = $expect "
            f"SET n.{prop} = $value RETURN count(n) AS c",
            {"id": key, "expect": expect, "value": value},
        )
        return bool(rows and int(rows[0]["c"]) == 1)
    cur.execute(f"UPDATE {store} SET created_at = %s WHERE entry_id = %s AND created_at = %s",
                (value, key, expect))
    return cur.rowcount == 1


def _run(args, *, reverse: bool) -> int:
    out = Path(args.out)
    snap = json.loads((out / "snapshot.json").read_text())
    changes = snap["changes"]
    title = "wp-read-timestamp-repair:" + ("rollback" if reverse else "apply")
    prog = Progress(out / "progress.log", title, len(changes))
    pg, falkor = _connect(args)
    cur = pg.cursor()
    applied = 0
    for c in changes:
        expect, value = (c["new"], c["old"]) if reverse else (c["old"], c["new"])
        try:
            ok = _apply_one(cur, falkor, c["store"], c["key"], expect, value)
            applied += int(ok)
            prog.tick(anomaly=None if ok else f"{c['store']}:{c['key']}:value_changed_since_snapshot")
        except Exception as exc:  # noqa: BLE001 - logged and counted
            pg.rollback()
            prog.tick(error=True, anomaly=f"{c['store']}:{c['key']}:{type(exc).__name__}")
            continue
        pg.commit()
    print(json.dumps({"total": len(changes), "applied": applied, "errors": prog.errors,
                      "anomalies": prog.anomalies}, indent=1))
    return 0 if prog.errors == 0 else 1


def cmd_verify(args) -> int:
    out = Path(args.out)
    pg, falkor = _connect(args)
    cur = pg.cursor()
    s1, s2 = _server_times(cur)
    now = datetime.now(timezone.utc)
    nodes = _fetch_nodes(falkor)
    future = [n["node_id"] for n in nodes if parse_ts(n["observed_at"]) and parse_ts(n["observed_at"]) > now]
    later = [n["node_id"] for n in nodes
             if n["trace_id"] in s1 and parse_ts(n["observed_at"]) > s1[n["trace_id"]]]
    early = [n["node_id"] for n in nodes if n["trace_id"] in s1
             and (s1[n["trace_id"]] - parse_ts(n["observed_at"])).total_seconds() > TOLERANCE_S]
    stamp_future = [n["node_id"] for n in nodes if parse_ts(n.get(DECAY_STAMP)) and parse_ts(n[DECAY_STAMP]) > now]
    journal: dict[str, Any] = {}
    for table in ("journal_entries", "journal_entry_index"):
        rows = _fetch_journal(cur, table)
        jc, _ = plan_journal_changes(rows, table=table, handoff_at_by_trace=s1, stage2_at_by_trace=s2)
        journal[table] = {"rows": len(rows), "still_off": len(jc),
                          "future": sum(1 for r in rows if r["created_at"] > now)}
    result = {
        "checked_at": now.isoformat(), "falkor_nodes": len(nodes),
        "falkor_observed_at_in_future": len(future), "falkor_observed_after_write": len(later),
        "falkor_observed_more_than_tolerance_before_write": len(early),
        "falkor_decay_stamp_in_future": len(stamp_future),
        "journal": journal,
        "recency_sample": sorted(({"node_id": n["node_id"], "observed_at": n["observed_at"],
                                   "recency_score": n["recency_score"], "activation": n["activation"]}
                                  for n in nodes), key=lambda r: r["observed_at"], reverse=True)[:8],
    }
    (out / "verify.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result, indent=1, default=str))
    bad = (len(future) + len(later) + len(early) + len(stamp_future)
           + sum(j["still_off"] + j["future"] for j in journal.values()))
    return 0 if bad == 0 else 1


def cmd_report(args) -> int:
    out = Path(args.out)
    snap = json.loads((out / "snapshot.json").read_text())
    with (out / "before_after.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["store", "key", "old", "new", "delta_s_old_minus_server", "match_ref"])
        for c in snap["changes"]:
            w.writerow([c["store"], c["key"], c["old"], c["new"], c["delta_s"], c["match_ref"]])
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["plan", "apply", "verify", "rollback", "report"])
    p.add_argument("--out", default="/tmp/wp-read-timestamp-repair")
    p.add_argument("--pg-dsn", default="host=localhost port=55432 user=postgres password=postgres dbname=conjourney")
    p.add_argument("--falkor-uri", default="redis://localhost:6380")
    p.add_argument("--graph", default="orion_substrate")
    args = p.parse_args(argv)
    if args.command == "plan":
        return cmd_plan(args)
    if args.command == "apply":
        return _run(args, reverse=False)
    if args.command == "rollback":
        return _run(args, reverse=True)
    if args.command == "verify":
        return cmd_verify(args)
    return cmd_report(args)


if __name__ == "__main__":
    sys.exit(main())
