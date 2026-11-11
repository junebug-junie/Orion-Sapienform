#!/usr/bin/env python3
"""Back-fill reading acts' ENERGY and move the same watt-hours in the hourly attribution (patch 3e).

Patch 3a gave every read act (``hub.world_pulse_read``) its GPU slot-seconds, but no energy: the
hourly attribution (orion_energy_attribution_hourly, ``attribute_host_hour``) counted a read's
leases as "leased without an act", so charging the read too would have counted those Wh twice.
Patch 3e moves both sides together: the read act gets ``wh_by_host`` (and ``wh_gpu_by_host``,
``usd``, ``heat_c_by_host``) over its turns, and the hourly row counts those turns as metered. This
script does both for the past ``--days`` (default 30):

* ACTS: closed read acts with no ``basis["wh"]`` yet: the SAME late fill the writer runs live
  (orion/grammar/act_ledger.py ``late_cost_targets`` / ``late_read_cost`` / ``apply_late_cost``).
* HOURS: every stored hourly row of a host with pool cards, older than the hours the live writer
  still recomputes. Each hour is recomputed with the old rule (reads unmetered) and the new one.
  The old recompute must match the stored row's above-rest and sponsored energy (else the row is an
  anomaly and is left alone); then metered, leased-without-act and unattributed are written so the
  row still closes on its stored above-rest energy. Absolute values, so a second run changes nothing.

Each Wh is counted once, before and after: the per-act totals plus the hour's remainder
(sponsored + unattributed) equal the metered host energy above rest.

Backfill protocol (CLAUDE.md section 14):
1. Targets: the acts and hourly rows above.
2. Snapshot both (act cost_json; the hourly row) to ``<job-dir>/snapshot-<mode>-<time>.jsonl`` FIRST,
   a new file per run. Refuses above 100,000 rows (ask Juniper).
3. Progress: one line per batch to ``<job-dir>/progress.log``.
4. ``<job-dir>/report.md`` and ``<job-dir>/before_after.csv`` at the end.

DRY RUN is the default: the session is READ ONLY and every item is rolled back; nothing is written
to the database. ``--apply`` commits per act and per hour (both guarded: an act the live writer
filled meanwhile, and an hour it rewrote meanwhile, are skipped). ``--restore`` puts back the
earliest apply snapshot's values (run it only after reverting patch 3e, or the live writer
re-applies the new rule to the hours it still recomputes).

    python3 scripts/backfill_reading_act_energy.py              # dry run, 30 days
    python3 scripts/backfill_reading_act_energy.py --apply      # after patch 3e is deployed
    python3 scripts/backfill_reading_act_energy.py --restore    # undo from the snapshot
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, text  # noqa: E402

from orion.grammar import act_energy_fill as fill  # noqa: E402
from orion.grammar.act_ledger import (  # noqa: E402
    LATE_COST_BASIS_PREFIX,
    apply_late_cost,
    late_cost_targets,
    late_read_cost,
)

JOB = "reading-energy-backfill"
DEFAULT_URI = "postgresql://postgres:postgres@localhost:55432/conjourney"
MAX_ROWS = 100_000
PROGRESS_EVERY = 25
# The live writer recomputes the last 3 complete hours every tick (orion_energy_attribution.py
# RECOMPUTE_HOURS), plus the settle: those hours are its own, never this script's.
LIVE_HOURS = 4
# The stored row and the old-rule recompute must agree this closely on what the row measured.
DRIFT_TOL_WH = 0.01
ANOMALY_READ_WH = 500.0  # a read above this is flagged (not skipped): reads run ~10-60 Wh live

_HOURS_SQL = text(
    """
    SELECT hour_start, host, rest_w, rest_wh, above_rest_wh, sponsored_wh, metered_wh, leased_unmetered_wh,
           unattributed_wh, covered_sec, split_unknown_sec, updated_at
      FROM orion_energy_attribution_hourly
     WHERE hour_start >= :since AND hour_start < :until AND host = ANY(:hosts)
     ORDER BY hour_start, host
    """
)
_HOUR_UPDATE_SQL = text(
    """
    UPDATE orion_energy_attribution_hourly
       SET metered_wh = :metered_wh, leased_unmetered_wh = :leased_unmetered_wh,
           unattributed_wh = :unattributed_wh, updated_at = now()
     WHERE hour_start = :hour_start AND host = :host AND updated_at = :updated_at
    """
)
_HOUR_RESTORE_SQL = text(
    """
    UPDATE orion_energy_attribution_hourly
       SET metered_wh = :metered_wh, leased_unmetered_wh = :leased_unmetered_wh,
           unattributed_wh = :unattributed_wh, updated_at = now()
     WHERE hour_start = :hour_start AND host = :host
    """
)
_HOUR_FIELDS = ("rest_w", "rest_wh", "above_rest_wh", "sponsored_wh", "metered_wh", "leased_unmetered_wh",
                "unattributed_wh", "covered_sec", "split_unknown_sec")


def _iso(v: Any) -> Any:
    return v.isoformat() if isinstance(v, datetime) else v


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


class Progress:
    def __init__(self, path: Path, total: int, title: str) -> None:
        self.path, self.total, self.title = path, total, title
        self.t0 = time.monotonic()

    def line(self, done: int, errors: int, anomalies: int, note: str = "") -> str:
        el = time.monotonic() - self.t0
        rate = done / el if el > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else float("nan")
        pct = 100.0 * done / self.total if self.total else 100.0
        msg = (f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {self.title} {pct:5.1f}% "
               f"ETA {eta:6.0f}s rows {done}/{self.total} rate {rate:.2f}/s errors {errors} anomalies {anomalies}"
               + (f" {note}" if note else ""))
        with self.path.open("a") as fh:
            fh.write(msg + "\n")
        return msg


def pool_hosts() -> list[str]:
    """Hosts with a pool card: the only hosts whose hourly rows can have a read's lease."""
    locations, err = fill._card_locations()
    if err:
        raise SystemExit(f"gpu_pool.yaml unreadable ({err}): cannot tell which hosts carry leases")
    return sorted({h for h, _ in locations.values()})


def hour_targets(conn: Any, *, now: datetime, days: float) -> list[dict[str, Any]]:
    until = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=LIVE_HOURS)
    rows = conn.execute(_HOURS_SQL, {"since": now - timedelta(days=days), "until": until,
                                     "hosts": pool_hosts()}).mappings().all()
    return [dict(r) for r in rows]


def recompute_hour(conn: Any, row: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    """The hour under the old rule and the new one, and what to write (or why not)."""
    start = _aware(row["hour_start"])
    end = start + timedelta(hours=1)
    old, _, why_old = fill.attribute_host_hour(conn, row["host"], start, end, now=now, charge_reads=False)
    new, _, why_new = fill.attribute_host_hour(conn, row["host"], start, end, now=now, charge_reads=True)
    if old is None or new is None:
        return {"anomaly": f"no_rest: {why_old or why_new}"}
    drift = max(abs(old.above_rest_wh - float(row["above_rest_wh"])),
                abs(old.sponsored_wh - float(row["sponsored_wh"])),
                abs(old.metered_wh - float(row["metered_wh"])))
    if drift > DRIFT_TOL_WH:
        # The stored row was computed from other data or config than today's (late samples, a rest
        # change, an act that opened later): left as it is, said here.
        return {"anomaly": f"recompute_drift {drift:.3f} Wh", "old": old, "new": new}
    metered = new.metered_wh
    after = {"metered_wh": metered, "leased_unmetered_wh": new.leased_unmetered_wh,
             "unattributed_wh": float(row["above_rest_wh"]) - float(row["sponsored_wh"]) - metered}
    return {"anomaly": "", "old": old, "new": new, "after": after,
            "moved_wh": metered - float(row["metered_wh"])}


def run(conn: Any, *, days: float, apply: bool, job_dir: Path, now: datetime, limit: int = MAX_ROWS + 1,
        commit: Optional[Callable[[], Any]] = None, release: Optional[Callable[[], Any]] = None,
        out: Any = sys.stdout) -> dict[str, Any]:
    """The whole job on one connection. ``commit`` (apply) or ``release`` (dry run: roll back) runs
    after each item. Returns the summary also written to report.md."""
    job_dir.mkdir(parents=True, exist_ok=True)
    acts = late_cost_targets(conn, now=now, since=now - timedelta(days=days), limit=limit)
    hours = hour_targets(conn, now=now, days=days)
    total = len(acts) + len(hours)
    if total > MAX_ROWS:
        raise SystemExit(f"{total} target rows > {MAX_ROWS}: stop and ask Juniper (CLAUDE.md section 14)")
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with (job_dir / f"snapshot-{'apply' if apply else 'dry_run'}-{stamp}.jsonl").open("x") as fh:
        for t in acts:
            fh.write(json.dumps({"kind": "act", "act_id": t["act_id"], "closed_at": _iso(t["closed_at"]),
                                 "cost_json": t["cost_json"]}, default=str) + "\n")
        for h in hours:
            fh.write(json.dumps({"kind": "hour", **{k: _iso(v) for k, v in h.items()}}, default=str) + "\n")
    if release is not None:
        release()
    title = f"{JOB} ({'apply' if apply else 'dry run'})"
    progress = Progress(job_dir / "progress.log", total, title)
    print(progress.line(0, 0, 0, f"acts {len(acts)} hours {len(hours)}"), file=out)
    rows: list[dict[str, Any]] = []
    errors = anomalies = done = 0
    acts_written = acts_skipped = hours_written = hours_skipped = 0
    unmeasured_why: dict[str, int] = {}

    def _after_item() -> None:
        if apply and commit is not None:
            commit()
        elif not apply and release is not None:
            release()

    def _failed(kind: str, ident: str, exc: Exception) -> None:
        rows.append({"kind": kind, "id": ident, "error": f"{type(exc).__name__}: {exc}"[:300]})
        if hasattr(conn, "rollback"):
            conn.rollback()

    for t in acts:
        done += 1
        try:
            late = late_read_cost(conn, t, now=now)
            if apply:
                if apply_late_cost(conn, late, now=now):
                    acts_written += 1
                else:
                    acts_skipped += 1
            _after_item()
        except Exception as exc:  # noqa: BLE001 -- one bad act is counted, never stops the job
            errors += 1
            _failed("act", str(t["act_id"]), exc)
            continue
        wh = sum((late.wh_by_host or {}).values())
        basis_wh = str((late.cost.get("basis") or {}).get("wh") or "")
        anomaly = ("unmeasured" if basis_wh.startswith("unmeasured") else
                   "no_lease" if not late.wh_by_host else
                   "over_500wh" if wh > ANOMALY_READ_WH else "")
        anomalies += bool(anomaly)
        before = t["cost_json"] if isinstance(t["cost_json"], dict) else {}
        rows.append({"kind": "act", "id": late.act_id, "when": _iso(t["closed_at"]),
                     "before": json.dumps((before or {}).get("wh_by_host")),
                     "after": json.dumps(late.wh_by_host), "wh": round(wh, 4),
                     "by_host": late.wh_by_host or {}, "note": late.read_end, "anomaly": anomaly,
                     "basis_wh": basis_wh[:160], "error": ""})
        if anomaly == "unmeasured":
            unmeasured_why[basis_wh.split(" (")[0][:120]] = unmeasured_why.get(basis_wh.split(" (")[0][:120], 0) + 1
        if done % PROGRESS_EVERY == 0:
            print(progress.line(done, errors, anomalies, "acts"), file=out)

    for h in hours:
        done += 1
        ident = f"{h['host']}@{_iso(h['hour_start'])}"
        try:
            got = recompute_hour(conn, h, now=now)
            if apply and got.get("after"):
                res = conn.execute(_HOUR_UPDATE_SQL, {**got["after"], "hour_start": h["hour_start"],
                                                      "host": h["host"], "updated_at": h["updated_at"]})
                if getattr(res, "rowcount", 0):
                    hours_written += 1
                else:
                    hours_skipped += 1
            _after_item()
        except Exception as exc:  # noqa: BLE001
            errors += 1
            _failed("hour", ident, exc)
            continue
        anomalies += bool(got["anomaly"])
        after = got.get("after") or {}
        rows.append({"kind": "hour", "id": ident, "host": h["host"], "when": _iso(h["hour_start"]),
                     "before": json.dumps({k: round(float(h[k]), 4) for k in
                                           ("metered_wh", "leased_unmetered_wh", "unattributed_wh")}),
                     "after": json.dumps({k: round(v, 4) for k, v in after.items()}) if after else "",
                     "wh": round(got.get("moved_wh", 0.0), 4), "above_rest_wh": float(h["above_rest_wh"]),
                     "sponsored_wh": float(h["sponsored_wh"]), "after_raw": after,
                     "anomaly": got["anomaly"], "note": "", "error": ""})
        if done % PROGRESS_EVERY == 0:
            print(progress.line(done, errors, anomalies, "hours"), file=out)
    print(progress.line(done, errors, anomalies, "done"), file=out)

    fields = ["kind", "id", "when", "before", "after", "wh", "note", "anomaly", "error"]
    with (job_dir / "before_after.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    summary = summarize(rows, mode="apply" if apply else "dry_run", now=now, days=days, n_acts=len(acts),
                        n_hours=len(hours), errors=errors, anomalies=anomalies)
    summary.update(acts_unmeasured_by_reason=dict(sorted(unmeasured_why.items(), key=lambda kv: -kv[1])),
                   acts_written=acts_written, acts_skipped_filled_meanwhile=acts_skipped,
                   hours_written=hours_written, hours_skipped_rewritten_meanwhile=hours_skipped)
    _report(job_dir / "report.md", summary, rows)
    return summary


def summarize(rows: list[dict[str, Any]], *, mode: str, now: datetime, days: float, n_acts: int, n_hours: int,
              errors: int, anomalies: int) -> dict[str, Any]:
    acts = [r for r in rows if r["kind"] == "act" and not r.get("error")]
    hours = [r for r in rows if r["kind"] == "hour" and not r.get("error")]
    per = sorted(r["wh"] for r in acts if r["wh"] > 0)
    read_by_host: dict[str, float] = {}
    for r in acts:
        for host, wh in (r.get("by_host") or {}).items():
            read_by_host[host] = read_by_host.get(host, 0.0) + float(wh)
    moved_by_host: dict[str, float] = {}
    closure_err = 0.0
    for r in hours:
        if r.get("after_raw"):
            moved_by_host[r["host"]] = moved_by_host.get(r["host"], 0.0) + r["wh"]
            a = r["after_raw"]
            closure_err = max(closure_err, abs(r["above_rest_wh"] - r["sponsored_wh"] - a["metered_wh"]
                                               - a["unattributed_wh"]))
    read_wh, moved_wh = sum(read_by_host.values()), sum(moved_by_host.values())
    return {
        "mode": mode, "now": now.isoformat(), "days": days, "errors": errors, "anomalies": anomalies,
        "act_targets": n_acts, "acts_gain_energy": len(per),
        "acts_no_lease": sum(1 for r in acts if r["anomaly"] == "no_lease"),
        "acts_unmeasured": sum(1 for r in acts if r["anomaly"] == "unmeasured"),
        "read_wh_total": round(read_wh, 1), "read_wh_by_host": {k: round(v, 1) for k, v in sorted(read_by_host.items())},
        "per_read_wh": ({"median": round(statistics.median(per), 2), "p90": round(per[int(0.9 * (len(per) - 1))], 2),
                         "max": round(per[-1], 2)} if per else None),
        "hour_targets": n_hours, "hours_changed": sum(1 for r in hours if r.get("after_raw") and abs(r["wh"]) > 1e-9),
        "hours_drift": sum(1 for r in hours if str(r["anomaly"]).startswith("recompute_drift")),
        "wh_moved_hourly_to_reads": round(moved_wh, 1),
        "wh_moved_by_host": {k: round(v, 1) for k, v in sorted(moved_by_host.items())},
        # Read acts count a lease whole on its own window; the hourly rows clip at the window edge and
        # skip drifted and live hours, so the two totals agree closely, not to the Wh.
        "read_vs_moved_ratio": round(read_wh / moved_wh, 4) if moved_wh else None,
        "max_hour_closure_error_wh": round(closure_err, 9),
    }


def _report(path: Path, s: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    applied = s["mode"] == "apply"
    verdict = ("APPLIED" if applied else "DRY RUN (nothing written)") + (
        f"; {s['errors']} error(s)" if s["errors"] else "")
    act_ex = [r for r in rows if r["kind"] == "act" and not r.get("error")][:5]
    hour_ex = sorted((r for r in rows if r["kind"] == "hour" and r.get("after_raw")), key=lambda r: -abs(r["wh"]))[:5]
    lines = [
        f"# {JOB}: {s['mode']}", "",
        f"Verdict: {verdict}.", "",
        f"- Window: last {s['days']:g} days before {s['now']}.",
        f"- Read acts with no energy yet: {s['act_targets']}. Gain energy: {s['acts_gain_energy']}. "
        f"No lease: {s['acts_no_lease']}. Unmeasured: {s['acts_unmeasured']} "
        f"({s.get('acts_unmeasured_by_reason', {})}).",
        f"- Read energy filled: {s['read_wh_total']} Wh (by host {s['read_wh_by_host']}); per read {s['per_read_wh']}.",
        f"- Hourly rows examined: {s['hour_targets']}; changed: {s['hours_changed']}; left alone for drift: "
        f"{s['hours_drift']}.",
        f"- Wh moved in the hourly rows from leased-without-act to metered: {s['wh_moved_hourly_to_reads']} "
        f"(by host {s['wh_moved_by_host']}); read Wh / moved Wh = {s['read_vs_moved_ratio']}.",
        f"- Every changed hour still closes on its stored above-rest energy: max error "
        f"{s['max_hour_closure_error_wh']} Wh.",
        f"- Acts written: {s.get('acts_written', 0)} (skipped, filled meanwhile: "
        f"{s.get('acts_skipped_filled_meanwhile', 0)}); hours written: {s.get('hours_written', 0)} (skipped, "
        f"rewritten meanwhile: {s.get('hours_skipped_rewritten_meanwhile', 0)}).",
        f"- Errors: {s['errors']}. Anomalies (no lease, unmeasured, > {ANOMALY_READ_WH:g} Wh, recompute drift): "
        f"{s['anomalies']}.",
        f"- Needs another pass: {'yes (errors)' if s['errors'] else 'no'}"
        + ("" if applied else "; run with --apply after patch 3e is deployed") + ".", "",
        "## Read acts: before / after (wh_by_host)", "",
        "| act_id | before | after | read end |", "|---|---|---|---|",
    ]
    lines += [f"| {r['id']} | {r['before']} | {r['after']} | {r['note']} |" for r in act_ex]
    lines += ["", "## Hourly rows: largest moves (metered / leased_unmetered / unattributed Wh)", "",
              "| host@hour | before | after | moved Wh |", "|---|---|---|---|"]
    lines += [f"| {r['id']} | {r['before']} | {r['after']} | {r['wh']} |" for r in hour_ex]
    lines += ["", "Files: snapshot-<mode>-<time>.jsonl, progress.log, before_after.csv, report.md (this file).", ""]
    path.write_text("\n".join(lines))


def restore(conn: Any, job_dir: Path, *, commit: Optional[Callable[[], Any]] = None) -> dict[str, int]:
    """Put back, from the EARLIEST apply snapshot holding each row: read acts' cost_json (only acts
    carrying a "wh" basis line) and the hourly rows' metered / leased-without-act / unattributed."""
    first: dict[tuple[str, str], dict[str, Any]] = {}
    for snap in sorted(job_dir.glob("snapshot-apply-*.jsonl")):
        for line in snap.read_text().splitlines():
            row = json.loads(line)
            key = ("act", row["act_id"]) if row.get("kind") == "act" else ("hour", f"{row['host']}@{row['hour_start']}")
            first.setdefault(key, row)
    n = {"acts": 0, "hours": 0}
    for (kind, _), row in first.items():
        if kind == "act":
            res = conn.execute(text(
                "UPDATE grammar_acts SET cost_json = CAST(:c AS jsonb), updated_at = now() "
                "WHERE act_id = :a AND cost_json->'basis' ? 'wh' AND cost_json->'basis'->>'gpu' LIKE :p"),
                {"c": json.dumps(row["cost_json"]) if row["cost_json"] is not None else None,
                 "a": row["act_id"], "p": LATE_COST_BASIS_PREFIX + "%"})
            n["acts"] += res.rowcount or 0
        else:
            res = conn.execute(_HOUR_RESTORE_SQL, {
                "hour_start": datetime.fromisoformat(row["hour_start"]), "host": row["host"],
                "metered_wh": row["metered_wh"], "leased_unmetered_wh": row["leased_unmetered_wh"],
                "unattributed_wh": row["unattributed_wh"]})
            n["hours"] += res.rowcount or 0
        if commit is not None:
            commit()
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--apply", action="store_true", help="write (default: dry run, nothing written)")
    ap.add_argument("--restore", action="store_true", help="restore from the job's earliest apply snapshots")
    ap.add_argument("--job-dir", type=Path, default=Path("/tmp") / JOB)
    ap.add_argument("--uri", default=os.environ.get("ORION_ANALYSIS_PG_URI", DEFAULT_URI))
    args = ap.parse_args(argv)
    engine = create_engine(args.uri)
    now = datetime.now(timezone.utc)
    with engine.connect() as conn:
        if args.restore:
            got = restore(conn, args.job_dir, commit=conn.commit)
            print(f"restored {got} from {args.job_dir}/snapshot-apply-*.jsonl")
            return 0
        if not args.apply:
            # Every transaction on this session is READ ONLY: a write anywhere below fails loudly.
            conn.execute(text("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY"))
            conn.commit()
        summary = run(conn, days=args.days, apply=args.apply, job_dir=args.job_dir, now=now,
                      commit=conn.commit if args.apply else None,
                      release=None if args.apply else conn.rollback)
        if not args.apply:
            conn.rollback()
    print(json.dumps(summary, indent=2))
    print(f"report: {args.job_dir}/report.md  before/after: {args.job_dir}/before_after.csv")
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
