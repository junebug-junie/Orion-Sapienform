#!/usr/bin/env python3
"""Grade the transport baseline gate's log-only week from transport_baseline_hourly.

Acceptance check 1 of docs/superpowers/specs/2026-09-24-metacog-capture-and-
transport-ewma-baseline-design.md, made runnable:

- Per hop with traffic, during quiet hours 01:00-06:00 MDT (07:00-12:00 UTC,
  i.e. rows whose ``hour_start`` UTC hour is 7..11): the median z must be
  within +/-0.5 and the median saturation ratio within [0.8, 1.3]. Medians are
  across hourly rows, weighted by ``windows_evaluated`` (each row already holds
  the median of its own evaluated windows -- a median of medians, stated, not
  hidden).
- No hop's floor may rise more than 1.5x without a ``regime_shift`` in between.
  Only upward moves are flagged: the floor follows improvements quickly by
  design (90 s half-life down), and "busy quietly becoming normal" is the upward
  failure this guards. A config change (new fingerprint) cold-starts the gate,
  so it also starts a new comparison segment.

Also prints what the gate WOULD have published per condition per day (the
would-emit counts), which is what sizes the EMIT decision.

Read-only. Source: Postgres (``--dsn``, default ``POSTGRES_URI``) or a JSONL
file of rows (``--input``, one TransportBaselineHourlyV1-shaped object per line).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

QUIET_UTC_HOURS = frozenset({7, 8, 9, 10, 11})  # 01:00-06:00 MDT (UTC-6)
Z_BAND = 0.5
RATIO_BAND = (0.8, 1.3)
FLOOR_RISE_LIMIT = 1.5
DEFAULT_DSN = "postgresql://postgres:postgres@localhost:55432/conjourney"

COLUMNS = (
    "service", "instance", "key", "hour_start", "flush_reason", "windows_seen",
    "windows_evaluated", "success_count", "timeout_count", "z_p50", "z_p90",
    "saturation_ratio_p50", "baseline_ms", "floor_ms_start", "floor_ms",
    "calls_per_min_mean", "conditions_opened", "open_at_hour_end",
    "would_emit_by_condition", "excluded", "warm", "emit_effective", "config_fingerprint",
)


def _ts(v: Any) -> datetime:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _dict(v: Any) -> dict:
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return {}
    return v if isinstance(v, dict) else {}


def weighted_median(pairs: Iterable[tuple[float, int]]) -> float | None:
    items = sorted((float(v), int(w)) for v, w in pairs if v is not None and w and w > 0)
    total = sum(w for _, w in items)
    if total == 0:
        return None
    half, acc = total / 2.0, 0
    for i, (v, w) in enumerate(items):
        acc += w
        if acc > half:
            return v
        if acc == half:  # exactly between two values
            return (v + items[i + 1][0]) / 2.0 if i + 1 < len(items) else v
    return items[-1][0]


@dataclass
class KeyGrade:
    ident: str
    excluded: bool
    hours: int
    quiet_hours_with_traffic: int
    z_median: float | None
    ratio_median: float | None
    floor_flags: list[str] = field(default_factory=list)
    verdict: str = "NOT_GRADABLE"
    sentence: str = ""


def _ident(r: dict) -> str:
    return f"{r['service']}|{r.get('instance') or ''}|{r['key']}"


def floor_rises(rows: list[dict]) -> list[str]:
    """Upward floor moves > FLOOR_RISE_LIMIT inside a segment with no
    regime_shift. Segments restart at a regime_shift or a config change."""
    flags: list[str] = []
    seg_min: float | None = None
    fp: str | None = None
    for r in sorted(rows, key=lambda r: _ts(r["hour_start"])):
        if r.get("config_fingerprint") != fp:
            fp, seg_min = r.get("config_fingerprint"), None
        shifted = int(_dict(r.get("conditions_opened")).get("regime_shift", 0) or 0) > 0
        # In the hour a regime_shift was stated, the end-of-hour floor is the
        # re-seeded new normal: only the start is judged against the old segment.
        values = (r.get("floor_ms_start"),) if shifted else (r.get("floor_ms_start"), r.get("floor_ms"))
        for v in values:
            if v is None or v <= 0:
                continue
            if seg_min is not None and v / seg_min > FLOOR_RISE_LIMIT:
                flags.append(
                    f"floor rose {v / seg_min:.2f}x ({seg_min:.0f} -> {v:.0f} ms) by "
                    f"{_ts(r['hour_start']):%Y-%m-%d %H:00} UTC with no regime_shift"
                )
                seg_min = v  # report each further 1.5x step once, not every hour
            seg_min = v if seg_min is None else min(seg_min, v)
        if shifted:
            seg_min = r.get("floor_ms")  # re-seeded after the stated shift
    return flags


def grade_key(ident: str, rows: list[dict]) -> KeyGrade:
    quiet = [
        r for r in rows
        if _ts(r["hour_start"]).hour in QUIET_UTC_HOURS and int(r.get("windows_evaluated") or 0) > 0
    ]
    z = weighted_median((r.get("z_p50"), r.get("windows_evaluated")) for r in quiet)
    ratio = weighted_median((r.get("saturation_ratio_p50"), r.get("windows_evaluated")) for r in quiet)
    g = KeyGrade(
        ident=ident,
        excluded=any(bool(r.get("excluded")) for r in rows),
        hours=len({_ts(r["hour_start"]) for r in rows}),
        quiet_hours_with_traffic=len({_ts(r["hour_start"]) for r in quiet}),
        z_median=z,
        ratio_median=ratio,
        floor_flags=floor_rises(rows),
    )
    problems: list[str] = []
    gradable = z is not None or ratio is not None
    if gradable:
        if z is None or abs(z) > Z_BAND:
            problems.append(f"its typical quiet-hour z was {'missing' if z is None else f'{z:+.2f}'} (needs within +/-{Z_BAND})")
        lo, hi = RATIO_BAND
        if ratio is None or not (lo <= ratio <= hi):
            problems.append(
                f"its typical quiet-hour slow-vs-best ratio was {'missing' if ratio is None else f'{ratio:.2f}'} (needs {lo}-{hi})"
            )
    problems.extend(g.floor_flags)
    if not gradable and not g.floor_flags:
        g.verdict = "NOT_GRADABLE"
        g.sentence = "had no judged traffic in quiet hours, so there is nothing to say about its resting state yet."
        return g
    g.verdict = "FAIL" if problems else "PASS"
    if g.verdict == "PASS":
        g.sentence = (
            f"rests where it should: when things are quiet it reads z {z:+.2f} and runs at "
            f"{ratio:.2f}x its best recent speed, and its baseline never crept up unannounced."
        )
    elif g.verdict == "FAIL":
        g.sentence = "does not pass: " + "; ".join(problems) + "."
    return g


def would_emit_by_day(rows: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        day = f"{_ts(r['hour_start']):%Y-%m-%d}"
        for cond, n in _dict(r.get("would_emit_by_condition")).items():
            out[day][cond] += int(n or 0)
    return {d: dict(v) for d, v in sorted(out.items())}


def grade(rows: list[dict]) -> dict[str, Any]:
    by_key: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_key[_ident(r)].append(r)
    grades = [grade_key(k, v) for k, v in sorted(by_key.items())]
    gradable = [g for g in grades if g.verdict != "NOT_GRADABLE"]
    overall = "NO_DATA" if not gradable else ("PASS" if all(g.verdict == "PASS" for g in gradable) else "FAIL")
    hours = sorted({_ts(r["hour_start"]) for r in rows})
    return {
        "overall": overall,
        "keys": grades,
        "would_emit": would_emit_by_day(rows),
        "fingerprints": sorted({str(r.get("config_fingerprint")) for r in rows}),
        "first_hour": hours[0] if hours else None,
        "last_hour": hours[-1] if hours else None,
    }


def render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    if report["first_hour"] is None:
        return "No transport_baseline_hourly rows in range: the gate has not recorded anything yet (UNVERIFIED).\n"
    span = (report["last_hour"] - report["first_hour"]).total_seconds() / 86400.0 + 1 / 24
    counts = defaultdict(int)
    for g in report["keys"]:
        counts[g.verdict] += 1
    lines.append(
        f"Overall: {report['overall']}. {span:.1f} days of hourly readings "
        f"({report['first_hour']:%Y-%m-%d %H:00} to {report['last_hour']:%Y-%m-%d %H:00} UTC): "
        f"{counts['PASS']} hops pass, {counts['FAIL']} fail, {counts['NOT_GRADABLE']} had no quiet-hour traffic."
    )
    if span < 7:
        lines.append("The spec asks for a full week; this is less, so treat the verdict as provisional.")
    if len(report["fingerprints"]) > 1:
        lines.append(
            f"The gate's settings changed during this range ({len(report['fingerprints'])} config fingerprints), "
            "which restarts its learning each time."
        )
    lines.append("")
    lines.append("Per hop (service | instance | hop):")
    for g in sorted(report["keys"], key=lambda g: ({"FAIL": 0, "PASS": 1}.get(g.verdict, 2), g.ident)):
        tag = " [excluded: measured, never triggers]" if g.excluded else ""
        lines.append(
            f"- {g.verdict:<12} {g.ident}{tag}: {g.sentence} "
            f"({g.quiet_hours_with_traffic} quiet hours with traffic, {g.hours} hours seen)"
        )
    lines.append("")
    lines.append("What the gate would have published with EMIT on (condition:phase per day, excluded hops never count):")
    if not report["would_emit"]:
        lines.append("- nothing on any day.")
    for day, conds in report["would_emit"].items():
        total = sum(conds.values())
        detail = ", ".join(f"{k} {v}" for k, v in sorted(conds.items()))
        lines.append(f"- {day}: {total} rows ({detail})")
    lines.append("")
    lines.append("Would-emit counts are before the hourly publish budget (EQUILIBRIUM_TRANSPORT_BASELINE_MAX_TRIGGERS_PER_HOUR).")
    return "\n".join(lines) + "\n"


def load_rows_pg(dsn: str, days: int) -> list[dict] | None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _pg_readonly import open_readonly_connection

    conn = open_readonly_connection(dsn, connect_timeout=10, statement_timeout_ms=60000)
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {', '.join(COLUMNS)} FROM transport_baseline_hourly "
                "WHERE hour_start >= now() - (%s * interval '1 day') ORDER BY hour_start",
                (days,),
            )
            return [dict(zip(COLUMNS, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def load_rows_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--input", type=Path, default=None, help="JSONL rows instead of Postgres")
    ap.add_argument("--dsn", default=os.environ.get("POSTGRES_URI", DEFAULT_DSN))
    args = ap.parse_args(argv)
    rows = load_rows_jsonl(args.input) if args.input else load_rows_pg(args.dsn, args.days)
    if rows is None:
        print("Could not open a read-only Postgres session; nothing graded (UNVERIFIED).", file=sys.stderr)
        return 2
    report = grade(rows)
    sys.stdout.write(render(report))
    return {"PASS": 0, "NO_DATA": 3}.get(report["overall"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
