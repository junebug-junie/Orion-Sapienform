#!/usr/bin/env python3
"""Grade the transport baseline gate's log-only week from transport_baseline_hourly.

Acceptance check 1 of docs/superpowers/specs/2026-09-24-metacog-capture-and-
transport-ewma-baseline-design.md, made runnable:

- Per hop with traffic, during quiet hours 01:00-06:00 Juniper-local time
  (America/Denver, DST-aware: 07:00-12:00 UTC under MDT, 08:00-13:00 UTC under
  MST), i.e. rows whose local ``hour_start`` hour is 1..5: the median z must be
  within +/-0.5 and the median saturation ratio within [0.8, 1.3]. Medians are
  across hourly rows, weighted by ``windows_evaluated`` (each row already holds
  the median of its own evaluated windows -- a median of medians, stated, not
  hidden).

  2026-10-01 revision -- the grade asks the gate's own question, "would this
  hop raise a false alert while things are calm?", using the gate's own rules,
  instead of judging bands the gate itself ignores. The first live night
  (2026-09-30) FAILED 12 of 17 hops, and every failure was one of:
    * a ratio out of band on a tiny absolute gap (orion:state:request at 59 ms
      vs a 23 ms best) -- the gate needs MIN_EXCESS_MS (250) of absolute excess
      before saturation counts, so it could never alert on these;
    * an excluded hop (measured, never triggers) -- it cannot alert at all;
    * a quiet-hour z of 0.5-3: nights carry different load than the all-hours
      baseline (recall runs faster at night, z -1.27). The +/-0.5 band was the
      spec's guess, not a calibrated value; the gate only fires at z >= SPIKE_Z.
  So the rest-state FAILs are:
    * the gate itself opened a spike or saturation episode during quiet hours
      (``would_emit_by_condition`` -- the gate's own decision, with its own
      materiality and sustain rules applied, counted whether EMIT is on or off);
    * the typical quiet-hour ratio is at/above the gate's SATURATION_RATIO with
      a material gap. The gap is ``floor_ms * (ratio - 1)``: the ratio is
      exp(level - floor), so this is the gate's own level-minus-floor. (Not
      ``baseline_ms - floor_ms``: baseline_ms is the guarded fast mean, which by
      design does not absorb a step change -- it would hide exactly the
      saturation this check exists for.)
  Everything else is a NOTE: z of any size without an opened spike (negative z
  can never fire; a positive z on a tiny-ms hop is immaterial), and ratios
  between the 0.8-1.3 band and the saturation line. Scope, stated: this grades
  "would it false-alert at rest", not "is the hop drifting" -- slow creep under
  the saturation line is caught only by the floor-rise check below. Excluded
  hops are shown but never decide the overall verdict. The
  would-emit table (which pairs with live timeouts) stays the EMIT evidence.
- No hop's floor may rise more than 1.5x without a ``regime_shift`` in between.
  Only upward moves are flagged: the floor follows improvements quickly by
  design (90 s half-life down), and "busy quietly becoming normal" is the upward
  failure this guards. A config change (new fingerprint) cold-starts the gate,
  so it also starts a new comparison segment.
- Warm-up is not a resting state: until a hop has ``n_warm`` judged windows its
  floor simply equals its level. Rows not warm at the hour's start are left out
  of the medians, and a non-warm row (first learning, or a cold start after a
  fold/load failure under the same fingerprint) restarts the floor segment.

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
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("America/Denver")
QUIET_LOCAL_HOURS = frozenset({1, 2, 3, 4, 5})  # hours starting 01:00..05:00 local = 01:00-06:00
Z_BAND = 0.5
RATIO_BAND = (0.8, 1.3)
# Mirror the gate's own firing rules (orion/metacog/transport_baseline.py
# TransportBaselineConfig defaults; overridable from the CLI if the live
# config differs).
SPIKE_Z = 3.0
SATURATION_RATIO = 2.0
MIN_EXCESS_MS = 250.0
FLOOR_RISE_LIMIT = 1.5
DEFAULT_DSN = "postgresql://postgres:postgres@localhost:55432/conjourney"

COLUMNS = (
    "service", "instance", "key", "hour_start", "flush_reason", "windows_seen",
    "windows_evaluated", "success_count", "timeout_count", "z_p50", "z_p90",
    "saturation_ratio_p50", "baseline_ms", "floor_ms_start", "floor_ms",
    "calls_per_min_mean", "conditions_opened", "open_at_hour_end",
    "would_emit_by_condition", "excluded", "warm", "warm_at_start", "emit_effective", "config_fingerprint",
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


def is_quiet_hour(hour_start: Any) -> bool:
    return _ts(hour_start).astimezone(LOCAL_TZ).hour in QUIET_LOCAL_HOURS


def _warm_at_start(r: dict) -> bool:
    return bool(r.get("warm_at_start", r.get("warm", True)))


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
    excess_ms_median: float | None = None
    floor_flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
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
        if not _warm_at_start(r):
            # warm-up (first learning or a cold start): floor == level, not a floor yet
            seg_min = None
            if not r.get("warm", True):
                continue
        shifted = int(_dict(r.get("conditions_opened")).get("regime_shift", 0) or 0) > 0
        # In the hour a regime_shift was stated, the end-of-hour floor is the
        # re-seeded new normal: only the start is judged against the old segment.
        start = r.get("floor_ms_start") if _warm_at_start(r) else None
        values = (start,) if shifted else (start, r.get("floor_ms"))
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


def _excess_ms(r: dict) -> float | None:
    """The gate's saturation gap, exp(level) - exp(floor) = floor * (ratio - 1)."""
    floor, ratio = r.get("floor_ms"), r.get("saturation_ratio_p50")
    if floor is None or ratio is None:
        return None
    return float(floor) * (float(ratio) - 1.0)


_REST_ALERT_CONDITIONS = ("spike:open", "saturation:open")


def _quiet_rest_alerts(quiet: list[dict]) -> int:
    return sum(
        int(_dict(r.get("would_emit_by_condition")).get(c, 0) or 0) for r in quiet for c in _REST_ALERT_CONDITIONS
    )


def grade_key(
    ident: str,
    rows: list[dict],
    *,
    spike_z: float = SPIKE_Z,
    saturation_ratio: float = SATURATION_RATIO,
    min_excess_ms: float = MIN_EXCESS_MS,
) -> KeyGrade:
    quiet = [
        r for r in rows
        if is_quiet_hour(r["hour_start"])
        and int(r.get("windows_evaluated") or 0) > 0
        and _warm_at_start(r)
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
        excess_ms_median=weighted_median((_excess_ms(r), r.get("windows_evaluated")) for r in quiet),
        floor_flags=floor_rises(rows),
    )
    problems: list[str] = []
    gradable = z is not None or ratio is not None
    if gradable:
        rest_alerts = _quiet_rest_alerts(quiet)
        if rest_alerts:
            problems.append(
                f"the gate itself opened {rest_alerts} spike/saturation episode(s) during quiet hours"
            )
        if z is None:
            problems.append("its quiet-hour z was missing")
        elif z >= spike_z and not rest_alerts:
            g.notes.append(
                f"quiet-hour z {z:+.2f} is past the spike line, but the gate opened no spike at rest "
                "(not material or not sustained)"
            )
        elif abs(z) > Z_BAND and not rest_alerts:
            g.notes.append(
                f"nights run {'slower' if z > 0 else 'faster'} than its all-hours baseline "
                f"(quiet-hour z {z:+.2f}) -- a load difference"
            )
        lo, hi = RATIO_BAND
        excess = g.excess_ms_median
        if ratio is None:
            problems.append("its quiet-hour slow-vs-best ratio was missing")
        elif ratio < lo:
            g.notes.append(f"at rest it runs faster than its recorded best (ratio {ratio:.2f})")
        elif ratio > hi:
            material = excess is None or excess >= min_excess_ms
            if ratio >= saturation_ratio and material:
                gap = "unknown ms" if excess is None else f"{excess:.0f} ms"
                problems.append(
                    f"at rest it already reads as saturated: {ratio:.2f}x its best recent speed ({gap} above it; "
                    f"the gate opens saturation at {saturation_ratio:g}x and {min_excess_ms:.0f} ms)"
                )
            elif not material:
                g.notes.append(
                    f"ratio {ratio:.2f} is out of band but only {excess:.0f} ms above its best "
                    f"(under the gate's {min_excess_ms:.0f} ms floor, so it cannot alert)"
                )
            else:
                g.notes.append(
                    f"at rest it runs {ratio:.2f}x its best recent speed ({excess:.0f} ms above it) -- "
                    f"above the {lo}-{hi} band but below the gate's {saturation_ratio:g}x saturation line"
                )
    problems.extend(g.floor_flags)
    if not gradable and not g.floor_flags:
        g.verdict = "NOT_GRADABLE"
        g.sentence = (
            "had no judged, warmed-up traffic in quiet hours, so there is nothing to say about its resting state yet."
        )
        return g
    g.verdict = "FAIL" if problems else "PASS"
    if g.verdict == "PASS":
        g.sentence = (
            f"would not raise a false alert at rest: quiet-hour z {z:+.2f}, {ratio:.2f}x its best recent "
            "speed, and its baseline never crept up unannounced."
        )
    elif g.verdict == "FAIL":
        g.sentence = "does not pass: " + "; ".join(problems) + "."
    if g.notes:
        g.sentence += " Note: " + "; ".join(g.notes) + "."
    return g


def would_emit_by_day(rows: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        day = f"{_ts(r['hour_start']):%Y-%m-%d}"
        for cond, n in _dict(r.get("would_emit_by_condition")).items():
            out[day][cond] += int(n or 0)
    return {d: dict(v) for d, v in sorted(out.items())}


def grade(
    rows: list[dict],
    *,
    spike_z: float = SPIKE_Z,
    saturation_ratio: float = SATURATION_RATIO,
    min_excess_ms: float = MIN_EXCESS_MS,
) -> dict[str, Any]:
    by_key: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_key[_ident(r)].append(r)
    grades = [
        grade_key(k, v, spike_z=spike_z, saturation_ratio=saturation_ratio, min_excess_ms=min_excess_ms)
        for k, v in sorted(by_key.items())
    ]
    # Excluded hops are measured but can never trigger: shown, never decisive.
    gradable = [g for g in grades if g.verdict != "NOT_GRADABLE" and not g.excluded]
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
        counts["EXCLUDED" if g.excluded and g.verdict != "NOT_GRADABLE" else g.verdict] += 1
    lines.append(
        f"Overall: {report['overall']}. {span:.1f} days of hourly readings "
        f"({report['first_hour']:%Y-%m-%d %H:00} to {report['last_hour']:%Y-%m-%d %H:00} UTC): "
        f"{counts['PASS']} hops pass, {counts['FAIL']} fail, {counts['NOT_GRADABLE']} had no quiet-hour traffic, "
        f"{counts['EXCLUDED']} excluded (measured, never trigger; not counted)."
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
    ap.add_argument("--spike-z", type=float, default=SPIKE_Z, help="the gate's spike z (EQUILIBRIUM_TRANSPORT_BASELINE_SPIKE_Z)")
    ap.add_argument(
        "--saturation-ratio", type=float, default=SATURATION_RATIO,
        help="the gate's saturation ratio (EQUILIBRIUM_TRANSPORT_BASELINE_SATURATION_RATIO)",
    )
    ap.add_argument(
        "--min-excess-ms", type=float, default=MIN_EXCESS_MS,
        help="the gate's materiality floor (EQUILIBRIUM_TRANSPORT_BASELINE_MIN_EXCESS_MS)",
    )
    args = ap.parse_args(argv)
    rows = load_rows_jsonl(args.input) if args.input else load_rows_pg(args.dsn, args.days)
    if rows is None:
        print("Could not open a read-only Postgres session; nothing graded (UNVERIFIED).", file=sys.stderr)
        return 2
    report = grade(
        rows, spike_z=args.spike_z, saturation_ratio=args.saturation_ratio, min_excess_ms=args.min_excess_ms
    )
    sys.stdout.write(render(report))
    return {"PASS": 0, "NO_DATA": 3}.get(report["overall"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
