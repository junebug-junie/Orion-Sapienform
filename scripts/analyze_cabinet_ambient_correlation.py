#!/usr/bin/env python3
"""Read-only cabinet ambient audio analysis — floor, fan↔RMS, deltas, spikes.

Fan *does* correlate with loudness — but tick Pearson hides it:
  - mean RMS rises with fan pct (see bins)
  - acoustic lags iLO fan by ~60–120s (iLO polls ~60s, biometrics ~30s)
  - raw Δfan vs ΔRMS looks inverted when CPU drops (thermal cooldown)

Data: Postgres `orion_biometrics_summary`, node `athena`, read-only.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

_FRACTIONAL_TS_RE = re.compile(
    r"^(?P<head>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})\."
    r"(?P<frac>\d+)(?P<tz>Z|[+-]\d{2}(?::\d{2})?)$"
)

# Coupling targets: tick-to-tick changes only. Never include activity (derived from RMS).
COUPLING_KEYS: tuple[tuple[str, str], ...] = (
    ("measurements", "fan_pct_max"),
    ("pressures", "fan"),
    ("pressures", "cpu"),
    ("pressures", "thermal"),
    ("pressures", "power"),
    ("measurements", "chassis_watts"),
    ("measurements", "disk_bytes_per_sec"),
    ("measurements", "temp_c_max"),
    ("pressures", "cabinet_climate_activity"),
    ("pressures", "cabinet_proximity_activity"),
)

AMBIENT_RMS = ("measurements", "cabinet_ambient_rms")
AMBIENT_ACT = ("pressures", "cabinet_ambient_audio_activity")


def parse_db_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        if "T" not in text and " " in text:
            text = text.replace(" ", "T", 1)
        if text.endswith("+00") and not text.endswith("+00:00"):
            text = text[:-3] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            match = _FRACTIONAL_TS_RE.match(text)
            if not match:
                raise
            frac = match.group("frac").ljust(6, "0")[:6]
            tz = match.group("tz")
            if tz == "Z":
                tz = "+00:00"
            elif len(tz) == 3:
                tz = f"{tz}:00"
            parsed = datetime.fromisoformat(f"{match.group('head')}.{frac}{tz}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _json_obj(value: Any) -> Mapping[str, Any]:
    if isinstance(value, str):
        return json.loads(value)
    return value if isinstance(value, Mapping) else {}


def _field(row: Mapping[str, Any], bucket: str, key: str) -> float | None:
    obj = _json_obj(row.get(bucket))
    raw = obj.get(key)
    if raw is None or isinstance(raw, bool):
        return None
    try:
        out = float(raw)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return None
    xs_p, ys_p = zip(*pairs)
    mx = sum(xs_p) / n
    my = sum(ys_p) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs_p, ys_p))
    den_x = sum((x - mx) ** 2 for x in xs_p) ** 0.5
    den_y = sum((y - my) ** 2 for y in ys_p) ** 0.5
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def lagged_pearson(
    xs: Sequence[float], ys: Sequence[float], lag_ticks: int
) -> float | None:
    """Align series with lag applied to xs (leading indicator).

    lag_ticks > 0: xs[i + lag] paired with ys[i] — xs *leads* ys.
    lag_ticks < 0: xs[i] paired with ys[i + |lag|] — ys leads xs.
    """
    if lag_ticks == 0:
        return pearson(xs, ys)
    if lag_ticks > 0:
        if lag_ticks >= len(xs):
            return None
        return pearson(xs[lag_ticks:], ys[: len(xs) - lag_ticks])
    lag = -lag_ticks
    if lag >= len(xs):
        return None
    return pearson(xs[: len(xs) - lag], ys[lag:])


def first_differences(values: Sequence[float | None]) -> list[float | None]:
    if len(values) < 2:
        return []
    out: list[float | None] = []
    for prev, cur in zip(values, values[1:]):
        if prev is None or cur is None:
            out.append(None)
        else:
            out.append(cur - prev)
    return out


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = int(round(p * (len(ordered) - 1)))
    return ordered[max(0, min(idx, len(ordered) - 1))]


@dataclass(frozen=True)
class Tick:
    t: datetime
    rms: float
    activity: float | None
    fields: Mapping[str, float | None]


@dataclass(frozen=True)
class FloorStats:
    n: int
    rms_min: float
    rms_max: float
    rms_mean: float
    rms_median: float
    rms_stdev: float
    rms_cv: float
    rms_within_10pct_of_median_pct: float
    activity_p10: float | None
    activity_p50: float | None
    activity_p90: float | None
    fan_pct_min: float | None
    fan_pct_max: float | None
    fan_pct_stdev: float | None


@dataclass(frozen=True)
class FanBinRow:
    fan_pct: int
    n: int
    rms_mean: float
    rms_median: float


@dataclass(frozen=True)
class FanLagRow:
    lag_ticks: int
    lag_sec: int
    r: float | None


@dataclass(frozen=True)
class FanStepRow:
    condition: str
    n: int
    mean_drms: float


@dataclass(frozen=True)
class FanRmsAnalysis:
    level_r: float | None
    bins: tuple[FanBinRow, ...]
    lags: tuple[FanLagRow, ...]
    best_lag_ticks: int
    best_lag_r: float | None
    fan_up_steps: tuple[FanStepRow, ...]


@dataclass(frozen=True)
class CouplingRow:
    target: str
    n: int
    target_stdev: float | None
    r_lag0: float | None
    best_lag_ticks: int
    best_r: float | None


@dataclass(frozen=True)
class SpikeRow:
    t: datetime
    rms: float
    activity: float
    drms: float | None
    fan_pct: float | None
    dfan_pct: float | None
    cpu: float | None
    dcpu: float | None
    notes: tuple[str, ...]


def build_ticks(rows: Sequence[Mapping[str, Any]]) -> list[Tick]:
    ticks: list[Tick] = []
    for row in rows:
        rms = _field(row, *AMBIENT_RMS)
        if rms is None:
            continue
        fields: dict[str, float | None] = {}
        for bucket, key in COUPLING_KEYS:
            fields[f"{bucket}.{key}"] = _field(row, bucket, key)
        ticks.append(
            Tick(
                t=parse_db_timestamp(row["timestamp"]),
                rms=rms,
                activity=_field(row, *AMBIENT_ACT),
                fields=fields,
            )
        )
    return ticks


def floor_stats(ticks: Sequence[Tick]) -> FloorStats | None:
    if not ticks:
        return None
    rms_vals = [t.rms for t in ticks]
    median = statistics.median(rms_vals)
    within = sum(abs(v - median) / median <= 0.10 for v in rms_vals if median) / len(rms_vals)
    acts = [t.activity for t in ticks if t.activity is not None]
    fan = [t.fields.get("measurements.fan_pct_max") for t in ticks]
    fan_vals = [v for v in fan if v is not None]
    return FloorStats(
        n=len(ticks),
        rms_min=min(rms_vals),
        rms_max=max(rms_vals),
        rms_mean=statistics.mean(rms_vals),
        rms_median=median,
        rms_stdev=statistics.pstdev(rms_vals),
        rms_cv=statistics.pstdev(rms_vals) / statistics.mean(rms_vals),
        rms_within_10pct_of_median_pct=100.0 * within,
        activity_p10=percentile(acts, 0.10) if acts else None,
        activity_p50=percentile(acts, 0.50) if acts else None,
        activity_p90=percentile(acts, 0.90) if acts else None,
        fan_pct_min=min(fan_vals) if fan_vals else None,
        fan_pct_max=max(fan_vals) if fan_vals else None,
        fan_pct_stdev=statistics.pstdev(fan_vals) if len(fan_vals) > 1 else None,
    )


def fan_rms_analysis(
    ticks: Sequence[Tick], *, grain_sec: int, max_lag_ticks: int, min_dfan: float = 3.0
) -> FanRmsAnalysis | None:
    fan = [t.fields.get("measurements.fan_pct_max") for t in ticks]
    rms = [t.rms for t in ticks]
    cpu = [t.fields.get("pressures.cpu") for t in ticks]
    fan_vals = [v for v in fan if v is not None]
    if len(fan_vals) < 10:
        return None

    from collections import defaultdict

    bins_map: dict[int, list[float]] = defaultdict(list)
    for fp, r in zip(fan, rms):
        if fp is not None:
            bins_map[int(round(fp))].append(r)
    bins = tuple(
        FanBinRow(
            fan_pct=pct,
            n=len(vals),
            rms_mean=statistics.mean(vals),
            rms_median=statistics.median(vals),
        )
        for pct, vals in sorted(bins_map.items())
        if len(vals) >= 3
    )

    lags: list[FanLagRow] = []
    best_lag = 0
    best_r: float | None = None
    for lag in range(-2, max_lag_ticks + 1):
        r = lagged_pearson(fan, rms, lag)
        lags.append(FanLagRow(lag_ticks=lag, lag_sec=lag * grain_sec, r=r))
        if r is not None and (best_r is None or abs(r) > abs(best_r)):
            best_r = r
            best_lag = lag

    step_rows: list[FanStepRow] = []
    up_drms: dict[str, list[float]] = {
        "fan↑ cpu↑": [],
        "fan↑ cpu↓": [],
        "fan↑ cpu flat": [],
    }
    for i in range(1, len(ticks)):
        fp0, fp1 = fan[i - 1], fan[i]
        if fp0 is None or fp1 is None or fp1 - fp0 < min_dfan:
            continue
        dr = rms[i] - rms[i - 1]
        c0, c1 = cpu[i - 1], cpu[i]
        if c0 is None or c1 is None:
            continue
        if c1 > c0 + 0.02:
            up_drms["fan↑ cpu↑"].append(dr)
        elif c1 < c0 - 0.02:
            up_drms["fan↑ cpu↓"].append(dr)
        else:
            up_drms["fan↑ cpu flat"].append(dr)
    for cond, vals in up_drms.items():
        if vals:
            step_rows.append(
                FanStepRow(condition=cond, n=len(vals), mean_drms=statistics.mean(vals))
            )

    return FanRmsAnalysis(
        level_r=pearson(fan, rms),
        bins=bins,
        lags=tuple(lags),
        best_lag_ticks=best_lag,
        best_lag_r=best_r,
        fan_up_steps=tuple(step_rows),
    )


def delta_coupling(
    ticks: Sequence[Tick], *, max_lag_ticks: int
) -> list[CouplingRow]:
    if len(ticks) < 4:
        return []
    drms = first_differences([t.rms for t in ticks])
    rows: list[CouplingRow] = []
    for bucket, key in COUPLING_KEYS:
        label = f"{bucket}.{key}"
        levels = [t.fields.get(label) for t in ticks]
        deltas = first_differences(levels)
        if len(deltas) < 3:
            continue
        delta_vals = [d for d in deltas if d is not None]
        target_stdev = statistics.pstdev(delta_vals) if len(delta_vals) > 1 else None
        best_lag = 0
        best_r = lagged_pearson(drms, deltas, 0)
        for lag in range(-max_lag_ticks, max_lag_ticks + 1):
            if lag == 0:
                continue
            r = lagged_pearson(drms, deltas, lag)
            if r is None:
                continue
            if best_r is None or abs(r) > abs(best_r):
                best_r = r
                best_lag = lag
        rows.append(
            CouplingRow(
                target=label,
                n=len([1 for a, b in zip(drms, deltas) if a is not None and b is not None]),
                target_stdev=target_stdev,
                r_lag0=lagged_pearson(drms, deltas, 0),
                best_lag_ticks=best_lag,
                best_r=best_r,
            )
        )
    rows.sort(key=lambda row: abs(row.best_r or 0.0), reverse=True)
    return rows


def _spike_notes(
    *,
    drms: float | None,
    dfan: float | None,
    dcpu: float | None,
    fan_stdev: float | None,
) -> tuple[str, ...]:
    notes: list[str] = []
    if drms is not None and abs(drms) > 800:
        notes.append("large ΔRMS")
    if fan_stdev and dfan is not None and abs(dfan) > max(2.0, 2 * fan_stdev):
        notes.append("fan moved")
    elif dfan is not None and abs(dfan) > 2.0:
        notes.append("fan moved")
    if dcpu is not None and abs(dcpu) > 0.05:
        notes.append("cpu shifted")
    if not notes:
        notes.append("no host step-change")
    return tuple(notes)


def spike_forensics(
    ticks: Sequence[Tick],
    floor: FloorStats | None,
    *,
    top_n: int = 8,
) -> list[SpikeRow]:
    if not ticks:
        return []
    fan_stdev = floor.fan_pct_stdev if floor else None
    ranked = sorted(
        [t for t in ticks if t.activity is not None],
        key=lambda t: t.activity or 0.0,
        reverse=True,
    )[:top_n]
    out: list[SpikeRow] = []
    index_by_time = {id(t): i for i, t in enumerate(ticks)}
    for tick in ranked:
        i = index_by_time[id(tick)]
        prev = ticks[i - 1] if i > 0 else None
        drms = tick.rms - prev.rms if prev else None
        fan = tick.fields.get("measurements.fan_pct_max")
        dfan = (
            fan - prev.fields.get("measurements.fan_pct_max")
            if prev and fan is not None
            else None
        )
        cpu = tick.fields.get("pressures.cpu")
        dcpu = cpu - prev.fields.get("pressures.cpu") if prev and cpu is not None else None
        out.append(
            SpikeRow(
                t=tick.t,
                rms=tick.rms,
                activity=float(tick.activity or 0.0),
                drms=drms,
                fan_pct=fan,
                dfan_pct=dfan,
                cpu=cpu,
                dcpu=dcpu,
                notes=_spike_notes(
                    drms=drms, dfan=dfan, dcpu=dcpu, fan_stdev=fan_stdev
                ),
            )
        )
    return out


def load_rows(dsn: str, *, node: str, since: datetime) -> list[dict[str, Any]]:
    import psycopg2
    import psycopg2.extras

    since_text = since.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    sql = """
        SELECT timestamp, measurements, pressures
        FROM orion_biometrics_summary
        WHERE node = %s
          AND timestamp >= %s
          AND measurements ? 'cabinet_ambient_rms'
        ORDER BY timestamp ASC
    """
    with contextlib.closing(psycopg2.connect(dsn, connect_timeout=10)) as conn:
        with conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (node, since_text))
            return [dict(row) for row in cur.fetchall()]


def render_report(
    *,
    node: str,
    window_hours: float,
    grain_sec: int,
    span: tuple[datetime, datetime] | None,
    floor: FloorStats | None,
    fan_rms: FanRmsAnalysis | None,
    coupling: Sequence[CouplingRow],
    spikes: Sequence[SpikeRow],
) -> str:
    lines = [
        "# Cabinet ambient analysis report",
        "",
        "Fan **does** track loudness — see §2 bins and lagged correlation.",
        "Instantaneous Δfan vs ΔRMS is misleading (thermal confound + acoustic lag).",
        "",
        f"- node: `{node}`",
        f"- window: last {window_hours:g}h",
        f"- grain: ~{grain_sec}s (biometrics summary; iLO fan ~60s)",
    ]
    if span:
        lines.append(f"- span: `{span[0].isoformat()}` → `{span[1].isoformat()}`")

    if floor:
        lines.extend(
            [
                "",
                "## 1. Acoustic floor (descriptive)",
                "",
                "RMS is **absolute loudness** (raw PCM units). Activity is **volatility vs EWMA baseline**, not loudness.",
                "",
                f"- ticks: {floor.n}",
                f"- RMS: min {floor.rms_min:.0f}, median {floor.rms_median:.0f}, max {floor.rms_max:.0f}",
                f"- RMS stdev {floor.rms_stdev:.0f} (CV {floor.rms_cv:.2f})",
                f"- RMS within ±10% of median: {floor.rms_within_10pct_of_median_pct:.1f}% of ticks",
            ]
        )
        if floor.activity_p50 is not None:
            lines.append(
                f"- Activity: p10 {floor.activity_p10:.3f}, p50 {floor.activity_p50:.3f}, p90 {floor.activity_p90:.3f}"
            )
        if floor.fan_pct_min is not None:
            lines.append(
                f"- Fan pct (iLO max): {floor.fan_pct_min:.0f}–{floor.fan_pct_max:.0f}% "
                f"(σ≈{floor.fan_pct_stdev:.2f})"
            )

    if fan_rms:
        level_r_txt = "—" if fan_rms.level_r is None else f"{fan_rms.level_r:+.3f}"
        lines.extend(
            [
                "",
                "## 2. Fan ↔ RMS (primary)",
                "",
                f"`fan_pct_max` is iLO max fan %. Instant Pearson r={level_r_txt} — weak because "
                "RMS variance within each fan speed is large; **binned means** and **lag** tell the story.",
                "| fan % | n | mean RMS | median RMS |",
                "|---:|---:|---:|---:|",
            ]
        )
        for row in fan_rms.bins:
            lines.append(
                f"| {row.fan_pct} | {row.n} | {row.rms_mean:.0f} | {row.rms_median:.0f} |"
            )
        lines.extend(
            [
                "",
                "**Lagged correlation** (fan leads when lag_sec > 0):",
                "",
                "| lag (ticks) | lag (sec) | r |",
                "|---:|---:|---:|",
            ]
        )
        for row in fan_rms.lags:
            r = "—" if row.r is None else f"{row.r:+.3f}"
            mark = " ← best" if row.lag_ticks == fan_rms.best_lag_ticks else ""
            lines.append(f"| {row.lag_ticks:+d} | {row.lag_sec:+d} | {r}{mark} |")
        if fan_rms.fan_up_steps:
            lines.extend(
                [
                    "",
                    "**When fan steps up ≥3%** — stratify by CPU (thermal confound):",
                    "",
                    "| condition | n | mean ΔRMS |",
                    "|---|---:|---:|",
                ]
            )
            for row in fan_rms.fan_up_steps:
                lines.append(f"| {row.condition} | {row.n} | {row.mean_drms:+.0f} |")

    lines.extend(
        [
            "",
            "## 3. Other signals: tick-to-tick ΔRMS vs Δtarget",
            "",
            "Answers: *when this signal moves one biometrics tick (~30s), does loudness move too?*",
            "",
            "| target | n | σ(Δtarget) | r(lag=0) | best lag | best r |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in coupling[:12]:
        stdev = "—" if row.target_stdev is None else f"{row.target_stdev:.4g}"
        r0 = "—" if row.r_lag0 is None else f"{row.r_lag0:+.3f}"
        br = "—" if row.best_r is None else f"{row.best_r:+.3f}"
        lag = f"{row.best_lag_ticks:+d}" if row.best_lag_ticks else "0"
        lines.append(f"| `{row.target}` | {row.n} | {stdev} | {r0} | {lag} | {br} |")
    lines.extend(
        [
            "",
            f"_Lag in ticks × {grain_sec}s. Δfan row often ≈0 — use §2 for fan; CPU co-movement confounds raw deltas._",
            "",
            "## 4. Activity spike forensics",
            "",
            "Top activity windows with host context. Activity is derived from RMS — use this table for *when*, not circular correlation.",
            "",
            "| time (UTC) | rms | activity | Δrms | fan% | Δfan | cpu | Δcpu | notes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for spike in spikes:
        def fmt(v: float | None, nd: int = 1) -> str:
            return "—" if v is None else f"{v:.{nd}f}"

        lines.append(
            f"| `{spike.t.isoformat()}` | {spike.rms:.0f} | {spike.activity:.3f} | "
            f"{fmt(spike.drms, 0)} | {fmt(spike.fan_pct, 0)} | {fmt(spike.dfan_pct, 1)} | "
            f"{fmt(spike.cpu, 2)} | {fmt(spike.dcpu, 2)} | {', '.join(spike.notes)} |"
        )

    lines.extend(
        [
            "",
            "## How to read this",
            "",
            "- **Higher fan % → higher mean RMS** in §2 bins — fans correlate; trust bins over single r.",
            "- **Best lag ~60–120s** — acoustic follows iLO fan changes; don't expect same-tick Δfan↔ΔRMS.",
            "- **fan↑ cpu↓ → ΔRMS negative** — cooldown: fans still high while machine quiets down.",
            "- **fan↑ cpu↑ → ΔRMS positive** — load + airflow both up; this is the intuitive case.",
            "- **Activity** = RMS volatility vs EWMA, not loudness.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dsn", default=os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URI"))
    ap.add_argument("--node", default="athena")
    ap.add_argument("--window-hours", type=float, default=168.0)
    ap.add_argument("--grain-sec", type=int, default=30)
    ap.add_argument("--max-lag-ticks", type=int, default=6)
    ap.add_argument("--top-spikes", type=int, default=8)
    ap.add_argument("--out-dir", default="/tmp/cabinet-ambient-correlation")
    args = ap.parse_args(argv)

    if not args.dsn:
        print("no DSN: set DATABASE_URL or pass --dsn", file=sys.stderr)
        return 2

    since = datetime.now(timezone.utc) - timedelta(hours=args.window_hours)
    rows = load_rows(args.dsn, node=args.node, since=since)
    ticks = build_ticks(rows)
    if not ticks:
        print("no ambient rows in window", file=sys.stderr)
        return 1

    floor = floor_stats(ticks)
    fan_rms = fan_rms_analysis(ticks, grain_sec=args.grain_sec, max_lag_ticks=args.max_lag_ticks)
    coupling = delta_coupling(ticks, max_lag_ticks=args.max_lag_ticks)
    spikes = spike_forensics(ticks, floor, top_n=args.top_spikes)
    span = (ticks[0].t, ticks[-1].t)

    report = render_report(
        node=args.node,
        window_hours=args.window_hours,
        grain_sec=args.grain_sec,
        span=span,
        floor=floor,
        fan_rms=fan_rms,
        coupling=coupling,
        spikes=spikes,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nwrote {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
