#!/usr/bin/env python3
"""Read-only SelfStateV1 signal-quality assessment.

Phase 1 hard gate, `docs/superpowers/specs/2026-07-18-objective-3-
consciousness-scaffolded-roadmap-design.md`: before the AST/HOT reducer's
Phase 1 output can be trusted for anything downstream, `SelfStateV1`'s own
signal quality must be measured against real historical data -- noise floor,
drift, oscillation period, per dimension -- not assumed. Real problems are
already known (not hypothetical) from the design pass this script implements:
a `confidence`/`available_capacity` merge-polarity masking bug, several
dead/folded-away `channel_dimension_map` entries, and a pre-fix coherence/
uncertainty sawtooth whose upstream field-level cause is closed but whose
full propagation through to `SelfStateV1`'s own values was never
independently re-checked.

Method: replay real historical `substrate_self_state` rows (same fetch
pattern as `scripts/analysis/measure_origination_gate.py`'s
`fetch_self_state_rows`), parse each into the real `SelfStateV1` model, and
for every dimension `SelfStateDimensionV1.dimension_id` can carry, compute
basic time-series diagnostics: rolling-std noise floor, a simple linear
drift-per-hour estimate, and a zero-crossing-based oscillation period
estimate. This is deliberately not sophisticated -- the spec's own bar is
"the same kind of measurement discipline used everywhere else in this
program," not a new ML pipeline.

This performs NO writes, emits NO events, flips NO flags, and proposes NO
`SelfStateV1` v2. Findings are reported plainly, including anything that
looks broken (e.g. a dimension still pinned or oscillating) -- fixing any
such finding is explicitly out of scope for this script and for Phase 1;
see the report's own "Findings for Juniper's sign-off" section.

Run:
    python scripts/analysis/measure_self_state_signal_quality.py --window-hours 48
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orion.analysis.self_state_signal_quality")

DEFAULT_WINDOW_HOURS: float = 48.0
MAX_ROWS: int = 200_000
ROLLING_WINDOW: int = 20

# All twelve real dimension_ids SelfStateDimensionV1 can carry
# (orion/schemas/self_state.py) -- measured unconditionally, in this fixed
# order, so a dimension that never appears in the window (e.g. a dead/
# folded-away channel) is still visible in the report as "n=0", not silently
# omitted.
DIMENSION_IDS: tuple[str, ...] = (
    "field_intensity",
    "coherence",
    "uncertainty",
    "agency_readiness",
    "resource_pressure",
    "execution_pressure",
    "reasoning_pressure",
    "reliability_pressure",
    "continuity_pressure",
    "introspection_pressure",
    "social_pressure",
    "transport_integrity",
)

# Heuristic thresholds -- deliberately simple, named so they're easy to
# argue with later, not buried magic numbers.
PINNED_STD_THRESHOLD = 0.01          # noise floor below this -> "pinned/flat"
HIGH_DRIFT_PER_HOUR_THRESHOLD = 0.05  # |slope| above this -> "drifting"
FAST_OSCILLATION_TICK_THRESHOLD = 6.0  # median period below this many ticks -> "oscillating fast"

OUTPUT_DIR = Path("/tmp/self-state-signal-quality")
REPORT_PATH = OUTPUT_DIR / "report.md"
CSV_PATH = OUTPUT_DIR / "dimensions.csv"
PROGRESS_PATH = OUTPUT_DIR / "progress.log"

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"


# ===========================================================================
# Pure summary layer -- no I/O, unit-testable on synthetic series.
# ===========================================================================


@dataclass
class DimensionSample:
    generated_at: datetime
    score: float
    confidence: float


@dataclass
class DimensionDiagnostics:
    dimension_id: str
    n: int = 0
    score_median: Optional[float] = None
    score_min: Optional[float] = None
    score_max: Optional[float] = None
    confidence_median: Optional[float] = None
    noise_floor_std: Optional[float] = None
    drift_per_hour: Optional[float] = None
    oscillation_period_ticks: Optional[float] = None
    zero_crossings: int = 0
    flags: list[str] = field(default_factory=list)


def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def rolling_std(values: list[float], window: int = ROLLING_WINDOW) -> list[float]:
    """Population std over a trailing window per point; first `window - 1`
    points use whatever's available so short series still produce output.
    """
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo : i + 1]
        m = _mean(chunk)
        var = _mean([(v - m) ** 2 for v in chunk])
        out.append(var ** 0.5)
    return out


def linear_drift_per_hour(times_sec: list[float], values: list[float]) -> Optional[float]:
    """Simple least-squares slope of value vs. time (seconds), converted to
    per-hour. None if fewer than 2 points or time has zero variance.
    """
    n = len(values)
    if n < 2:
        return None
    t_mean = _mean(times_sec)
    v_mean = _mean(values)
    num = sum((t - t_mean) * (v - v_mean) for t, v in zip(times_sec, values))
    den = sum((t - t_mean) ** 2 for t in times_sec)
    if den == 0:
        return None
    slope_per_sec = num / den
    return slope_per_sec * 3600.0


def zero_crossing_period_estimate(values: list[float]) -> tuple[Optional[float], int]:
    """Median distance (in tick count) between sign changes of
    `value - mean(values)`. A short median period relative to the series
    length indicates fast oscillation (e.g. a sawtooth). Returns
    (median_period_ticks, zero_crossing_count); period is None if fewer than
    2 crossings.
    """
    if len(values) < 3:
        return None, 0
    m = _mean(values)
    centered = [v - m for v in values]
    crossing_indices: list[int] = []
    prev_sign = None
    for i, v in enumerate(centered):
        if v == 0:
            continue
        sign = v > 0
        if prev_sign is not None and sign != prev_sign:
            crossing_indices.append(i)
        prev_sign = sign
    if len(crossing_indices) < 2:
        return None, len(crossing_indices)
    gaps = [b - a for a, b in zip(crossing_indices, crossing_indices[1:])]
    return _median([float(g) for g in gaps]), len(crossing_indices)


def diagnose_dimension(dimension_id: str, samples: list[DimensionSample]) -> DimensionDiagnostics:
    diag = DimensionDiagnostics(dimension_id=dimension_id, n=len(samples))
    if not samples:
        diag.flags.append("no_data")
        return diag

    ordered = sorted(samples, key=lambda s: s.generated_at)
    scores = [s.score for s in ordered]
    confidences = [s.confidence for s in ordered]
    t0 = ordered[0].generated_at
    times_sec = [(s.generated_at - t0).total_seconds() for s in ordered]

    diag.score_median = _median(scores)
    diag.score_min = min(scores)
    diag.score_max = max(scores)
    diag.confidence_median = _median(confidences)

    stds = rolling_std(scores)
    diag.noise_floor_std = _median(stds)

    diag.drift_per_hour = linear_drift_per_hour(times_sec, scores)

    period, crossings = zero_crossing_period_estimate(scores)
    diag.oscillation_period_ticks = period
    diag.zero_crossings = crossings

    if diag.noise_floor_std is not None and diag.noise_floor_std < PINNED_STD_THRESHOLD:
        diag.flags.append("pinned_or_flat")
    if diag.drift_per_hour is not None and abs(diag.drift_per_hour) > HIGH_DRIFT_PER_HOUR_THRESHOLD:
        diag.flags.append("drifting")
    if period is not None and period < FAST_OSCILLATION_TICK_THRESHOLD:
        diag.flags.append("fast_oscillation_sawtooth_suspect")
    if not diag.flags:
        diag.flags.append("nominal")
    return diag


# ===========================================================================
# I/O layer -- psycopg2 read-only. Mirrors measure_origination_gate.py's
# fetch_self_state_rows exactly.
# ===========================================================================


def open_readonly_connection(dsn: str):
    try:
        import psycopg2
    except Exception:  # pragma: no cover
        logger.error("psycopg2 unavailable; cannot open DB session")
        return None
    try:
        conn = psycopg2.connect(dsn)
    except Exception:
        logger.error("failed to connect to postgres", exc_info=True)
        return None
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SET default_transaction_read_only = on;")
            cur.execute("SHOW default_transaction_read_only;")
            value = cur.fetchone()
        if not value or str(value[0]).lower() != "on":
            logger.error("refusing to run: session is not read-only (got %r)", value)
            conn.close()
            return None
    except Exception:
        logger.error("failed to enforce read-only session", exc_info=True)
        try:
            conn.close()
        except Exception:
            pass
        return None
    return conn


def fetch_self_state_rows(conn, since: datetime, max_rows: int = MAX_ROWS) -> tuple[list[tuple[datetime, dict]], bool]:
    """Fetch (generated_at, self_state_json) ordered ASC. Returns (rows, truncated).
    Identical fetch shape to measure_origination_gate.py's function of the
    same name -- deliberately not shared as a common import so this script
    stays a standalone, independently-runnable measurement like its sibling.
    """
    if conn is None:
        return [], False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT generated_at, self_state_json
                FROM substrate_self_state
                WHERE generated_at >= %s
                ORDER BY generated_at ASC
                LIMIT %s
                """,
                (since, max_rows),
            )
            rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch substrate_self_state", exc_info=True)
        return [], False
    out: list[tuple[datetime, dict]] = []
    for generated_at, raw_json in rows:
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=timezone.utc)
        payload = raw_json
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                continue
        if not isinstance(payload, dict):
            continue
        out.append((generated_at, payload))
    return out, len(rows) >= max_rows


def parse_and_bucket_by_dimension(
    rows: list[tuple[datetime, dict]],
) -> tuple[dict[str, list[DimensionSample]], int]:
    from orion.schemas.self_state import SelfStateV1

    buckets: dict[str, list[DimensionSample]] = {d: [] for d in DIMENSION_IDS}
    skipped = 0
    for generated_at, payload in rows:
        try:
            state = SelfStateV1.model_validate(payload)
        except Exception:
            skipped += 1
            continue
        for dim_id, dim in state.dimensions.items():
            if dim_id not in buckets:
                buckets.setdefault(dim_id, [])
            buckets[dim_id].append(
                DimensionSample(generated_at=generated_at, score=dim.score, confidence=dim.confidence)
            )
    return buckets, skipped


# ===========================================================================
# Report rendering + orchestration.
# ===========================================================================


class ProgressLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._start = time.monotonic()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("w", encoding="utf-8")
        except Exception:
            self._fh = None

    def emit(self, title: str, *, percent: float, processed: int, total: int) -> None:
        elapsed = max(time.monotonic() - self._start, 1e-6)
        rate = processed / elapsed
        line = (
            f"{datetime.now(timezone.utc).isoformat()} | {title} | "
            f"{percent:5.1f}% | rows={processed}/{total} | rate={rate:.1f}/s"
        )
        logger.info(line)
        if self._fh is not None:
            try:
                self._fh.write(line + "\n")
                self._fh.flush()
            except Exception:
                pass

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass


def write_dimensions_csv(path: Path, diagnostics: list[DimensionDiagnostics]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "dimension_id", "n", "score_median", "score_min", "score_max",
                "confidence_median", "noise_floor_std", "drift_per_hour",
                "oscillation_period_ticks", "zero_crossings", "flags",
            ]
        )
        for d in diagnostics:
            writer.writerow(
                [
                    d.dimension_id, d.n,
                    _fmt(d.score_median), _fmt(d.score_min), _fmt(d.score_max),
                    _fmt(d.confidence_median), _fmt(d.noise_floor_std), _fmt(d.drift_per_hour),
                    _fmt(d.oscillation_period_ticks), d.zero_crossings,
                    ";".join(d.flags),
                ]
            )


def _fmt(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.4f}"


def render_report(
    *,
    window_label: str,
    window_start: datetime,
    window_end: datetime,
    n_rows: int,
    rows_truncated: bool,
    rows_skipped: int,
    diagnostics: list[DimensionDiagnostics],
    caveats: list[str],
) -> str:
    table_lines = "\n".join(
        f"| {d.dimension_id} | {d.n} | {_fmt(d.score_median)} | "
        f"[{_fmt(d.score_min)}, {_fmt(d.score_max)}] | {_fmt(d.noise_floor_std)} | "
        f"{_fmt(d.drift_per_hour)} | {_fmt(d.oscillation_period_ticks)} | "
        f"{d.zero_crossings} | {', '.join(d.flags)} |"
        for d in diagnostics
    )

    flagged = [d for d in diagnostics if d.flags and d.flags != ["nominal"] and d.flags != ["no_data"]]
    no_data = [d for d in diagnostics if "no_data" in d.flags]

    lines = [
        "# SelfStateV1 Signal-Quality Assessment (Phase 1 hard gate)",
        "",
        "Read-only. No writes, no events, no flag/config changes, no SelfStateV1 v2 "
        "proposed. Replays real `substrate_self_state` rows through the real "
        "`SelfStateV1` model and computes basic time-series diagnostics per dimension.",
        "",
        f"- Window: last {window_label} ({window_start.isoformat()} -> {window_end.isoformat()})",
        f"- self_state rows replayed: {n_rows}",
        f"- Rows truncated at MAX_ROWS={MAX_ROWS}: {rows_truncated}",
        f"- Rows skipped (failed to parse): {rows_skipped}",
        "",
        "## Per-dimension diagnostics",
        "",
        "| dimension_id | n | score median | score range | noise_floor (rolling std) | "
        "drift/hour | oscillation period (ticks) | zero-crossings | flags |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        table_lines or "| (no data) | | | | | | | | |",
        "",
        f"Thresholds used (named, not buried): pinned/flat if noise_floor < "
        f"{PINNED_STD_THRESHOLD}; drifting if |drift_per_hour| > "
        f"{HIGH_DRIFT_PER_HOUR_THRESHOLD}; fast-oscillation/sawtooth-suspect if median "
        f"oscillation period < {FAST_OSCILLATION_TICK_THRESHOLD} ticks.",
        "",
        "## Findings for Juniper's sign-off",
        "",
    ]

    if no_data:
        lines.append(
            f"- **{len(no_data)} dimension(s) with ZERO samples in this window**: "
            f"{', '.join(d.dimension_id for d in no_data)}. Consistent with the spec's "
            f"already-known \"dead/folded-away channel_dimension_map entries\" finding "
            f"-- these dimension_ids are never populated by the live builder over this "
            f"real window, not just theoretically unused."
        )
    if flagged:
        for d in flagged:
            lines.append(
                f"- **{d.dimension_id}**: flags=[{', '.join(d.flags)}] "
                f"(n={d.n}, noise_floor={_fmt(d.noise_floor_std)}, "
                f"drift/hour={_fmt(d.drift_per_hour)}, "
                f"oscillation_period_ticks={_fmt(d.oscillation_period_ticks)}, "
                f"zero_crossings={d.zero_crossings})"
            )
    if not no_data and not flagged:
        lines.append(
            "- No dimension in this window tripped the pinned/drifting/fast-oscillation "
            "heuristics above. This does NOT by itself clear the "
            "confidence/available_capacity merge-polarity bug named in the spec (that "
            "bug is about a cross-field merge, not a single dimension's own time "
            "series, and is out of scope to re-diagnose here) -- it only reports what "
            "this specific per-dimension pass found."
        )

    lines.extend(
        [
            "",
            "This is a **report only** -- per the spec's own framing, if any finding "
            "above looks like it needs fixing (a SelfStateV1 v2), that is a blocking "
            "prerequisite decision for Juniper, not something this script or Phase 1's "
            "reducer patch attempts to resolve.",
            "",
            "## Coverage caveats",
            "",
        ]
    )
    if caveats:
        lines.extend(f"- {c}" for c in caveats)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def run(window: timedelta, window_label: str) -> int:
    now = datetime.now(timezone.utc)
    window_start = now - window
    dsn = os.environ.get("POSTGRES_URI", DEFAULT_POSTGRES_URI)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = ProgressLog(PROGRESS_PATH)
    caveats: list[str] = []

    progress.emit("connect", percent=0.0, processed=0, total=0)
    conn = open_readonly_connection(dsn)
    if conn is None:
        caveats.append("postgres unavailable or not read-only; nothing measured")
        progress.close()
        print("ERROR: could not open a read-only postgres connection; see log")
        return 2

    rows, truncated = fetch_self_state_rows(conn, window_start)
    if truncated:
        caveats.append(f"self-state rows truncated at MAX_ROWS={MAX_ROWS}")
    progress.emit("self_state loaded", percent=30.0, processed=len(rows), total=len(rows))

    try:
        conn.close()
    except Exception:
        pass

    if not rows:
        caveats.append("no substrate_self_state rows in window; nothing to measure")
        progress.close()
        report = render_report(
            window_label=window_label, window_start=window_start, window_end=now,
            n_rows=0, rows_truncated=truncated, rows_skipped=0, diagnostics=[], caveats=caveats,
        )
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(report)
        return 2

    progress.emit("bucketing + diagnosing", percent=50.0, processed=0, total=len(rows))
    buckets, skipped = parse_and_bucket_by_dimension(rows)

    diagnostics = [diagnose_dimension(dim_id, buckets.get(dim_id, [])) for dim_id in DIMENSION_IDS]
    extra_dims = sorted(set(buckets) - set(DIMENSION_IDS))
    if extra_dims:
        caveats.append(f"unexpected dimension_ids present in live data (not in the known 12): {extra_dims}")
        diagnostics.extend(diagnose_dimension(dim_id, buckets[dim_id]) for dim_id in extra_dims)

    progress.emit("diagnosis done", percent=95.0, processed=len(rows), total=len(rows))

    write_dimensions_csv(CSV_PATH, diagnostics)
    report = render_report(
        window_label=window_label,
        window_start=window_start,
        window_end=now,
        n_rows=len(rows),
        rows_truncated=truncated,
        rows_skipped=skipped,
        diagnostics=diagnostics,
        caveats=caveats,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    progress.emit("done", percent=100.0, processed=len(rows), total=len(rows))
    progress.close()

    print(report)
    print(f"\nartifacts: {REPORT_PATH}, {CSV_PATH}, {PROGRESS_PATH}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only SelfStateV1 signal-quality assessment.")
    parser.add_argument(
        "--window-hours", type=float, default=DEFAULT_WINDOW_HOURS,
        help=f"analysis window in hours (default {DEFAULT_WINDOW_HOURS})",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_arg_parser().parse_args(argv)
    window = timedelta(hours=args.window_hours)
    window_label = f"{args.window_hours:g}h"
    return run(window, window_label)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
