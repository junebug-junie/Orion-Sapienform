#!/usr/bin/env python3
"""Read-only replay of Candidate A (precision-weighted prediction-error salience) over
real `substrate_reduction_receipts` history.

Sentience Striving Program (`orion/sentience_striving_program/README.md`) §7 ("measure
before minting") and
`docs/superpowers/specs/2026-07-21-attention-salience-cathedral-replacement-tentative-
plan.md`'s "Candidate A" section. Replays
`orion/attention/field_attention/candidate_precision_weighted.py::
precision_weighted_salience()` -- a pure function, imported and exercised here, never
reimplemented -- over real historical prediction-error receipts, the same discipline as
`measure_emergent_clustering_probe.py` / `measure_origination_gate.py` /
`measure_ast_hot_reducer.py`.

This performs NO writes, emits NO events, flips NO flags, and imports nothing from
`orion.spark.concept_induction`.

## Honest scoping, re-verified live at run time, not assumed from a prior check

`orion/substrate/prediction_error.py` has five instruments (execution, transport,
biometrics, chat, route), each of which -- when `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES`
is enabled -- writes a `substrate_reduction_receipts` row via `_prediction_error_receipt()`
(`services/orion-substrate-runtime/app/worker.py`) with `reducer_name` equal to the
receipt's `state_deltas[0].reducer_id` (`f"substrate.{reducer_key}"`). This script queries
all five real `reducer_name` values (`REDUCER_NAMES` below) and reports, per reducer,
whether it has enough real rows (`QUALIFYING_MIN_ROWS`) to run a meaningful replay --
it does NOT hardcode the assumption that only biometrics qualifies, so a future run
against a live deployment where execution/transport/chat/route have accumulated real
history will pick them up automatically, honestly, without a code change here.

**A structural constraint on ALL five reducers, not a today-only staleness note:**
`substrate_reduction_receipts` retains `success`-kind receipts for only
`ORION_RECEIPT_RETENTION_SUCCESS_MINUTES` (30 minutes in the live `.env` at time of
writing) before a background pruner deletes them -- confirmed empirically (6,353 total
live rows spanning 2026-07-03 -> run time, but every row's `expires_at` within ~30 minutes
of its own `created_at`). This means the real historical series available to Candidate A
is *always* bounded to a rolling window on that order, for whichever reducer(s) qualify --
not a limitation of this script, a property of the live data source it reads from.

## Disclosed scoping decisions

1. **Rolling window, not full-history expanding window, for precision estimation.**
   Feldman & Friston's "precision" is meant to capture a signal's *current* volatility,
   which can itself drift over time -- an expanding window that includes very old
   observations would let stale variance dominate a fresh estimate. Each tick's
   precision is computed from the `--rolling-window` (default 20) most recent samples
   up to and including that tick, matching `precision_weighted_salience()`'s own
   documented contract ("the LAST element is treated as the current error").
2. **Window-hours/gap-hours CLI flags kept for comparability with Candidate B's probe**
   (`measure_emergent_clustering_probe.py --window-hours 24 --gap-hours 12`), but this
   script's own `MIN_WINDOW_HOURS` default (0.05h = 3 minutes) is far smaller than that
   script's (4h), because the real retention-bounded data volume here is orders of
   magnitude smaller. If the real available span cannot support even a same-length
   two-window split at the requested `--window-hours`/`--gap-hours`, this script reports
   that honestly and falls back to a single full-window analysis -- it does NOT force an
   artificial split, per the task's explicit instruction not to fabricate alignment with
   Candidate B's window.
3. **Qualifying threshold** (`QUALIFYING_MIN_ROWS`, default 20): a reducer needs at least
   this many real rows before this script treats its precision estimate as meaningful.
   Below this, the reducer is reported as `NOT_QUALIFIED`, not silently skipped or
   silently analyzed anyway.

Run:
    python scripts/analysis/measure_precision_weighted_salience_probe.py \\
        --window-hours 24 --gap-hours 12 --rolling-window 20
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from orion.attention.field_attention.candidate_precision_weighted import (
    PRECISION_VARIANCE_FLOOR,
    PrecisionWeightedSalienceResult,
    normalize_across_targets,
    precision_weighted_salience,
)

logger = logging.getLogger("orion.analysis.precision_weighted_salience_probe")

# Real reducer_name values a prediction-error node write can carry -- see
# `_prediction_error_receipt()`'s `reducer_id=f"substrate.{reducer_key}"` and the
# `reducer_key=` call sites in `services/orion-substrate-runtime/app/worker.py`.
REDUCER_NAMES: tuple[str, ...] = (
    "substrate.node_biometrics",
    "substrate.execution_trajectory",
    "substrate.transport_bus",
    "substrate.chat_session",
    "substrate.route_arbitration",
)

DEFAULT_WINDOW_HOURS: float = 24.0
DEFAULT_GAP_HOURS: float = 12.0
MIN_WINDOW_HOURS: float = 0.05  # 3 minutes -- see module docstring disclosed decision #2
DEFAULT_ROLLING_WINDOW: int = 20
QUALIFYING_MIN_ROWS: int = 20
MAX_ROWS: int = 50_000

OUTPUT_DIR = Path("/tmp/precision-weighted-salience-probe")
REPORT_PATH = OUTPUT_DIR / "report.md"
CSV_DIR = OUTPUT_DIR / "ticks"
PROGRESS_PATH = OUTPUT_DIR / "progress.log"

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"


# ===========================================================================
# Pure layer -- no I/O. Runs the real candidate_precision_weighted module over
# aligned (timestamp, value) series and summarizes real numbers.
# ===========================================================================


def parse_prediction_error(raw: Any) -> Optional[float]:
    """Parse a raw `receipt_json->...->>'prediction_error'` value (a JSON text
    scalar, possibly None/malformed). Never raises."""
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def compute_rolling_results(
    values: list[float], rolling_window: int
) -> list[PrecisionWeightedSalienceResult]:
    """Run the real `precision_weighted_salience()` at each tick, using the
    `rolling_window` most recent samples up to and including that tick (see module
    docstring disclosed decision #1). Tick 0 gets a window of size 1."""
    results: list[PrecisionWeightedSalienceResult] = []
    for i in range(len(values)):
        start = max(0, i - rolling_window + 1)
        results.append(precision_weighted_salience(values[start : i + 1]))
    return results


@dataclass
class ResultSummary:
    n: int
    salience_min: float
    salience_max: float
    salience_mean: float
    salience_median: float
    salience_stdev: float
    precision_min: float
    precision_max: float
    precision_mean: float
    variance_floored_count: int
    variance_floored_pct: float
    raw_error_min: float
    raw_error_max: float
    raw_error_mean: float
    raw_error_stdev: float


def summarize_results(results: list[PrecisionWeightedSalienceResult]) -> Optional[ResultSummary]:
    if not results:
        return None
    saliences = [r.salience for r in results]
    precisions = [r.precision for r in results]
    errors = [r.current_error for r in results]
    n_floored = sum(1 for r in results if r.variance_floored)
    return ResultSummary(
        n=len(results),
        salience_min=min(saliences),
        salience_max=max(saliences),
        salience_mean=statistics.fmean(saliences),
        salience_median=statistics.median(saliences),
        salience_stdev=statistics.pstdev(saliences) if len(saliences) > 1 else 0.0,
        precision_min=min(precisions),
        precision_max=max(precisions),
        precision_mean=statistics.fmean(precisions),
        variance_floored_count=n_floored,
        variance_floored_pct=(n_floored / len(results)) * 100.0,
        raw_error_min=min(errors),
        raw_error_max=max(errors),
        raw_error_mean=statistics.fmean(errors),
        raw_error_stdev=statistics.pstdev(errors) if len(errors) > 1 else 0.0,
    )


@dataclass
class WindowSpec:
    start: datetime
    end: datetime
    label: str


def choose_windows(
    min_ts: datetime,
    max_ts: datetime,
    window_hours: float = DEFAULT_WINDOW_HOURS,
    gap_hours: float = DEFAULT_GAP_HOURS,
    min_window_hours: float = MIN_WINDOW_HOURS,
) -> Optional[tuple[WindowSpec, WindowSpec]]:
    """Same anchoring/degradation shape as `measure_emergent_clustering_probe.py`'s
    `choose_windows`, generic over a plain (min_ts, max_ts) span rather than a target
    universe. Returns None when even a back-to-back split can't produce two windows of
    at least `min_window_hours` each -- reported honestly, not fabricated."""
    if min_ts is None or max_ts is None or max_ts <= min_ts:
        return None
    total_hours = (max_ts - min_ts).total_seconds() / 3600.0

    if total_hours >= 2 * window_hours + gap_hours:
        a_start, a_end = min_ts, min_ts + timedelta(hours=window_hours)
        b_end = max_ts
        b_start = b_end - timedelta(hours=window_hours)
        return (
            WindowSpec(a_start, a_end, f"{window_hours:g}h (window A, start-anchored)"),
            WindowSpec(b_start, b_end, f"{window_hours:g}h (window B, end-anchored)"),
        )

    if total_hours >= 2 * min_window_hours:
        half_hours = total_hours / 2.0
        mid = min_ts + timedelta(hours=half_hours)
        return (
            WindowSpec(min_ts, mid, f"{half_hours * 60:.1f}min (window A, back-to-back split)"),
            WindowSpec(mid, max_ts, f"{half_hours * 60:.1f}min (window B, back-to-back split)"),
        )

    return None


def partition_by_window(
    timestamps: list[datetime],
    values: list[float],
    win_a: WindowSpec,
    win_b: WindowSpec,
) -> tuple[list[float], list[float]]:
    a: list[float] = []
    b: list[float] = []
    for ts, v in zip(timestamps, values):
        if win_a.start <= ts < win_a.end:
            a.append(v)
        elif win_b.start <= ts <= win_b.end:
            b.append(v)
    return a, b


# ===========================================================================
# I/O layer -- psycopg2 read-only. Same connection contract as
# measure_emergent_clustering_probe.py (refuses a non-read-only session).
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


def fetch_reducer_rows(
    conn, reducer_name: str, max_rows: int = MAX_ROWS
) -> tuple[list[datetime], list[float]]:
    """Real (timestamp, prediction_error) rows for one reducer_name, ASC by
    created_at. Skips rows whose value fails to parse (never fatal)."""
    if conn is None:
        return [], []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT created_at,
                       receipt_json->'state_deltas'->0->'after'->'pressure_hints'
                         ->>'prediction_error' AS pe
                FROM substrate_reduction_receipts
                WHERE reducer_name = %s
                ORDER BY created_at ASC
                LIMIT %s
                """,
                (reducer_name, max_rows),
            )
            rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch rows for reducer_name=%s", reducer_name, exc_info=True)
        return [], []

    timestamps: list[datetime] = []
    values: list[float] = []
    for ts, raw_pe in rows:
        pe = parse_prediction_error(raw_pe)
        if pe is None or ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        timestamps.append(ts)
        values.append(pe)
    return timestamps, values


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

    def emit(self, title: str, *, percent: float, processed: int, total: int, note: str = "") -> None:
        elapsed = max(time.monotonic() - self._start, 1e-6)
        rate = processed / elapsed
        line = (
            f"{datetime.now(timezone.utc).isoformat()} | {title} | "
            f"{percent:5.1f}% | rows={processed}/{total} | rate={rate:.1f}/s"
            f"{(' | ' + note) if note else ''}"
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


def write_ticks_csv(
    path: Path, timestamps: list[datetime], results: list[PrecisionWeightedSalienceResult]
) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["created_at", "current_error", "variance", "precision", "salience", "n_samples", "variance_floored"]
        )
        for ts, r in zip(timestamps, results):
            writer.writerow(
                [ts.isoformat(), f"{r.current_error:.6f}", f"{r.variance:.8f}", f"{r.precision:.4f}", f"{r.salience:.4f}", r.n_samples, r.variance_floored]
            )


def _fmt(value: Optional[float], nd: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{nd}f}"


def _summary_block(label: str, summary: Optional[ResultSummary]) -> list[str]:
    if summary is None:
        return [f"### {label}: no rows", ""]
    return [
        f"### {label} ({summary.n} ticks)",
        "",
        f"- Raw `|prediction_error|` (current-tick value): min={_fmt(summary.raw_error_min)} "
        f"max={_fmt(summary.raw_error_max)} mean={_fmt(summary.raw_error_mean)} "
        f"stdev={_fmt(summary.raw_error_stdev)}",
        f"- Precision: min={_fmt(summary.precision_min, 2)} max={_fmt(summary.precision_max, 2)} "
        f"mean={_fmt(summary.precision_mean, 2)}",
        f"- Salience (precision x |error|): min={_fmt(summary.salience_min, 2)} "
        f"max={_fmt(summary.salience_max, 2)} mean={_fmt(summary.salience_mean, 2)} "
        f"median={_fmt(summary.salience_median, 2)} stdev={_fmt(summary.salience_stdev, 2)}",
        f"- Variance-floor instability rate: {summary.variance_floored_count}/{summary.n} "
        f"({summary.variance_floored_pct:.1f}%) ticks hit `PRECISION_VARIANCE_FLOOR="
        f"{PRECISION_VARIANCE_FLOOR:g}` (precision pinned at its ceiling, "
        f"{1.0 / PRECISION_VARIANCE_FLOOR:g})",
        "",
    ]


def render_report(
    *,
    rolling_window: int,
    reducer_reports: dict[str, dict[str, Any]],
    caveats: list[str],
    cross_target_normalization: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# Precision-Weighted Salience Probe (Candidate A) -- Real Prediction-Error Receipt History",
        "",
        "Read-only. No writes, no events, no flag/config changes. Replays the real, pure "
        "`orion/attention/field_attention/candidate_precision_weighted.py::"
        "precision_weighted_salience()` function over real `substrate_reduction_receipts` "
        "history, per the Sentience Striving Program §7 ('measure before minting') and "
        "`docs/superpowers/specs/2026-07-21-attention-salience-cathedral-replacement-"
        "tentative-plan.md`'s Candidate A section. Imports nothing from "
        "`orion.spark.concept_induction`.",
        "",
        f"- Rolling window for precision estimation: {rolling_window} samples",
        f"- Reducer names checked (all 5 real `orion/substrate/prediction_error.py` "
        f"instruments): {', '.join(f'`{r}`' for r in REDUCER_NAMES)}",
        "",
        "## Per-reducer results",
        "",
    ]

    for reducer_name in REDUCER_NAMES:
        rep = reducer_reports.get(reducer_name, {})
        status = rep.get("status", "UNKNOWN")
        n_rows = rep.get("n_rows", 0)
        lines.append(f"### `{reducer_name}` -- **{status}**")
        lines.append("")
        lines.append(f"- Real rows found (bounded by live receipt retention): {n_rows}")
        if status == "NOT_QUALIFIED":
            lines.append(
                f"- Below `QUALIFYING_MIN_ROWS={QUALIFYING_MIN_ROWS}` -- not enough real "
                "history to compute a meaningful variance/precision estimate. No analysis "
                "run for this reducer (honest scoping, not silently extended)."
            )
            lines.append("")
            continue
        if status == "NO_ROWS":
            lines.append("- No real receipts at all for this reducer_name at run time.")
            lines.append("")
            continue

        real_span = rep.get("real_span_label", "n/a")
        lines.append(f"- Real data span: {real_span}")
        lines.append("")
        lines.extend(_summary_block("Full available window", rep.get("full_summary")))

        window_status = rep.get("window_status")
        if window_status == "INSUFFICIENT_FOR_TWO_WINDOW_SPLIT":
            lines.append(
                "**Two-window comparison (matching Candidate B's `--window-hours`/"
                "`--gap-hours` convention): not attempted.** The real available span for "
                f"this reducer ({real_span}) is too short to carve out two non-overlapping "
                f"windows at the requested size, even with this script's own reduced "
                f"`MIN_WINDOW_HOURS={MIN_WINDOW_HOURS:g}h` floor (itself far smaller than "
                "Candidate B's 4h default, given how much smaller the real retention-bounded "
                "data volume is here). Reported honestly rather than forcing an artificial "
                "split -- see module docstring disclosed decision #2."
            )
            lines.append("")
        elif window_status == "SPLIT":
            win_a_label = rep.get("win_a_label", "")
            win_b_label = rep.get("win_b_label", "")
            lines.append(f"**Two-window comparison** -- Window A: {win_a_label}; Window B: {win_b_label}")
            lines.append("")
            lines.extend(_summary_block("Window A", rep.get("window_a_summary")))
            lines.extend(_summary_block("Window B", rep.get("window_b_summary")))

    lines.extend(["## Cross-target normalization", ""])
    lines.append(
        "Raw `precision_weighted_salience().salience` is unbounded and dominated by each "
        "target's own historical variance scale -- not comparable across reducers, and not "
        "a valid `FieldAttentionTargetV1.salience_score` ([0,1]) drop-in, without "
        "`normalize_across_targets()`. This section compares each qualified reducer's most "
        "recent real tick as the current competing set."
    )
    lines.append("")
    ctn = cross_target_normalization or {}
    qualified_count = ctn.get("qualified_count", 0)
    if qualified_count >= 2:
        raw = ctn.get("raw", {})
        normalized = ctn.get("normalized", {})
        lines.append(f"| Reducer | Raw salience (latest tick) | Normalized [0,1] |")
        lines.append(f"|---|---|---|")
        for reducer_name in sorted(raw, key=lambda r: -normalized.get(r, 0.0)):
            lines.append(
                f"| `{reducer_name}` | {raw[reducer_name]:.2f} | {normalized[reducer_name]:.4f} |"
            )
        lines.append("")
    else:
        lines.append(
            f"**Not demonstrated this run** -- only {qualified_count} reducer(s) qualified. "
            "`normalize_across_targets()` needs at least 2 real competing targets to show a "
            "meaningful comparison; reported honestly rather than faked against a single-"
            "target competing set. Re-run once more real receipt history has accumulated "
            "across multiple domains (e.g. after PR #1239's execution/route fix has been "
            "live long enough to accumulate real history)."
        )
        lines.append("")

    lines.extend(["## Coverage caveats", ""])
    if caveats:
        lines.extend(f"- {c}" for c in caveats)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def run(
    window_hours: float = DEFAULT_WINDOW_HOURS,
    gap_hours: float = DEFAULT_GAP_HOURS,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
) -> int:
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

    reducer_reports: dict[str, dict[str, Any]] = {}
    total_reducers = len(REDUCER_NAMES)

    for idx, reducer_name in enumerate(REDUCER_NAMES):
        percent = (idx / total_reducers) * 90.0
        progress.emit(f"fetching {reducer_name}", percent=percent, processed=idx, total=total_reducers)
        timestamps, values = fetch_reducer_rows(conn, reducer_name, MAX_ROWS)
        n_rows = len(values)

        if n_rows == 0:
            reducer_reports[reducer_name] = {"status": "NO_ROWS", "n_rows": 0}
            continue
        if n_rows < QUALIFYING_MIN_ROWS:
            reducer_reports[reducer_name] = {"status": "NOT_QUALIFIED", "n_rows": n_rows}
            continue

        min_ts, max_ts = timestamps[0], timestamps[-1]
        span_hours = (max_ts - min_ts).total_seconds() / 3600.0
        real_span_label = f"{min_ts.isoformat()} -> {max_ts.isoformat()} ({span_hours * 60:.1f} min)"

        full_results = compute_rolling_results(values, rolling_window)
        full_summary = summarize_results(full_results)

        report: dict[str, Any] = {
            "status": "QUALIFIED",
            "n_rows": n_rows,
            "real_span_label": real_span_label,
            "full_summary": full_summary,
            # Most recent real tick's raw (unnormalized) salience -- oldest-first
            # contract of compute_rolling_results means the last element is the
            # latest real observation. Used below for cross-target normalization.
            "latest_raw_salience": full_results[-1].salience,
        }

        windows = choose_windows(min_ts, max_ts, window_hours, gap_hours, MIN_WINDOW_HOURS)
        if windows is None:
            report["window_status"] = "INSUFFICIENT_FOR_TWO_WINDOW_SPLIT"
        else:
            win_a, win_b = windows
            vals_a, vals_b = partition_by_window(timestamps, values, win_a, win_b)
            if len(vals_a) < QUALIFYING_MIN_ROWS or len(vals_b) < QUALIFYING_MIN_ROWS:
                report["window_status"] = "INSUFFICIENT_FOR_TWO_WINDOW_SPLIT"
                caveats.append(
                    f"{reducer_name}: a two-window split was geometrically possible but at "
                    f"least one window had < QUALIFYING_MIN_ROWS={QUALIFYING_MIN_ROWS} real "
                    f"rows (A={len(vals_a)}, B={len(vals_b)}) -- reported as insufficient "
                    "rather than run on too little data."
                )
            else:
                report["window_status"] = "SPLIT"
                report["win_a_label"] = win_a.label
                report["win_b_label"] = win_b.label
                report["window_a_summary"] = summarize_results(compute_rolling_results(vals_a, rolling_window))
                report["window_b_summary"] = summarize_results(compute_rolling_results(vals_b, rolling_window))

        reducer_reports[reducer_name] = report
        write_ticks_csv(CSV_DIR / f"{reducer_name.replace('.', '_')}.csv", timestamps, full_results)

    try:
        conn.close()
    except Exception:
        pass

    progress.emit("done", percent=100.0, processed=total_reducers, total=total_reducers)
    progress.close()

    qualified = [r for r, rep in reducer_reports.items() if rep.get("status") == "QUALIFIED"]
    if not qualified:
        caveats.append(
            "no reducer had enough real receipt history to qualify for analysis at run "
            "time -- see per-reducer status above"
        )

    # Cross-target normalization (added in review, 2026-07-22): raw
    # precision_weighted_salience() output is unbounded and dominated by each
    # target's own historical variance scale -- not directly comparable across
    # reducers, and not a valid FieldAttentionTargetV1.salience_score ([0,1])
    # drop-in. Reports each qualified reducer's own most-recent real tick as the
    # "current competing set" and normalizes across it, honestly, rather than
    # silently comparing raw magnitudes or fabricating a competition when fewer
    # than 2 real reducers qualify this run.
    latest_raw: dict[str, float] = {
        reducer_name: reducer_reports[reducer_name]["latest_raw_salience"] for reducer_name in qualified
    }
    cross_target_normalization: dict[str, Any] = {"qualified_count": len(qualified)}
    if len(qualified) >= 2:
        cross_target_normalization["raw"] = latest_raw
        cross_target_normalization["normalized"] = normalize_across_targets(latest_raw)
    else:
        caveats.append(
            f"cross-target normalization not demonstrated this run: only "
            f"{len(qualified)} reducer(s) qualified, and normalize_across_targets() "
            "needs at least 2 real competing targets to show a meaningful "
            "comparison -- reported honestly rather than faked with a single-target "
            "competing set."
        )

    report_md = render_report(
        rolling_window=rolling_window,
        reducer_reports=reducer_reports,
        caveats=caveats,
        cross_target_normalization=cross_target_normalization,
    )
    REPORT_PATH.write_text(report_md, encoding="utf-8")
    print(report_md)
    print(f"\nartifacts: {REPORT_PATH}, {CSV_DIR}/, {PROGRESS_PATH}")
    return 0 if qualified else 2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only replay of Candidate A (precision-weighted salience) over real prediction-error receipt history."
    )
    parser.add_argument(
        "--window-hours", type=float, default=DEFAULT_WINDOW_HOURS,
        help=f"size of each of the two comparison windows in hours, for comparability with "
             f"Candidate B's probe (default {DEFAULT_WINDOW_HOURS})",
    )
    parser.add_argument(
        "--gap-hours", type=float, default=DEFAULT_GAP_HOURS,
        help=f"untouched real history between the two windows, in hours (default {DEFAULT_GAP_HOURS})",
    )
    parser.add_argument(
        "--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW,
        help=f"number of most-recent real samples used to estimate precision at each tick "
             f"(default {DEFAULT_ROLLING_WINDOW})",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_arg_parser().parse_args(argv)
    return run(args.window_hours, args.gap_hours, args.rolling_window)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
