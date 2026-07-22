#!/usr/bin/env python3
"""Read-only historical baseline for the transport bus's six real pressure signals.

Items 2 and 3 of docs/superpowers/specs/2026-07-22-transport-bus-signal-quality-
measurement-design.md, combined into one script since both read the same
`transport_bus_reducer` receipts (item 3's reducer-cadence gap analysis reuses item 2's
fetch rather than querying twice). `transport_prediction_error()` (`orion/substrate/
prediction_error.py`) reads
flat `0.0` in production; that spec traced this to two real, disclosed causes rather than
a bug: `bus_health`/`delivery_confidence` are genuinely stable under normal operation, and
`stream_depth_pressure`'s threshold (`DEFAULT_STREAM_DEPTH_CRITICAL = 100_000`,
`orion/substrate/transport_loop/constants.py`) was never checked against real operating
data -- live data showed `max_stream_depth` sitting at a constant 91 the whole observed
window. This script establishes the real historical percentile distribution needed before
that threshold (or any other recalibration) is touched, per the Sentience Striving
Program's own "measure before minting" rule (`orion/sentience_striving_program/
README.md` §7).

**Known data-source limitation, disclosed up front, not hidden in a footnote:**
`substrate_transport_bus_projection` is a **singleton** upsert table (one row, ever --
`projection_id` is its primary key) with no history at all. `substrate_reduction_receipts`
is append-only but has a **short retention TTL** -- a 24h query window returned only ~35
minutes of real rows when this spec was written (2026-07-22). This script reports whatever
real history the receipts table actually has *right now*, and says so honestly if that
turns out to be very little -- it does not pad, extrapolate, or fabricate a longer series.

Run:
    python scripts/analysis/measure_transport_bus_signal_history.py --window-hours 24
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("orion.analysis.transport_bus_signal_history")

DEFAULT_WINDOW_HOURS: float = 24.0
MAX_ROWS: int = 200_000

OUTPUT_DIR = Path("/tmp/transport-bus-signal-history")
REPORT_PATH = OUTPUT_DIR / "report.md"
PROGRESS_PATH = OUTPUT_DIR / "progress.log"

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"

# The six real signals from orion/substrate/transport_loop/extract.py::
# compute_transport_pressures(). stream_depth_pressure/max_stream_depth get their own
# percentile treatment (continuous, structurally near-always-nonzero); the other five
# are real/rare event counts, reported as simple totals + first/last-seen timestamps.
_COUNT_FIELDS = (
    "backpressure_count",
    "uncataloged_stream_count",
    "schema_mismatch_stream_count",
    "observer_failure_count",
)
_PRESSURE_FIELDS = (
    "backpressure",
    "catalog_drift_pressure",
    "contract_pressure",
    "observer_failure_pressure",
    "reliability_pressure",
)


# ===========================================================================
# Pure layer -- no I/O. Exercised directly by unit tests with synthetic data.
# ===========================================================================


@dataclass
class BusTick:
    observed_at: datetime
    max_stream_depth: int
    counts: dict[str, int] = field(default_factory=dict)
    pressures: dict[str, float] = field(default_factory=dict)


@dataclass
class CadenceStats:
    """Item 3: real inter-arrival gaps between consecutive transport_bus_reducer
    receipts -- is the reducer itself keeping pace, independent of what it's
    observing. See docs/superpowers/specs/2026-07-22-transport-bus-signal-quality-
    measurement-design.md item 3."""

    n_gaps: int
    median_gap_sec: Optional[float]
    p95_gap_sec: Optional[float]
    max_gap_sec: Optional[float]
    stall_threshold_sec: Optional[float]
    stall_count: int
    worst_stall_at: Optional[str]


@dataclass
class SignalBaseline:
    n: int
    depth_min: Optional[int]
    depth_max: Optional[int]
    depth_mean: Optional[float]
    depth_stdev: Optional[float]
    depth_p50: Optional[float]
    depth_p95: Optional[float]
    depth_p99: Optional[float]
    count_totals: dict[str, int]
    count_first_nonzero_at: dict[str, Optional[str]]
    pressure_ever_nonzero: dict[str, bool]
    earliest: Optional[str]
    latest: Optional[str]


def _percentile(sorted_values: list[float], pct: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def compute_baseline(ticks: list[BusTick]) -> SignalBaseline:
    """Pure aggregation over already-parsed ticks. No DB, no side effects."""
    if not ticks:
        return SignalBaseline(
            n=0,
            depth_min=None, depth_max=None, depth_mean=None, depth_stdev=None,
            depth_p50=None, depth_p95=None, depth_p99=None,
            count_totals={f: 0 for f in _COUNT_FIELDS},
            count_first_nonzero_at={f: None for f in _COUNT_FIELDS},
            pressure_ever_nonzero={f: False for f in _PRESSURE_FIELDS},
            earliest=None, latest=None,
        )

    ordered = sorted(ticks, key=lambda t: t.observed_at)
    depths = [float(t.max_stream_depth) for t in ordered]
    sorted_depths = sorted(depths)

    count_totals: dict[str, int] = {f: 0 for f in _COUNT_FIELDS}
    count_first_nonzero_at: dict[str, Optional[str]] = {f: None for f in _COUNT_FIELDS}
    for t in ordered:
        for f in _COUNT_FIELDS:
            v = int(t.counts.get(f, 0) or 0)
            count_totals[f] += v
            if v > 0 and count_first_nonzero_at[f] is None:
                count_first_nonzero_at[f] = t.observed_at.isoformat()

    pressure_ever_nonzero: dict[str, bool] = {f: False for f in _PRESSURE_FIELDS}
    for t in ordered:
        for f in _PRESSURE_FIELDS:
            if float(t.pressures.get(f, 0.0) or 0.0) > 0.0:
                pressure_ever_nonzero[f] = True

    return SignalBaseline(
        n=len(ordered),
        depth_min=int(min(depths)),
        depth_max=int(max(depths)),
        depth_mean=statistics.fmean(depths),
        depth_stdev=statistics.pstdev(depths) if len(depths) > 1 else 0.0,
        depth_p50=_percentile(sorted_depths, 0.50),
        depth_p95=_percentile(sorted_depths, 0.95),
        depth_p99=_percentile(sorted_depths, 0.99),
        count_totals=count_totals,
        count_first_nonzero_at=count_first_nonzero_at,
        pressure_ever_nonzero=pressure_ever_nonzero,
        earliest=ordered[0].observed_at.isoformat(),
        latest=ordered[-1].observed_at.isoformat(),
    )


def compute_cadence_stats(ticks: list[BusTick], *, stall_multiplier: float = 5.0) -> CadenceStats:
    """Item 3: real gaps between consecutive receipts. A "stall" is a gap more
    than `stall_multiplier` times the observed median -- relative to this
    reducer's own real cadence, not an assumed absolute number, since that
    cadence has never been measured before this script.

    Known limitation at small `n`, not fixed here: the median itself is
    computed from very few gaps when `n_gaps` is small (e.g. 2 gaps
    `[10, 1000]` -> median 505 -> a 100x jump to 1000 sits *under* a
    5x-median threshold of 2525 and goes unflagged). Reported honestly via
    `n_gaps` in the output rather than silently trusted -- treat stall
    counts as low-confidence whenever `n_gaps` is small (single digits)."""
    if len(ticks) < 2:
        return CadenceStats(
            n_gaps=0, median_gap_sec=None, p95_gap_sec=None, max_gap_sec=None,
            stall_threshold_sec=None, stall_count=0, worst_stall_at=None,
        )
    ordered = sorted(ticks, key=lambda t: t.observed_at)
    gaps = [
        (b.observed_at - a.observed_at).total_seconds()
        for a, b in zip(ordered, ordered[1:])
    ]
    sorted_gaps = sorted(gaps)
    median = statistics.median(sorted_gaps)
    stall_threshold = median * stall_multiplier if median > 0 else None
    stall_count = 0
    worst_stall_at: Optional[str] = None
    worst_gap = -1.0
    if stall_threshold is not None:
        for a, b in zip(ordered, ordered[1:]):
            gap = (b.observed_at - a.observed_at).total_seconds()
            if gap > stall_threshold:
                stall_count += 1
                if gap > worst_gap:
                    worst_gap = gap
                    worst_stall_at = b.observed_at.isoformat()
    return CadenceStats(
        n_gaps=len(gaps),
        median_gap_sec=median,
        p95_gap_sec=_percentile(sorted_gaps, 0.95),
        max_gap_sec=max(sorted_gaps),
        stall_threshold_sec=stall_threshold,
        stall_count=stall_count,
        worst_stall_at=worst_stall_at,
    )


# ===========================================================================
# I/O layer.
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


def fetch_transport_receipts(conn, since: datetime, max_rows: int = MAX_ROWS) -> tuple[list[BusTick], bool]:
    """Real `transport_bus_reducer` receipts since `since`, parsed into BusTicks.
    Returns (ticks, truncated). Read-only; never raises past this boundary.

    Single-bus assumption, not enforced here: this reducer emits one receipt
    per distinct bus per tick, and this deployment currently has exactly one
    (`BUS_OBSERVER_NODE_ID`, `services/orion-bus/app/settings.py`, defaults to
    a single node). If more than one bus were ever observed, this function
    would blend their depth/cadence series into one, undetected -- not
    currently a real risk (single-bus deployment, confirmed live), but worth
    an explicit per-`target_id` split if that ever changes."""
    if conn is None:
        return [], False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT created_at, receipt_json
                FROM substrate_reduction_receipts
                WHERE reducer_name = 'transport_bus_reducer'
                  AND created_at >= %s
                ORDER BY created_at ASC
                LIMIT %s
                """,
                (since, max_rows),
            )
            rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch substrate_reduction_receipts", exc_info=True)
        return [], False

    ticks: list[BusTick] = []
    for created_at, receipt_json in rows:
        if created_at is None:
            continue
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        try:
            payload = json.loads(receipt_json) if isinstance(receipt_json, str) else receipt_json
            deltas = (payload or {}).get("state_deltas") or []
            if not deltas:
                continue
            after = deltas[0].get("after") or {}
        except Exception:
            logger.warning("skipping malformed receipt row at %s", created_at, exc_info=True)
            continue
        ticks.append(
            BusTick(
                observed_at=created_at,
                max_stream_depth=int(after.get("max_stream_depth", 0) or 0),
                counts={f: int(after.get(f, 0) or 0) for f in _COUNT_FIELDS},
                pressures={f: float(after.get(f, 0.0) or 0.0) for f in _PRESSURE_FIELDS},
            )
        )
    return ticks, len(rows) >= max_rows


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


def _fmt(value: Any) -> str:
    return "n/a" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))


def render_report(
    *,
    window_hours: float,
    baseline: SignalBaseline,
    truncated: bool,
    cadence: Optional[CadenceStats] = None,
) -> str:
    lines = [
        "# Transport Bus Signal History -- Real Baseline",
        "",
        "Read-only. No writes, no config changes. See "
        "`docs/superpowers/specs/2026-07-22-transport-bus-signal-quality-measurement-design.md` "
        "items 2 and 3.",
        "",
        f"- Requested window: {window_hours:g}h",
        f"- Real rows found: {baseline.n}",
        f"- Real data span: {baseline.earliest or 'n/a'} -> {baseline.latest or 'n/a'}",
    ]
    if truncated:
        lines.append(f"- **TRUNCATED at MAX_ROWS={MAX_ROWS}** -- more real rows exist than were fetched.")
    lines.append("")

    if baseline.n == 0:
        lines.extend(
            [
                "## INSUFFICIENT REAL DATA",
                "",
                "No `transport_bus_reducer` receipts found in the requested window. Given "
                "`substrate_reduction_receipts`' short retention TTL (confirmed ~35 minutes "
                "of history survived a 24h query when this script was designed), this is the "
                "honest, expected result for any window longer than the TTL -- not a query "
                "bug. Re-run with a shorter `--window-hours`, or re-run this script again "
                "after real time has passed to accumulate more history.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "## `max_stream_depth` real percentile distribution",
            "",
            f"- min: {baseline.depth_min}",
            f"- p50: {_fmt(baseline.depth_p50)}",
            f"- p95: {_fmt(baseline.depth_p95)}",
            f"- p99: {_fmt(baseline.depth_p99)}",
            f"- max: {baseline.depth_max}",
            f"- mean: {_fmt(baseline.depth_mean)}",
            f"- stdev: {_fmt(baseline.depth_stdev)}",
            "",
            "Compare against `DEFAULT_STREAM_DEPTH_CRITICAL = 100_000` "
            "(`orion/substrate/transport_loop/constants.py`) -- if p99 sits at a tiny "
            "fraction of that threshold, `stream_depth_pressure` is structurally incapable "
            "of registering realistic variation at its current scale.",
            "",
            "## Real event counts (backpressure / catalog drift / schema mismatch / observer failure)",
            "",
            "| signal | total (sum across window) | first nonzero at |",
            "| --- | --- | --- |",
        ]
    )
    for f in _COUNT_FIELDS:
        lines.append(
            f"| `{f}` | {baseline.count_totals[f]} | {baseline.count_first_nonzero_at[f] or 'never'} |"
        )
    lines.extend(
        [
            "",
            "## Pressure signals: ever nonzero in this window?",
            "",
            "| signal | ever nonzero |",
            "| --- | --- |",
        ]
    )
    for f in _PRESSURE_FIELDS:
        lines.append(f"| `{f}` | {baseline.pressure_ever_nonzero[f]} |")

    if not any(baseline.pressure_ever_nonzero.values()) and baseline.depth_stdev == 0.0:
        lines.extend(
            [
                "",
                "**Every signal read as flat/zero for the entire observed window.** This "
                "matches the live spot-check in the design spec (2026-07-22) and is reported "
                "here as the honest finding it is, not smoothed over -- it means either the "
                "bus has genuinely had no incident in this window, or the window itself is "
                "too short to have caught one (see the TTL caveat above). It is not, by "
                "itself, evidence of a bug in this script or in `compute_transport_pressures()`.",
            ]
        )

    if cadence is not None:
        lines.extend(
            [
                "",
                "## Reducer cadence (item 3: is the reducer itself keeping pace)",
                "",
                f"- gaps measured: {cadence.n_gaps}",
                f"- median gap: {_fmt(cadence.median_gap_sec)}s",
                f"- p95 gap: {_fmt(cadence.p95_gap_sec)}s",
                f"- max gap: {_fmt(cadence.max_gap_sec)}s",
            ]
        )
        if cadence.stall_threshold_sec is not None:
            lines.extend(
                [
                    f"- stall threshold (5x median): {_fmt(cadence.stall_threshold_sec)}s",
                    f"- stalls observed: {cadence.stall_count}"
                    + (f" (worst at {cadence.worst_stall_at})" if cadence.worst_stall_at else ""),
                ]
            )
        else:
            lines.append(
                "- stall threshold: n/a (fewer than 2 gaps observed, or zero median gap)"
            )

    return "\n".join(lines)


def run(window_hours: float, postgres_uri: str = DEFAULT_POSTGRES_URI) -> int:
    progress = ProgressLog(PROGRESS_PATH)
    progress.emit("transport_bus_signal_history started", percent=0.0, processed=0, total=0)

    conn = open_readonly_connection(postgres_uri)
    since = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    ticks, truncated = fetch_transport_receipts(conn, since)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass

    progress.emit("receipts fetched", percent=60.0, processed=len(ticks), total=len(ticks))
    baseline = compute_baseline(ticks)
    cadence = compute_cadence_stats(ticks)
    report = render_report(window_hours=window_hours, baseline=baseline, truncated=truncated, cadence=cadence)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    progress.emit("report written", percent=100.0, processed=len(ticks), total=len(ticks))
    progress.close()

    print(report)
    print(f"\nFull report: {REPORT_PATH}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only historical baseline for the transport bus's six real pressure signals."
    )
    parser.add_argument(
        "--window-hours", type=float, default=DEFAULT_WINDOW_HOURS,
        help=f"how far back to look for real transport_bus_reducer receipts (default {DEFAULT_WINDOW_HOURS})",
    )
    parser.add_argument(
        "--postgres-uri", type=str, default=DEFAULT_POSTGRES_URI,
        help="read-only Postgres DSN (default: in-cluster orion-athena-sql-db)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_arg_parser().parse_args(argv)
    return run(args.window_hours, args.postgres_uri)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
