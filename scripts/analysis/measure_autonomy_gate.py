#!/usr/bin/env python3
"""Read-only autonomy-origination measurement gate.

This worker answers two empirical questions from durable history and prints two
deterministic GO / NO-GO verdicts. It performs NO writes, emits NO events, and
flips NO flags. It is a measurement instrument, not a cognition change.

Question (a) — endogenous drift:
    Does ``SelfStateV1`` drift during exogenous silence? (gates a downstream
    "endogenous origination" spec)

Question (b) — internal economy:
    How often do >=2 drives co-activate, and does ``resource_pressure`` actually
    rise? (gates a downstream "internal economy" spec)

Design contract (see AGENTS.md / task brief):

* PURE functions (this module's top half) have NO I/O. They operate on
  in-memory normalized dataclasses and are the only surface the unit tests
  exercise. Window classification, metric computation, and the verdict rules
  all live here.
* I/O ADAPTERS (this module's bottom half) fetch rows over a read-only psycopg2
  session. The drive-audit time-series is read ONLY from the ``drive_audits``
  Postgres table written by orion-sql-writer. The old Fuseki DriveAudit graph
  (frozen since 2026-06-19, flat-pinned-era data) is deliberately NOT read --
  windows the table does not cover are honestly UNMEASURABLE, never backfilled
  from a dead sensor. Every adapter degrades gracefully to empty / None on
  absent input and NEVER raises on missing data or columns.

Run:
    python scripts/analysis/measure_autonomy_gate.py --window-days 7
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger("orion.analysis.autonomy_gate")

# ---------------------------------------------------------------------------
# Module-level constants (thresholds are visible + testable on purpose).
# ---------------------------------------------------------------------------

WINDOW_SEC: int = 300  # bucket width for silent/busy classification
DEFAULT_WINDOW_DAYS: int = 7

# Verdict (a) thresholds.
DRIFT_MIN_MEDIAN_ABS_TRAJECTORY: float = 0.03
DRIFT_VARIANCE_RATIO: float = 0.25

# Verdict (b) thresholds.
COACTIVATION_MIN_FRAC: float = 0.10
RESOURCE_PRESSURE_LEVEL: float = 0.3
RESOURCE_PRESSURE_MIN_FRAC: float = 0.05

# Verdict (b) saturation guards. Co-activation >= COACTIVATION_MIN_FRAC is
# trivially satisfied by an always-on saturated state (live 2026-07-15:
# coactivation_frac 0.9506 with 74% of ticks at 5-of-6 drives active and one
# drive dominant in 96% of ticks). GO must mean a functioning economy with
# churn and turn-taking, so a monoculture surfaces as its own verdict.
SATURATION_DOMINANT_SHARE: float = 0.90  # one drive dominant in >=90% of audits = no turn-taking
SATURATION_MIN_ACTIVE: int = 5  # 5 of the 6 drives simultaneously active = near-all-on
SATURATION_ALL_ACTIVE_FRAC: float = 0.75  # >=75% of audits in that near-all-on state

# Hard cap so no query result set grows unbounded.
MAX_ROWS: int = 500_000

STABLE_CONDITION = "stable"

# Output locations (I/O layer only).
OUTPUT_DIR = Path("/tmp/autonomy-gate")
REPORT_PATH = OUTPUT_DIR / "report.md"
CSV_PATH = OUTPUT_DIR / "before_after.csv"
PROGRESS_PATH = OUTPUT_DIR / "progress.log"

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"



# ===========================================================================
# Normalized in-memory records (pure layer boundary types)
# ===========================================================================


@dataclass
class SelfStateRecord:
    """A normalized ``SelfStateV1`` row, stripped to what the metrics need."""

    generated_at: datetime
    dimensions: dict[str, float]  # dimension_id -> score
    dimension_trajectory: dict[str, float]
    trajectory_condition: str
    overall_surprise: float
    resource_pressure: float


@dataclass
class BucketActivity:
    """Exogenous-input activity counters for a single fixed time bucket."""

    receipt_count: int = 0
    turn_count: int = 0


@dataclass
class SelfStateMetrics:
    """Q(a) drift metrics for one bucket class (silent or busy)."""

    row_count: int = 0
    mean_abs_trajectory: float = 0.0
    median_abs_trajectory: float = 0.0
    dim_score_variance: float = 0.0
    nonstable_frac: float = 0.0
    mean_surprise: float = 0.0


@dataclass
class ResourcePressureStats:
    """Q(b) resource-pressure distribution over the whole window."""

    row_count: int = 0
    median: Optional[float] = None
    p90: Optional[float] = None
    frac_gt_level: float = 0.0


@dataclass
class DriveStats:
    """Q(b) drive co-activation stats over the whole window."""

    record_count: int = 0
    coactivation_frac: float = 0.0
    concurrent_active_hist: dict[int, int] = field(default_factory=dict)
    # Fraction of audits with active_count >= SATURATION_MIN_ACTIVE (derived
    # from the same histogram -- the "near-all-on" saturation signal).
    all_active_frac: float = 0.0
    # dominant_drive -> audit count over the window (NULL dominants excluded
    # from the counts but still present in record_count's denominator).
    dominant_counts: dict[str, int] = field(default_factory=dict)
    # Share of the most common non-null dominant_drive over ALL records.
    top_dominant_share: float = 0.0


# ===========================================================================
# PURE LAYER — no I/O. Unit tests exercise only these.
# ===========================================================================


def _as_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def bucket_index(ts: datetime, window_start: datetime, window_sec: int = WINDOW_SEC) -> int:
    """Return the fixed-width bucket index for ``ts`` relative to window start.

    A timestamp exactly on a bucket boundary lands in the *later* bucket, i.e.
    ``window_start + window_sec`` is the first instant of bucket 1.
    """

    delta = (_as_utc(ts) - _as_utc(window_start)).total_seconds()
    return int(delta // window_sec)


def classify_bucket(activity: BucketActivity) -> str:
    """A bucket is 'silent' iff it saw no receipts and no turns, else 'busy'."""

    if activity.receipt_count == 0 and activity.turn_count == 0:
        return "silent"
    return "busy"


def build_bucket_activity(
    *,
    receipt_timestamps: Iterable[datetime],
    turn_timestamps: Iterable[datetime],
    window_start: datetime,
    window_sec: int = WINDOW_SEC,
) -> dict[int, BucketActivity]:
    """Fold receipt + turn timestamps into per-bucket activity counters."""

    buckets: dict[int, BucketActivity] = {}
    for ts in receipt_timestamps:
        idx = bucket_index(ts, window_start, window_sec)
        buckets.setdefault(idx, BucketActivity()).receipt_count += 1
    for ts in turn_timestamps:
        idx = bucket_index(ts, window_start, window_sec)
        buckets.setdefault(idx, BucketActivity()).turn_count += 1
    return buckets


def bucket_class_for(
    ts: datetime,
    buckets: dict[int, BucketActivity],
    window_start: datetime,
    window_sec: int = WINDOW_SEC,
) -> str:
    """Return 'silent' / 'busy' for the bucket that contains ``ts``.

    A bucket with no recorded activity is, by definition, silent.
    """

    idx = bucket_index(ts, window_start, window_sec)
    return classify_bucket(buckets.get(idx, BucketActivity()))


def split_self_states_by_class(
    records: Iterable[SelfStateRecord],
    buckets: dict[int, BucketActivity],
    window_start: datetime,
    window_sec: int = WINDOW_SEC,
) -> tuple[list[SelfStateRecord], list[SelfStateRecord]]:
    """Partition self-state rows into (silent_rows, busy_rows)."""

    silent: list[SelfStateRecord] = []
    busy: list[SelfStateRecord] = []
    for rec in records:
        if bucket_class_for(rec.generated_at, buckets, window_start, window_sec) == "silent":
            silent.append(rec)
        else:
            busy.append(rec)
    return silent, busy


def _abs_trajectory_for_row(rec: SelfStateRecord) -> float:
    """Per-row mean of |dimension_trajectory values| (0.0 when empty)."""

    values = list(rec.dimension_trajectory.values())
    if not values:
        return 0.0
    return statistics.fmean(abs(v) for v in values)


def per_row_abs_trajectory(records: Iterable[SelfStateRecord]) -> list[float]:
    return [_abs_trajectory_for_row(rec) for rec in records]


def _percentile(values: list[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile (q in [0,1]); None on empty input."""

    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * frac)


def dim_score_variance(records: Iterable[SelfStateRecord]) -> float:
    """Mean over dimensions of the variance of that dimension's score.

    Scores are read in generated_at order (consecutive rows). A dimension seen
    in fewer than two rows contributes variance 0. Empty input -> 0.0.
    """

    ordered = sorted(records, key=lambda r: _as_utc(r.generated_at))
    if not ordered:
        return 0.0
    by_dim: dict[str, list[float]] = {}
    for rec in ordered:
        for dim_id, score in rec.dimensions.items():
            by_dim.setdefault(dim_id, []).append(float(score))
    if not by_dim:
        return 0.0
    variances = [statistics.pvariance(scores) if len(scores) > 1 else 0.0 for scores in by_dim.values()]
    return statistics.fmean(variances)


def compute_self_state_metrics(records: list[SelfStateRecord]) -> SelfStateMetrics:
    """Q(a) drift metrics for a single bucket class. Empty -> all-zero."""

    if not records:
        return SelfStateMetrics()
    abs_traj = per_row_abs_trajectory(records)
    nonstable = sum(1 for r in records if r.trajectory_condition != STABLE_CONDITION)
    return SelfStateMetrics(
        row_count=len(records),
        mean_abs_trajectory=statistics.fmean(abs_traj),
        median_abs_trajectory=statistics.median(abs_traj),
        dim_score_variance=dim_score_variance(records),
        nonstable_frac=nonstable / len(records),
        mean_surprise=statistics.fmean(r.overall_surprise for r in records),
    )


def compute_resource_pressure_stats(
    records: list[SelfStateRecord],
    level: float = RESOURCE_PRESSURE_LEVEL,
) -> ResourcePressureStats:
    """Q(b) resource_pressure distribution over self-state rows. Empty -> zeros."""

    if not records:
        return ResourcePressureStats()
    values = [float(r.resource_pressure) for r in records]
    above = sum(1 for v in values if v >= level)
    return ResourcePressureStats(
        row_count=len(values),
        median=float(statistics.median(values)),
        p90=_percentile(values, 0.9),
        frac_gt_level=above / len(values),
    )


def drive_stats_from_histogram(hist: dict[int, int]) -> DriveStats:
    """Build Q(b) drive co-activation stats from an active-count histogram.

    ``hist`` maps active-drive-count -> number of audits with that count. Empty
    histogram -> all-zero DriveStats. Pure + unit-testable.
    """

    record_count = sum(hist.values())
    if record_count == 0:
        return DriveStats()
    coactive = sum(n for k, n in hist.items() if k >= 2)
    all_active = sum(n for k, n in hist.items() if k >= SATURATION_MIN_ACTIVE)
    return DriveStats(
        record_count=record_count,
        coactivation_frac=coactive / record_count,
        concurrent_active_hist=dict(hist),
        all_active_frac=all_active / record_count,
    )


def parse_dominant_rows(rows: Iterable[Any]) -> dict[str, int]:
    """Parse ``SELECT dominant_drive, count(*) ... GROUP BY dominant_drive``
    rows into ``{dominant_drive: audits}``.

    A NULL dominant_drive is allowed (rows where no drive dominated): it is
    excluded from the returned counts but its rows still sit in the share
    denominator via the histogram's ``record_count`` (live shape check:
    84 of 1727 rows had NULL dominant). Malformed rows (short tuples,
    non-numeric counts) and negative counts are skipped. Never raises.
    Pure + unit-testable, mirrors ``parse_postgres_histogram_rows``.
    """

    out: dict[str, int] = {}
    for row in rows:
        try:
            key = row[0]
            audits = int(row[1])
        except (TypeError, ValueError, IndexError, OverflowError):
            continue
        if audits < 0:
            continue
        if key is None:
            continue
        out[str(key)] = audits
    return out


def apply_dominant_counts(stats: DriveStats, dominant_counts: dict[str, int]) -> DriveStats:
    """Return a copy of ``stats`` with dominance fields filled in.

    ``top_dominant_share`` uses ``stats.record_count`` (ALL audits in the
    window, NULL dominants included) as the denominator, not the sum of the
    non-null dominant counts -- a drive dominating every row that HAS a
    dominant must not be inflated by dropping the NULL rows. Empty counts or
    zero records -> share 0.0 (the dominance clause simply can't fire).
    Pure + unit-testable.
    """

    share = 0.0
    if dominant_counts and stats.record_count > 0:
        share = max(dominant_counts.values()) / stats.record_count
    return replace(
        stats,
        dominant_counts=dict(dominant_counts),
        top_dominant_share=share,
    )


def parse_postgres_histogram_rows(rows: Iterable[Any]) -> dict[int, int]:
    """Parse ``SELECT active_count, count(*) ... GROUP BY active_count`` rows
    into ``{active_count: audits}``.

    Malformed rows (short tuples, non-numeric values) are skipped; negative
    values are ALSO skipped, not clamped -- this writer derives
    ``active_count = len(active_drives) >= 0``, so a negative can only come
    from foreign/manual writes, and clamping it to 0 could overwrite the
    real 0 bucket (destroying genuine audits from the denominator). Never
    raises. Pure + unit-testable.
    """

    out: dict[int, int] = {}
    for row in rows:
        try:
            active_count = int(row[0])
            audits = int(row[1])
        except (TypeError, ValueError, IndexError, OverflowError):
            continue
        if active_count < 0 or audits < 0:
            continue
        out[active_count] = audits
    return out


def is_undefined_table_error(exc: BaseException) -> bool:
    """True iff ``exc`` is Postgres "relation does not exist" (SQLSTATE 42P01).

    Checked structurally (``pgcode`` attribute / exception class name) so the
    caller can distinguish "the drive_audits table hasn't been created yet by
    the parallel writer track" from any other query failure WITHOUT importing
    psycopg2 at module scope. Message sniffing is a last-resort fallback for
    driver objects that carry neither. Pure + unit-testable.
    """

    if getattr(exc, "pgcode", None) == "42P01":
        return True
    if type(exc).__name__ == "UndefinedTable":
        return True
    msg = str(exc).lower()
    return "relation" in msg and "does not exist" in msg


UNMEASURABLE = "UNMEASURABLE"
SATURATED = "SATURATED"


def verdict_drift(silent: SelfStateMetrics, busy: SelfStateMetrics) -> str:
    """Verdict (a): GO iff silent-bucket self-state genuinely drifts.

    UNMEASURABLE iff there is no self-state data at all in the window (both
    classes empty) -- a dead/unreachable source resolves every metric to 0.0,
    which would otherwise read as a real "flat, NO-GO" behavioral finding
    instead of "we didn't measure anything." Never silently downgrade a dead
    sensor into a behavioral verdict.

    Otherwise, GO iff, in SILENT buckets:
      median(per-row |trajectory|) >= DRIFT_MIN_MEDIAN_ABS_TRAJECTORY
      AND silent dim_score_variance >= DRIFT_VARIANCE_RATIO * busy variance.
    When busy variance is 0, the ratio test passes iff silent variance > 0.
    """

    if silent.row_count == 0 and busy.row_count == 0:
        return UNMEASURABLE
    if silent.median_abs_trajectory < DRIFT_MIN_MEDIAN_ABS_TRAJECTORY:
        return "NO-GO"
    if busy.dim_score_variance == 0.0:
        variance_ok = silent.dim_score_variance > 0.0
    else:
        variance_ok = silent.dim_score_variance >= DRIFT_VARIANCE_RATIO * busy.dim_score_variance
    return "GO" if variance_ok else "NO-GO"


def retention_caveat(
    source_name: str, oldest_available: Optional[datetime], window_start: datetime
) -> Optional[str]:
    """Caveat when a source's actual retention doesn't reach back to
    window_start -- the exact "busy/silent classification validity bound"
    gap: a longer window silently under-covers older buckets with zero rows
    (not measured absence) with nothing in the report saying so. Returns
    None when the source covers the full window, or when oldest_available
    itself is unknown (a separate unavailability caveat already covers that
    case). Pure, unit-testable.
    """

    if oldest_available is None:
        return None
    oldest = _as_utc(oldest_available)
    start = _as_utc(window_start)
    if oldest <= start:
        return None
    gap_hours = (oldest - start).total_seconds() / 3600.0
    return (
        f"{source_name} retention only covers back to {oldest.isoformat()}, "
        f"{gap_hours:.1f}h short of the requested window start ({start.isoformat()}) "
        "-- classification/metrics for the uncovered period reflect zero rows, "
        "not measured absence"
    )


def verdict_economy(drive: DriveStats, pressure: ResourcePressureStats) -> str:
    """Verdict (b): GO iff drives co-activate AND resource_pressure rises.

    UNMEASURABLE iff EITHER input has zero rows -- the rule reads both
    drive.coactivation_frac (Postgres drive_audits) and
    pressure.frac_gt_level (Postgres self-state), two independent sources. Guarding only one (e.g. only
    drive.record_count) would let the other silently degrade to 0.0 and
    resolve as a real "NO-GO" string -- exactly the failure mode this
    function exists to prevent, just left open on whichever input isn't
    checked.

    SATURATED iff the co-activation bar IS met but the underlying state is a
    monoculture: one drive dominant in >= SATURATION_DOMINANT_SHARE of audits
    (no turn-taking), OR >= SATURATION_ALL_ACTIVE_FRAC of audits have
    >= SATURATION_MIN_ACTIVE drives simultaneously active (always-on). This is
    the same "instrument blesses a degenerate state" failure class the
    UNMEASURABLE guard fixed for zero rows -- live 2026-07-15 the gate read GO
    with coactivation_frac 0.9506 while dominant_drive was "relational" in 96%
    of ticks and 74% of ticks sat at 5-of-6 drives active. Co-activation alone
    cannot distinguish an economy (churn, turn-taking) from saturation, so
    saturation gets its own honest verdict. If dominance data degraded
    (empty dominant_counts) the dominance clause can't fire, but the
    histogram-only all_active_frac clause still can.

    Precedence: UNMEASURABLE > SATURATED > NO-GO/GO (a coactivation shortfall
    still short-circuits to NO-GO before the saturation check -- saturation is
    only a meaningful diagnosis of a coactivation bar that was met).

    Otherwise, GO iff coactivation_frac >= COACTIVATION_MIN_FRAC
      AND frac_gt(resource_pressure >= level) >= RESOURCE_PRESSURE_MIN_FRAC.
    """

    if drive.record_count == 0 or pressure.row_count == 0:
        return UNMEASURABLE
    if drive.coactivation_frac < COACTIVATION_MIN_FRAC:
        return "NO-GO"
    if (
        drive.top_dominant_share >= SATURATION_DOMINANT_SHARE
        or drive.all_active_frac >= SATURATION_ALL_ACTIVE_FRAC
    ):
        return SATURATED
    if pressure.frac_gt_level < RESOURCE_PRESSURE_MIN_FRAC:
        return "NO-GO"
    return "GO"


# ===========================================================================
# I/O LAYER — psycopg2 read-only. Never exercised by unit tests.
# Every adapter degrades to empty / None on absent input; none raise on
# missing data or columns.
# ===========================================================================


def open_readonly_connection(dsn: str):
    """Open a psycopg2 connection and force a read-only session.

    Refuses to return a connection that is not read-only. Returns None on any
    connection failure (degrade, do not crash the whole run).
    """

    try:
        import psycopg2  # lazy import so the module imports cleanly for tests
    except Exception:  # pragma: no cover - environment without psycopg2
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


def _coerce_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return _as_utc(value)
    if isinstance(value, str):
        try:
            return _as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except Exception:
            return None
    return None


def _normalize_self_state(payload: dict, generated_at: datetime) -> SelfStateRecord:
    """Map a self_state_json payload to a SelfStateRecord, degrading per-field."""

    dims_raw = payload.get("dimensions") or {}
    dimensions: dict[str, float] = {}
    if isinstance(dims_raw, dict):
        for dim_id, dim_val in dims_raw.items():
            if isinstance(dim_val, dict) and "score" in dim_val:
                try:
                    dimensions[dim_id] = float(dim_val["score"])
                except (TypeError, ValueError):
                    continue
    trajectory_raw = payload.get("dimension_trajectory") or {}
    trajectory: dict[str, float] = {}
    if isinstance(trajectory_raw, dict):
        for k, v in trajectory_raw.items():
            try:
                trajectory[k] = float(v)
            except (TypeError, ValueError):
                continue
    resource_pressure = dimensions.get("resource_pressure", 0.0)
    try:
        surprise = float(payload.get("overall_surprise", 0.0) or 0.0)
    except (TypeError, ValueError):
        surprise = 0.0
    return SelfStateRecord(
        generated_at=generated_at,
        dimensions=dimensions,
        dimension_trajectory=trajectory,
        trajectory_condition=str(payload.get("trajectory_condition", "unknown") or "unknown"),
        overall_surprise=surprise,
        resource_pressure=resource_pressure,
    )


def fetch_self_state_records(conn, since: datetime, max_rows: int = MAX_ROWS) -> tuple[list[SelfStateRecord], bool]:
    """Fetch normalized self-state rows since ``since``. Returns (rows, truncated)."""

    if conn is None:
        return [], False
    out: list[SelfStateRecord] = []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT self_state_json, generated_at
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
    for raw_json, generated_at in rows:
        gen = _coerce_dt(generated_at)
        if gen is None:
            continue
        payload = raw_json
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                continue
        if not isinstance(payload, dict):
            continue
        out.append(_normalize_self_state(payload, gen))
    return out, len(rows) >= max_rows


def fetch_receipt_timestamps(conn, since: datetime, max_rows: int = MAX_ROWS) -> tuple[list[datetime], bool]:
    """Fetch reduction-receipt timestamps (the 'exogenous input happened' signal)."""

    if conn is None:
        return [], False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT created_at
                FROM substrate_reduction_receipts
                WHERE created_at >= %s
                ORDER BY created_at ASC
                LIMIT %s
                """,
                (since, max_rows),
            )
            rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch substrate_reduction_receipts", exc_info=True)
        return [], False
    out = [dt for dt in (_coerce_dt(r[0]) for r in rows) if dt is not None]
    return out, len(rows) >= max_rows


def fetch_earliest_self_state_ts(conn) -> Optional[datetime]:
    """Earliest substrate_self_state row overall (not window-bounded) -- the
    real retention floor for verdict (a)'s primary source. Degrades to None
    on any failure or an empty table; never raises.
    """

    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT MIN(generated_at) FROM substrate_self_state")
            row = cur.fetchone()
    except Exception:
        logger.error("failed to fetch earliest substrate_self_state timestamp", exc_info=True)
        return None
    return _coerce_dt(row[0]) if row else None


def fetch_earliest_receipt_ts(conn) -> Optional[datetime]:
    """Earliest substrate_reduction_receipts row overall -- the retention
    floor for the busy/silent classification signal. Degrades to None on any
    failure or an empty table; never raises.
    """

    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT MIN(created_at) FROM substrate_reduction_receipts")
            row = cur.fetchone()
    except Exception:
        logger.error("failed to fetch earliest substrate_reduction_receipts timestamp", exc_info=True)
        return None
    return _coerce_dt(row[0]) if row else None


def fetch_earliest_drive_audit_ts(conn) -> Optional[datetime]:
    """Earliest drive_audits row by ``COALESCE(observed_at, created_at)`` --
    the retention floor for the Postgres drive co-activation source. The
    table was only born 2026-07-15, so a long window (e.g. the original
    NO-GO's 120 days) can have rows for only its tail: without this floor
    feeding ``retention_caveat``, that tail would be silently presented as
    the full window. Degrades to None on any failure, including the table
    not existing yet; never raises.
    """

    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT MIN(COALESCE(observed_at, created_at)) FROM drive_audits")
            row = cur.fetchone()
    except Exception as exc:
        if not is_undefined_table_error(exc):
            logger.error("failed to fetch earliest drive_audits timestamp", exc_info=True)
        return None
    return _coerce_dt(row[0]) if row else None


def fetch_drive_stats_postgres(conn, window_start: datetime) -> tuple[DriveStats, str]:
    """Read the drive co-activation histogram from the Postgres ``drive_audits``
    table (written by orion-sql-writer; the successor source to the frozen
    Fuseki DriveAudit graph).

    ``active_count`` is already derived at write time, so the histogram is a
    single server-side ``GROUP BY active_count`` — no JSONB parsing here. A
    second cheap ``GROUP BY dominant_drive`` on the same connection fills the
    dominance fields for the SATURATED verdict; if it fails, the histogram
    result stands with empty ``dominant_counts`` and a note.
    Windowing follows the table contract:
    ``window_start <= COALESCE(observed_at, created_at) < window_end``, where
    ``window_end`` is captured ONCE and shared by both queries — on an
    autocommit connection they are separate statements, and without a shared
    upper bound rows landing between them would appear in ``dominant_counts``
    but not in ``record_count`` (the share denominator), letting
    ``top_dominant_share`` drift past its ≤1 contract (review finding).

    Reuses the script's existing read-only connection. Returns
    ``(DriveStats, note)`` and degrades to ``(DriveStats(), note)`` — never
    raises. A missing table (the parallel writer track may not have created it
    yet) is reported as such, distinct from "table present but 0 rows in
    window" and from any other query failure.
    """

    if conn is None:
        return DriveStats(), "postgres drive_audits unavailable: no read-only postgres connection"
    window_end = datetime.now(timezone.utc)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT active_count, COUNT(*)
                FROM drive_audits
                WHERE COALESCE(observed_at, created_at) >= %s
                  AND COALESCE(observed_at, created_at) < %s
                GROUP BY active_count
                """,
                (window_start, window_end),
            )
            rows = cur.fetchall()
    except Exception as exc:
        if is_undefined_table_error(exc):
            logger.warning("drive_audits table does not exist yet; postgres drive source unavailable")
            return DriveStats(), (
                "postgres drive_audits table does not exist yet "
                "(sql-writer track not deployed?); source unavailable"
            )
        logger.error("failed to fetch drive_audits histogram", exc_info=True)
        return DriveStats(), "postgres drive_audits query failed; source unavailable"
    hist = parse_postgres_histogram_rows(rows)
    stats = drive_stats_from_histogram(hist)
    if stats.record_count == 0:
        return stats, (
            f"postgres drive_audits: table present but 0 rows in window since "
            f"{window_start.isoformat()}"
        )
    # Second cheap query on the same connection: dominant-drive counts feeding
    # the SATURATED dominance clause. Same degrade posture as the histogram
    # query -- a dominance failure must NOT discard the histogram result, it
    # just leaves dominant_counts empty (the all_active_frac clause still
    # works histogram-only) with an honest note.
    dominance_note = ""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT dominant_drive, COUNT(*)
                FROM drive_audits
                WHERE COALESCE(observed_at, created_at) >= %s
                  AND COALESCE(observed_at, created_at) < %s
                GROUP BY dominant_drive
                """,
                (window_start, window_end),
            )
            dominant_rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch drive_audits dominant-drive counts", exc_info=True)
        dominant_rows = None
        dominance_note = (
            "; dominant_drive query failed -- dominance saturation check "
            "unavailable (histogram result stands)"
        )
    if dominant_rows is not None:
        stats = apply_dominant_counts(stats, parse_dominant_rows(dominant_rows))
    return stats, (
        f"drive source: postgres drive_audits ({stats.record_count} rows since "
        f"{window_start.isoformat()}){dominance_note}"
    )


# ===========================================================================
# Progress + report writers (I/O layer)
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

    def emit(self, title: str, *, percent: float, processed: int, total: int, anomalies: int) -> None:
        elapsed = max(time.monotonic() - self._start, 1e-6)
        rate = processed / elapsed
        line = (
            f"{datetime.now(timezone.utc).isoformat()} | {title} | "
            f"{percent:5.1f}% | rows={processed}/{total} | "
            f"rate={rate:.1f}/s | anomalies={anomalies}"
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


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def write_before_after_csv(path: Path, silent: SelfStateMetrics, busy: SelfStateMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "silent", "busy"])
        writer.writerow(["row_count", silent.row_count, busy.row_count])
        writer.writerow(["mean_abs_trajectory", _fmt(silent.mean_abs_trajectory), _fmt(busy.mean_abs_trajectory)])
        writer.writerow(["median_abs_trajectory", _fmt(silent.median_abs_trajectory), _fmt(busy.median_abs_trajectory)])
        writer.writerow(["dim_score_variance", _fmt(silent.dim_score_variance), _fmt(busy.dim_score_variance)])
        writer.writerow(["nonstable_frac", _fmt(silent.nonstable_frac), _fmt(busy.nonstable_frac)])
        writer.writerow(["mean_surprise", _fmt(silent.mean_surprise), _fmt(busy.mean_surprise)])


def render_report(
    *,
    window_label: str,
    window_start: datetime,
    window_end: datetime,
    silent: SelfStateMetrics,
    busy: SelfStateMetrics,
    drive: DriveStats,
    drive_source: str,
    pressure: ResourcePressureStats,
    verdict_a: str,
    verdict_b: str,
    caveats: list[str],
) -> str:
    hist = ", ".join(f"{k}:{v}" for k, v in sorted(drive.concurrent_active_hist.items())) or "(none)"
    dominants = ", ".join(
        f"{name}:{count}"
        for name, count in sorted(drive.dominant_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ) or "(none)"
    if drive.dominant_counts:
        top_name = max(drive.dominant_counts.items(), key=lambda kv: kv[1])[0]
        top_dominant = f"{top_name} (share {_fmt(drive.top_dominant_share)} of all audits)"
    else:
        top_dominant = "(none -- dominance data empty or unavailable)"
    lines = [
        "# Autonomy Origination Measurement Gate",
        "",
        "Read-only measurement. No writes, no events, no flag changes.",
        "",
        f"- Window: last {window_label} ({window_start.isoformat()} -> {window_end.isoformat()})",
        f"- Bucket width: {WINDOW_SEC}s",
        "",
        "## Verdicts",
        "",
        f"- (a) Endogenous drift during exogenous silence: **{verdict_a}**",
        f"- (b) Internal economy (drive co-activation + resource pressure): **{verdict_b}**",
        "",
        "## Q(a) self-state drift (silent vs busy buckets)",
        "",
        "| metric | silent | busy |",
        "| --- | --- | --- |",
        f"| row_count | {silent.row_count} | {busy.row_count} |",
        f"| mean_abs_trajectory | {_fmt(silent.mean_abs_trajectory)} | {_fmt(busy.mean_abs_trajectory)} |",
        f"| median_abs_trajectory | {_fmt(silent.median_abs_trajectory)} | {_fmt(busy.median_abs_trajectory)} |",
        f"| dim_score_variance | {_fmt(silent.dim_score_variance)} | {_fmt(busy.dim_score_variance)} |",
        f"| nonstable_frac | {_fmt(silent.nonstable_frac)} | {_fmt(busy.nonstable_frac)} |",
        f"| mean_surprise | {_fmt(silent.mean_surprise)} | {_fmt(busy.mean_surprise)} |",
        "",
        "Rule (a) GO iff silent median_abs_trajectory >= "
        f"{DRIFT_MIN_MEDIAN_ABS_TRAJECTORY} AND silent dim_score_variance >= "
        f"{DRIFT_VARIANCE_RATIO} * busy dim_score_variance "
        "(busy variance 0 -> pass iff silent variance > 0).",
        "",
        "## Q(b) internal economy (whole window)",
        "",
        f"- drive audit source: {drive_source}",
        f"- drive audit records: {drive.record_count}",
        f"- coactivation_frac (active_count >= 2): {_fmt(drive.coactivation_frac)}",
        f"- concurrent_active_hist: {hist}",
        f"- all_active_frac (active_count >= {SATURATION_MIN_ACTIVE}): {_fmt(drive.all_active_frac)}",
        f"- dominant_drive counts: {dominants}",
        f"- top dominant drive: {top_dominant}",
        f"- resource_pressure rows: {pressure.row_count}",
        f"- resource_pressure median: {_fmt(pressure.median)}",
        f"- resource_pressure p90: {_fmt(pressure.p90)}",
        f"- resource_pressure frac >= {RESOURCE_PRESSURE_LEVEL}: {_fmt(pressure.frac_gt_level)}",
        "",
        "Rule (b) GO iff coactivation_frac >= "
        f"{COACTIVATION_MIN_FRAC} AND resource_pressure frac >= "
        f"{RESOURCE_PRESSURE_LEVEL} is >= {RESOURCE_PRESSURE_MIN_FRAC}.",
        f"Rule (b) {SATURATED} iff the co-activation bar is met but "
        f"top_dominant_share >= {SATURATION_DOMINANT_SHARE} OR "
        f"all_active_frac >= {SATURATION_ALL_ACTIVE_FRAC}.",
        "",
    ]
    if verdict_b == SATURATED:
        lines.extend([
            f"Verdict {SATURATED} means the co-activation bar was met only by an "
            "always-on monoculture (one drive dominating nearly every audit and/or "
            "nearly all drives simultaneously active), not by a functioning economy "
            "with churn and turn-taking.",
        ])
        if pressure.frac_gt_level < RESOURCE_PRESSURE_MIN_FRAC:
            # Review note: SATURATED returns before the pressure check, so
            # without this line a reader could infer "fix saturation -> GO"
            # while the pressure bar would still fail.
            lines.append(
                "Note: the resource_pressure bar is ALSO currently unmet -- "
                "resolving saturation alone would read NO-GO, not GO."
            )
        lines.append("")
    lines.extend([
        "## Coverage caveats",
        "",
    ])
    if caveats:
        lines.extend(f"- {c}" for c in caveats)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


# ===========================================================================
# Orchestration (I/O layer)
# ===========================================================================


def run(window: timedelta, window_label: str) -> int:
    now = datetime.now(timezone.utc)
    window_start = now - window
    dsn = os.environ.get("POSTGRES_URI", DEFAULT_POSTGRES_URI)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = ProgressLog(PROGRESS_PATH)
    caveats: list[str] = []
    anomalies = 0

    progress.emit("connect", percent=0.0, processed=0, total=0, anomalies=anomalies)
    conn = open_readonly_connection(dsn)
    if conn is None:
        caveats.append("postgres unavailable or not read-only; self-state/receipt metrics empty")
        anomalies += 1

    self_states, ss_trunc = fetch_self_state_records(conn, window_start)
    if ss_trunc:
        caveats.append(f"self-state rows truncated at MAX_ROWS={MAX_ROWS}")
        anomalies += 1
    ss_retention_note = retention_caveat(
        "substrate_self_state", fetch_earliest_self_state_ts(conn), window_start
    )
    if ss_retention_note:
        caveats.append(ss_retention_note)
    progress.emit(
        "self_state loaded", percent=40.0, processed=len(self_states), total=len(self_states), anomalies=anomalies
    )

    receipts, r_trunc = fetch_receipt_timestamps(conn, window_start)
    if r_trunc:
        caveats.append(f"reduction receipts truncated at MAX_ROWS={MAX_ROWS}")
        anomalies += 1
    receipt_retention_note = retention_caveat(
        "substrate_reduction_receipts", fetch_earliest_receipt_ts(conn), window_start
    )
    if receipt_retention_note:
        caveats.append(receipt_retention_note)
    progress.emit(
        "receipts loaded", percent=60.0, processed=len(receipts), total=len(receipts), anomalies=anomalies
    )

    # Turns: no cheap turn store located in the repo; do not invent one.
    turn_timestamps: list[datetime] = []
    caveats.append("turns unavailable, turn_count=0 (no turn store wired into this measurement)")

    # Drive co-activation reads ONLY the live drive_audits table. The frozen
    # Fuseki DriveAudit graph (dead since 2026-06-19, flat-pinned-era data) is
    # deliberately NOT read -- windows the table does not cover are honestly
    # UNMEASURABLE, never backfilled from a dead sensor.
    drive_stats, drive_note = fetch_drive_stats_postgres(conn, window_start)
    caveats.append(drive_note)
    drive_source = (
        f"postgres drive_audits ({drive_stats.record_count} rows)"
        if drive_stats.record_count > 0
        else "postgres drive_audits (0 rows in window)"
    )
    drive_retention_note = retention_caveat(
        "drive_audits", fetch_earliest_drive_audit_ts(conn), window_start
    )
    if drive_retention_note:
        caveats.append(drive_retention_note)
    if drive_stats.record_count == 0:
        anomalies += 1
    progress.emit(
        "drive audit loaded",
        percent=80.0,
        processed=drive_stats.record_count,
        total=drive_stats.record_count,
        anomalies=anomalies,
    )

    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass

    # ---- Pure computation ----
    buckets = build_bucket_activity(
        receipt_timestamps=receipts,
        turn_timestamps=turn_timestamps,
        window_start=window_start,
    )
    silent_rows, busy_rows = split_self_states_by_class(self_states, buckets, window_start)
    silent_metrics = compute_self_state_metrics(silent_rows)
    busy_metrics = compute_self_state_metrics(busy_rows)
    pressure = compute_resource_pressure_stats(self_states)

    verdict_a = verdict_drift(silent_metrics, busy_metrics)
    verdict_b = verdict_economy(drive_stats, pressure)

    # ---- Emit artifacts ----
    write_before_after_csv(CSV_PATH, silent_metrics, busy_metrics)
    report = render_report(
        window_label=window_label,
        window_start=window_start,
        window_end=now,
        silent=silent_metrics,
        busy=busy_metrics,
        drive=drive_stats,
        drive_source=drive_source,
        pressure=pressure,
        verdict_a=verdict_a,
        verdict_b=verdict_b,
        caveats=caveats,
    )
    try:
        REPORT_PATH.write_text(report, encoding="utf-8")
    except Exception:
        logger.error("failed to write report", exc_info=True)

    total_rows = len(self_states) + len(receipts) + drive_stats.record_count
    progress.emit(
        "done",
        percent=100.0,
        processed=total_rows,
        total=total_rows,
        anomalies=anomalies,
    )
    progress.close()

    print(report)
    print(f"\nverdict (a) endogenous drift : {verdict_a}")
    print(f"verdict (b) internal economy : {verdict_b}")
    print(f"\nartifacts: {REPORT_PATH}, {CSV_PATH}, {PROGRESS_PATH}")
    # Exit code distinguishes "the instrument didn't work" (2) from a
    # completed measurement, whichever way it came out (0) -- a caller
    # (cron, CI, a human) must not have to parse the report text to tell
    # a dead sensor from a real GO/NO-GO. SATURATED is a completed
    # measurement (the sensor worked; the economy is degenerate), so it
    # exits 0 exactly like NO-GO -- it is NOT a pass, and nothing here
    # or downstream may treat it as GO.
    if UNMEASURABLE in (verdict_a, verdict_b):
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only autonomy-origination measurement gate.")
    window_group = parser.add_mutually_exclusive_group()
    window_group.add_argument(
        "--window-days",
        type=int,
        default=None,
        help=f"analysis window in days (default {DEFAULT_WINDOW_DAYS} if neither flag given)",
    )
    window_group.add_argument(
        "--window-hours",
        type=float,
        default=None,
        help="analysis window in hours (mutually exclusive with --window-days)",
    )
    return parser


def resolve_window(args: argparse.Namespace) -> tuple[timedelta, str]:
    """Turn parsed args into (timedelta, human label). Pure, unit-testable."""

    if args.window_hours is not None:
        return timedelta(hours=args.window_hours), f"{args.window_hours:g}h"
    days = args.window_days if args.window_days is not None else DEFAULT_WINDOW_DAYS
    return timedelta(days=days), f"{days} day(s)"


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_arg_parser().parse_args(argv)
    window, window_label = resolve_window(args)
    return run(window, window_label)


if __name__ == "__main__":  # pragma: no cover - guarded so import stays I/O-free
    raise SystemExit(main())
