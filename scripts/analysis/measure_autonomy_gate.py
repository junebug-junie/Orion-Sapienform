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
  session and read the durable drive-audit time-series via a read-only Fuseki
  SPARQL SELECT. Every adapter degrades gracefully to empty / None on absent
  input and NEVER raises on missing data or columns.

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
from dataclasses import dataclass, field
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

# Hard cap so no query result set grows unbounded.
MAX_ROWS: int = 500_000

STABLE_CONDITION = "stable"

# Output locations (I/O layer only).
OUTPUT_DIR = Path("/tmp/autonomy-gate")
REPORT_PATH = OUTPUT_DIR / "report.md"
CSV_PATH = OUTPUT_DIR / "before_after.csv"
PROGRESS_PATH = OUTPUT_DIR / "progress.log"

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"

# Durable drive co-activation source. DriveAuditV1 is ephemeral on the bus
# (redis pub/sub, no history), but orion-rdf-writer persists every audit to
# Fuseki as a `orion:DriveAudit` whose active-drive signal lives at the
# assessment level: DriveAudit --orion:hasDriveAssessment--> DriveAssessment,
# each assessment carrying a boolean `orion:driveActive`. An audit's
# active-drive count == number of its assessments with driveActive == true.
# (An older `orion:highlightsActiveDrive` projection is equivalent — one triple
# per active drive — but we read the assessment-level boolean as the primary
# schema.) We aggregate this server-side into a co-activation histogram so
# Fuseki returns ~7 rows regardless of window size, instead of transferring
# hundreds of thousands of per-audit rows (which timed out on large windows).
ORION_NS = "http://conjourney.net/orion#"
AUTONOMY_DRIVES_GRAPH = "http://conjourney.net/graph/autonomy/drives"
DEFAULT_FUSEKI_QUERY_URL = "http://orion-athena-fuseki:3030/orion/query"
# Server-side aggregation measured ~16s over full history; 60s gives headroom.
FUSEKI_TIMEOUT_SEC = 60.0


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
    return DriveStats(
        record_count=record_count,
        coactivation_frac=coactive / record_count,
        concurrent_active_hist=dict(hist),
    )


def build_drive_coactivation_histogram_sparql(
    since: datetime, graph_uri: str = AUTONOMY_DRIVES_GRAPH
) -> str:
    """SPARQL: co-activation histogram (active-count -> #audits) since ``since``.

    The inner query reduces per audit: each ``?audit``'s ``?activeCount`` is
    ``COUNT(DISTINCT ?activeDa)`` where ``?activeDa`` is bound ONLY to that
    audit's active assessments — the pattern matches the boolean literal
    directly (``?activeDa orion:driveActive true``), so no per-assessment IF is
    needed and the count is bounded by the number of drive keys (max 6).

    Two subtleties, both verified live against Jena/Fuseki:

    * ``orion:timestamp`` is MULTI-VALUED (up to 8 per audit). It MUST NOT be
      bound in the aggregation group: binding ``?ts`` alongside the assessment
      join cross-products (5 timestamps x 6 active assessments = 30 counted),
      producing impossible active counts and inflating co-activation. Instead
      the window is applied with ``FILTER EXISTS { ?audit orion:timestamp ?ts .
      FILTER(?ts >= since) }`` — an audit is in-window iff ANY of its timestamps
      is ``>= since``, which is correct for a multi-valued timestamp and cannot
      fan out the count.
    * The assessment join is OPTIONAL, so an assessment-less / all-inactive
      audit has ``?activeDa`` unbound and ``COUNT(DISTINCT ?activeDa) = 0``,
      landing it in the ``activeCount = 0`` bucket. ``COUNT(DISTINCT ...)`` also
      dedupes any duplicate assessment edges.

    The outer query bins the per-audit counts into a histogram, so Fuseki
    returns only ~7 rows regardless of window size — no per-audit transfer of
    hundreds of thousands of rows, no timeout. Pure (returns a string) so it is
    unit-testable without a SPARQL endpoint.
    """

    since_iso = _as_utc(since).isoformat()
    return (
        f"PREFIX orion: <{ORION_NS}>\n"
        "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
        "SELECT ?activeCount (COUNT(*) AS ?audits) WHERE {\n"
        "  SELECT ?audit (COUNT(DISTINCT ?activeDa) AS ?activeCount) WHERE {\n"
        f"    GRAPH <{graph_uri}> {{\n"
        "      ?audit a orion:DriveAudit .\n"
        f'      FILTER EXISTS {{ ?audit orion:timestamp ?ts . FILTER(?ts >= "{since_iso}"^^xsd:dateTime) }}\n'
        "      OPTIONAL { ?audit orion:hasDriveAssessment ?activeDa . ?activeDa orion:driveActive true . }\n"
        "    }\n"
        "  } GROUP BY ?audit\n"
        "} GROUP BY ?activeCount ORDER BY ?activeCount"
    )


def parse_sparql_histogram_bindings(bindings: Iterable[dict]) -> dict[int, int]:
    """Parse SPARQL-results-JSON histogram bindings into ``{activeCount: audits}``.

    Each binding carries ``activeCount`` and ``audits``, both SUM/COUNT literals
    Fuseki may return as integer or decimal strings (e.g. ``"2"`` or ``"2.0"``).
    Malformed rows (missing/garbage values) are skipped; never raises. Pure +
    unit-testable.
    """

    out: dict[int, int] = {}
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        cnt_node = binding.get("activeCount")
        aud_node = binding.get("audits")
        raw_cnt = cnt_node.get("value") if isinstance(cnt_node, dict) else None
        raw_aud = aud_node.get("value") if isinstance(aud_node, dict) else None
        try:
            active_count = int(float(raw_cnt))
            audits = int(float(raw_aud))
        except (TypeError, ValueError, OverflowError):
            continue
        out[max(0, active_count)] = max(0, audits)
    return out


def verdict_drift(silent: SelfStateMetrics, busy: SelfStateMetrics) -> str:
    """Verdict (a): GO iff silent-bucket self-state genuinely drifts.

    GO iff, in SILENT buckets:
      median(per-row |trajectory|) >= DRIFT_MIN_MEDIAN_ABS_TRAJECTORY
      AND silent dim_score_variance >= DRIFT_VARIANCE_RATIO * busy variance.
    When busy variance is 0, the ratio test passes iff silent variance > 0.
    """

    if silent.median_abs_trajectory < DRIFT_MIN_MEDIAN_ABS_TRAJECTORY:
        return "NO-GO"
    if busy.dim_score_variance == 0.0:
        variance_ok = silent.dim_score_variance > 0.0
    else:
        variance_ok = silent.dim_score_variance >= DRIFT_VARIANCE_RATIO * busy.dim_score_variance
    return "GO" if variance_ok else "NO-GO"


def verdict_economy(drive: DriveStats, pressure: ResourcePressureStats) -> str:
    """Verdict (b): GO iff drives co-activate AND resource_pressure rises.

    GO iff coactivation_frac >= COACTIVATION_MIN_FRAC
      AND frac_gt(resource_pressure >= level) >= RESOURCE_PRESSURE_MIN_FRAC.
    """

    if drive.coactivation_frac < COACTIVATION_MIN_FRAC:
        return "NO-GO"
    if pressure.frac_gt_level < RESOURCE_PRESSURE_MIN_FRAC:
        return "NO-GO"
    return "GO"


# ===========================================================================
# I/O LAYER — psycopg2 read-only + bus XRANGE. Never exercised by unit tests.
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


def fetch_drive_stats(
    query_url: str,
    since: datetime,
    graph_uri: str = AUTONOMY_DRIVES_GRAPH,
    timeout_sec: float = FUSEKI_TIMEOUT_SEC,
) -> tuple[DriveStats, str]:
    """Read durable drive co-activation stats from Fuseki via read-only SPARQL.

    The bus audit channel (``orion:memory:drives:audit``) is redis pub/sub with
    no replayable history, so it is useless for a backward-looking measurement.
    orion-rdf-writer, however, persists every DriveAuditV1 to the Fuseki
    autonomy/drives graph as a timestamped ``orion:DriveAudit`` whose active
    signal lives at the assessment level: ``orion:hasDriveAssessment`` ->
    ``DriveAssessment`` with a boolean ``orion:driveActive`` — a real historical
    time-series. We aggregate ``driveActive = true`` counts per audit into a
    co-activation histogram *server-side*, so Fuseki returns ~7 rows regardless
    of window size (transferring all per-audit rows blows the timeout on large
    windows).

    Read-only SPARQL SELECT (no UPDATE). Returns (DriveStats, coverage_note);
    degrades to (DriveStats(), note) on any endpoint/parse failure — never raises.
    """

    if not query_url:
        return DriveStats(), "drive co-activation unavailable: no AUTONOMY_GRAPH_QUERY_URL configured"

    import urllib.error  # lazy imports keep module import I/O-free for tests
    import urllib.parse
    import urllib.request

    sparql = build_drive_coactivation_histogram_sparql(since, graph_uri)
    body = urllib.parse.urlencode({"query": sparql}).encode("utf-8")
    req = urllib.request.Request(
        query_url,
        data=body,
        headers={
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        logger.error("drive-audit SPARQL query failed", exc_info=True)
        return DriveStats(), f"drive co-activation unavailable: SPARQL query to {query_url} failed"

    bindings = (((payload or {}).get("results") or {}).get("bindings")) or []
    hist = parse_sparql_histogram_bindings(bindings)
    stats = drive_stats_from_histogram(hist)
    if stats.record_count == 0:
        return stats, f"drive co-activation: Fuseki returned no DriveAudit rows since {since.isoformat()}"

    return stats, (
        f"drive co-activation: {stats.record_count} DriveAudit rows since "
        f"{since.isoformat()} (Fuseki histogram)"
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
    window_days: int,
    window_start: datetime,
    window_end: datetime,
    silent: SelfStateMetrics,
    busy: SelfStateMetrics,
    drive: DriveStats,
    pressure: ResourcePressureStats,
    verdict_a: str,
    verdict_b: str,
    caveats: list[str],
) -> str:
    hist = ", ".join(f"{k}:{v}" for k, v in sorted(drive.concurrent_active_hist.items())) or "(none)"
    lines = [
        "# Autonomy Origination Measurement Gate",
        "",
        "Read-only measurement. No writes, no events, no flag changes.",
        "",
        f"- Window: last {window_days} day(s) ({window_start.isoformat()} -> {window_end.isoformat()})",
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
        f"- drive audit records: {drive.record_count}",
        f"- coactivation_frac (active_count >= 2): {_fmt(drive.coactivation_frac)}",
        f"- concurrent_active_hist: {hist}",
        f"- resource_pressure rows: {pressure.row_count}",
        f"- resource_pressure median: {_fmt(pressure.median)}",
        f"- resource_pressure p90: {_fmt(pressure.p90)}",
        f"- resource_pressure frac >= {RESOURCE_PRESSURE_LEVEL}: {_fmt(pressure.frac_gt_level)}",
        "",
        "Rule (b) GO iff coactivation_frac >= "
        f"{COACTIVATION_MIN_FRAC} AND resource_pressure frac >= "
        f"{RESOURCE_PRESSURE_LEVEL} is >= {RESOURCE_PRESSURE_MIN_FRAC}.",
        "",
        "## Coverage caveats",
        "",
    ]
    if caveats:
        lines.extend(f"- {c}" for c in caveats)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


# ===========================================================================
# Orchestration (I/O layer)
# ===========================================================================


def run(window_days: int) -> int:
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=window_days)
    dsn = os.environ.get("POSTGRES_URI", DEFAULT_POSTGRES_URI)
    fuseki_url = os.environ.get("AUTONOMY_GRAPH_QUERY_URL", DEFAULT_FUSEKI_QUERY_URL)

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
    progress.emit(
        "self_state loaded", percent=40.0, processed=len(self_states), total=len(self_states), anomalies=anomalies
    )

    receipts, r_trunc = fetch_receipt_timestamps(conn, window_start)
    if r_trunc:
        caveats.append(f"reduction receipts truncated at MAX_ROWS={MAX_ROWS}")
        anomalies += 1
    progress.emit(
        "receipts loaded", percent=60.0, processed=len(receipts), total=len(receipts), anomalies=anomalies
    )

    # Turns: no cheap turn store located in the repo; do not invent one.
    turn_timestamps: list[datetime] = []
    caveats.append("turns unavailable, turn_count=0 (no turn store wired into this measurement)")

    drive_stats, coverage_note = fetch_drive_stats(fuseki_url, window_start)
    caveats.append(coverage_note)
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
        window_days=window_days,
        window_start=window_start,
        window_end=now,
        silent=silent_metrics,
        busy=busy_metrics,
        drive=drive_stats,
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
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only autonomy-origination measurement gate.")
    parser.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help=f"analysis window in days (default {DEFAULT_WINDOW_DAYS})",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_arg_parser().parse_args(argv)
    return run(args.window_days)


if __name__ == "__main__":  # pragma: no cover - guarded so import stays I/O-free
    raise SystemExit(main())
