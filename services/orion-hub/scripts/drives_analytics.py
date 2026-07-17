"""Hub Drives Analytics helpers + Postgres/store adapters.

Pure helpers (normalize, coverage, KPIs, colors, series) have no I/O.
Async adapters read ``drive_audits`` via Hub ``memory_pg_pool`` and degrade
never raise. KPI / saturation math imports from
``scripts/analysis/measure_autonomy_gate.py`` via file-path load so Hub's
``scripts`` package does not shadow the repo ``scripts/`` tree.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import logging
import os
from typing import Any, Literal

from orion.spark.concept_induction.drives import DRIVE_KEYS

logger = logging.getLogger("orion.hub.drives_analytics")

REPO_ROOT = Path(__file__).resolve().parents[3]
_GATE_MODULE_NAME = "_orion_measure_autonomy_gate_for_hub_drives"


def _load_measure_autonomy_gate():
    """Load measure_autonomy_gate without colliding with Hub ``scripts``."""
    existing = sys.modules.get(_GATE_MODULE_NAME)
    if existing is not None:
        return existing
    path = REPO_ROOT / "scripts" / "analysis" / "measure_autonomy_gate.py"
    spec = importlib.util.spec_from_file_location(_GATE_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load measure_autonomy_gate from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_GATE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


_gate = _load_measure_autonomy_gate()

# Re-export gate pure helpers + thresholds (do not fork).
drive_stats_from_histogram = _gate.drive_stats_from_histogram
parse_dominant_rows = _gate.parse_dominant_rows
apply_dominant_counts = _gate.apply_dominant_counts
retention_caveat = _gate.retention_caveat
SATURATION_DOMINANT_SHARE = _gate.SATURATION_DOMINANT_SHARE
SATURATION_MIN_ACTIVE = _gate.SATURATION_MIN_ACTIVE
SATURATION_ALL_ACTIVE_FRAC = _gate.SATURATION_ALL_ACTIVE_FRAC
COACTIVATION_MIN_FRAC = _gate.COACTIVATION_MIN_FRAC
UNMEASURABLE = _gate.UNMEASURABLE
SATURATED = _gate.SATURATED

ALLOWED_HOURS: set[int] = {1, 6, 24, 168}
DEFAULT_HOURS: int = 24
SUBJECT_ALLOWLIST: tuple[str, ...] = ("orion", "relationship", "juniper")
DEFAULT_SUBJECT: str = "orion"
GO_DRIVE_ONLY = "GO_DRIVE_ONLY"
AlignColor = Literal["green", "yellow", "red", "neutral"]


def normalize_hours(hours: int | None) -> int:
    if hours is None:
        return DEFAULT_HOURS
    try:
        value = int(hours)
    except (TypeError, ValueError):
        return DEFAULT_HOURS
    if value in ALLOWED_HOURS:
        return value
    return DEFAULT_HOURS


def normalize_subject(subject: str | None) -> str:
    if subject is None:
        return DEFAULT_SUBJECT
    cleaned = str(subject).strip()
    return cleaned or DEFAULT_SUBJECT


def _as_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def coverage_meta(
    *,
    requested_hours: int,
    oldest: datetime | None,
    newest: datetime | None,
    row_count: int,
) -> dict[str, Any]:
    coverage_hours: float | None = None
    retention_note: str | None = None
    if oldest is not None and newest is not None:
        oldest_u = _as_utc(oldest)
        newest_u = _as_utc(newest)
        coverage_hours = max(0.0, (newest_u - oldest_u).total_seconds() / 3600.0)
        window_start = newest_u - timedelta(hours=int(requested_hours))
        retention_note = retention_caveat("drive_audits", oldest_u, window_start)
    return {
        "requested_hours": int(requested_hours),
        "row_count": int(row_count),
        "oldest": _as_utc(oldest).isoformat() if oldest is not None else None,
        "newest": _as_utc(newest).isoformat() if newest is not None else None,
        "coverage_hours": coverage_hours,
        "retention_note": retention_note,
    }


def drive_economy_verdict_from_drive_stats(drive_stats: Any) -> str:
    """Drive-rail verdict for the Hub strip.

    Uses measure_autonomy_gate saturation/coactivation thresholds.
    Returns UNMEASURABLE | SATURATED | NO-GO | GO_DRIVE_ONLY.
    GO_DRIVE_ONLY means coactivation bar met and not saturated; full GO
    still requires resource_pressure (offline gate only). Never claim full GO.
    """
    record_count = int(getattr(drive_stats, "record_count", 0) or 0)
    if record_count == 0:
        return UNMEASURABLE
    coactivation = float(getattr(drive_stats, "coactivation_frac", 0.0) or 0.0)
    if coactivation < COACTIVATION_MIN_FRAC:
        return "NO-GO"
    top_share = float(getattr(drive_stats, "top_dominant_share", 0.0) or 0.0)
    all_active = float(getattr(drive_stats, "all_active_frac", 0.0) or 0.0)
    if top_share >= SATURATION_DOMINANT_SHARE or all_active >= SATURATION_ALL_ACTIVE_FRAC:
        return SATURATED
    return GO_DRIVE_ONLY


def build_window_kpis(
    *,
    active_count_hist: dict[int, int],
    dominant_counts: dict[str, int],
    mean_pressures: dict[str, float],
    coverage: dict[str, Any],
) -> dict[str, Any]:
    stats = apply_dominant_counts(
        drive_stats_from_histogram(dict(active_count_hist or {})),
        dict(dominant_counts or {}),
    )
    return {
        "record_count": stats.record_count,
        "coactivation_frac": stats.coactivation_frac,
        "all_active_frac": stats.all_active_frac,
        "top_dominant_share": stats.top_dominant_share,
        "dominant_counts": dict(stats.dominant_counts),
        "concurrent_active_hist": dict(stats.concurrent_active_hist),
        "mean_pressures": {k: float(mean_pressures.get(k, 0.0)) for k in DRIVE_KEYS},
        "gate_verdict_drive_only": drive_economy_verdict_from_drive_stats(stats),
        "coverage": dict(coverage or {}),
        # Explicit: Hub strip never claims full offline-gate GO.
        "gate_note": (
            "GO_DRIVE_ONLY is drive-rail only; full GO still needs resource_pressure"
            if drive_economy_verdict_from_drive_stats(stats) == GO_DRIVE_ONLY
            else None
        ),
    }


def aggregate_tick_attribution(rows: list[dict]) -> dict[str, Any]:
    per_drive = {k: 0.0 for k in DRIVE_KEYS}
    attributed = 0
    nulls = 0
    first_attributed_at: str | None = None
    for row in rows or []:
        attr = row.get("tick_attribution") if isinstance(row, dict) else None
        if attr is None:
            nulls += 1
            continue
        if not isinstance(attr, dict):
            nulls += 1
            continue
        attributed += 1
        for key in DRIVE_KEYS:
            try:
                per_drive[key] += float(attr.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
        observed = row.get("observed_at")
        if observed is not None:
            text = observed.isoformat() if isinstance(observed, datetime) else str(observed)
            if first_attributed_at is None or text < first_attributed_at:
                first_attributed_at = text
    return {
        "per_drive": per_drive,
        "attributed_row_count": attributed,
        "null_attribution_row_count": nulls,
        "first_attributed_at": first_attributed_at,
    }


def align_color_for_drive(
    *,
    pressure: float,
    has_matching_goal: bool,
    saturated: bool,
    starved: bool,
    stale: bool,
) -> AlignColor:
    """Align-mode color for one drive (regime overrides magnitude)."""
    if saturated or starved or stale:
        return "red"
    try:
        p = float(pressure)
    except (TypeError, ValueError):
        p = 0.0
    if p <= 0.0:
        return "neutral"
    if has_matching_goal:
        return "green"
    return "yellow"


def downsample_series(points: list[dict], max_points: int = 240) -> list[dict]:
    if max_points <= 0:
        return []
    if not points or len(points) <= max_points:
        return list(points)
    if max_points == 1:
        return [points[-1]]
    last_idx = len(points) - 1
    out: list[dict] = []
    for i in range(max_points):
        idx = round(i * last_idx / (max_points - 1))
        out.append(points[idx])
    return out


def bucket_tick_rates(
    timestamps: list[datetime],
    *,
    window_start: datetime,
    bucket_sec: int,
) -> list[dict[str, Any]]:
    if bucket_sec <= 0:
        return []
    start = _as_utc(window_start)
    counts: dict[int, int] = {}
    max_idx = -1
    for ts in timestamps or []:
        if not isinstance(ts, datetime):
            continue
        delta = (_as_utc(ts) - start).total_seconds()
        if delta < 0:
            continue
        idx = int(delta // bucket_sec)
        counts[idx] = counts.get(idx, 0) + 1
        if idx > max_idx:
            max_idx = idx
    if max_idx < 0:
        return []
    out: list[dict[str, Any]] = []
    for idx in range(max_idx + 1):
        bucket_t = start + timedelta(seconds=idx * bucket_sec)
        out.append({"t": bucket_t.isoformat(), "count": int(counts.get(idx, 0))})
    return out


# ---------------------------------------------------------------------------
# Async Postgres / store adapters (Task 3–4). Degrade never raise.
# ---------------------------------------------------------------------------

STALE_AFTER_SEC = 300
SERIES_ROW_CAP = 5000
SNAPSHOT_COLUMNS = (
    "artifact_id, subject, active_count, active_drives, dominant_drive, summary, "
    "drive_pressures, tick_attribution, tension_kinds, correlation_id, "
    "observed_at, created_at"
)
WINDOW_PRED = (
    "subject = $1 AND COALESCE(observed_at, created_at) >= $2 "
    "AND COALESCE(observed_at, created_at) < $3"
)

try:
    from asyncpg.exceptions import UndefinedTableError as _UndefinedTableError
except ImportError:  # pragma: no cover
    _UndefinedTableError = None  # type: ignore[misc, assignment]


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value
    return value


def _pressures_dict(raw: Any) -> dict[str, float]:
    parsed = _parse_jsonish(raw)
    if not isinstance(parsed, dict):
        return {k: 0.0 for k in DRIVE_KEYS}
    out: dict[str, float] = {}
    for key in DRIVE_KEYS:
        try:
            out[key] = float(parsed.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            out[key] = 0.0
    return out


def _is_undefined_table(exc: BaseException) -> bool:
    if _UndefinedTableError is not None and isinstance(exc, _UndefinedTableError):
        return True
    return bool(_gate.is_undefined_table_error(exc))


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _window_bounds(hours: int, *, now: datetime | None = None) -> tuple[datetime, datetime]:
    end = _as_utc(now or _now_utc())
    start = end - timedelta(hours=int(hours))
    return start, end


def _pool_missing_payload(**extra: Any) -> dict[str, Any]:
    return {
        "degraded": True,
        "error": "memory_pg_pool_unavailable",
        "source": {
            "table": "drive_audits",
            "degraded": True,
            "error": "memory_pg_pool_unavailable",
            **extra,
        },
    }


def _query_failed_payload(exc: BaseException, **extra: Any) -> dict[str, Any]:
    if _is_undefined_table(exc):
        error = "drive_audits table does not exist yet"
    else:
        error = str(exc) or exc.__class__.__name__
    return {
        "degraded": True,
        "error": error,
        "source": {"table": "drive_audits", "degraded": True, "error": error, **extra},
    }


async def fetch_subjects(pool: Any) -> dict[str, Any]:
    """Allowlist ∪ distinct DB subjects with coverage badges."""
    if pool is None:
        return {
            **_pool_missing_payload(),
            "subjects": [
                {
                    "subject": s,
                    "row_count": 0,
                    "oldest_ts": None,
                    "newest_ts": None,
                    "allowlisted": True,
                    "discovered": False,
                }
                for s in SUBJECT_ALLOWLIST
            ],
        }
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT subject,
                       COUNT(*)::bigint AS row_count,
                       MIN(COALESCE(observed_at, created_at)) AS oldest_ts,
                       MAX(COALESCE(observed_at, created_at)) AS newest_ts
                FROM drive_audits
                GROUP BY subject
                ORDER BY subject
                """
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("drives_analytics_subjects_failed error=%s", exc)
        payload = _query_failed_payload(exc)
        payload["subjects"] = [
            {
                "subject": s,
                "row_count": 0,
                "oldest_ts": None,
                "newest_ts": None,
                "allowlisted": True,
                "discovered": False,
            }
            for s in SUBJECT_ALLOWLIST
        ]
        return payload

    by_subject: dict[str, dict[str, Any]] = {}
    for row in rows:
        subject = str(row["subject"] or "").strip()
        if not subject:
            continue
        oldest = row["oldest_ts"]
        newest = row["newest_ts"]
        by_subject[subject] = {
            "subject": subject,
            "row_count": int(row["row_count"] or 0),
            "oldest_ts": _as_utc(oldest).isoformat() if isinstance(oldest, datetime) else None,
            "newest_ts": _as_utc(newest).isoformat() if isinstance(newest, datetime) else None,
            "allowlisted": subject in SUBJECT_ALLOWLIST,
            "discovered": True,
        }
    subjects: list[dict[str, Any]] = []
    for subject in SUBJECT_ALLOWLIST:
        if subject in by_subject:
            entry = dict(by_subject[subject])
            entry["allowlisted"] = True
            subjects.append(entry)
        else:
            subjects.append(
                {
                    "subject": subject,
                    "row_count": 0,
                    "oldest_ts": None,
                    "newest_ts": None,
                    "allowlisted": True,
                    "discovered": False,
                }
            )
    for subject, entry in sorted(by_subject.items()):
        if subject not in SUBJECT_ALLOWLIST:
            subjects.append(entry)
    return {
        "degraded": False,
        "source": {"table": "drive_audits", "degraded": False},
        "subjects": subjects,
    }


async def fetch_snapshot(pool: Any, *, subject: str | None = None) -> dict[str, Any]:
    subject_n = normalize_subject(subject)
    if pool is None:
        return {**_pool_missing_payload(subject=subject_n), "subject": subject_n}
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT {SNAPSHOT_COLUMNS}
                FROM drive_audits
                WHERE subject = $1
                ORDER BY COALESCE(observed_at, created_at) DESC
                LIMIT 1
                """,
                subject_n,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("drives_analytics_snapshot_failed error=%s", exc)
        return {**_query_failed_payload(exc, subject=subject_n), "subject": subject_n}

    if row is None:
        return {
            "degraded": True,
            "error": "no drive_audits rows for subject",
            "source": {
                "table": "drive_audits",
                "subject": subject_n,
                "degraded": True,
                "error": "no drive_audits rows for subject",
            },
            "subject": subject_n,
        }

    observed = row["observed_at"] or row["created_at"]
    observed_dt = _as_utc(observed) if isinstance(observed, datetime) else None
    stale = True
    if observed_dt is not None:
        stale = (_now_utc() - observed_dt).total_seconds() > STALE_AFTER_SEC
    active_drives = _parse_jsonish(row["active_drives"]) or []
    if not isinstance(active_drives, list):
        active_drives = []
    tension_kinds = _parse_jsonish(row["tension_kinds"]) or []
    if not isinstance(tension_kinds, list):
        tension_kinds = []
    tick_attribution = _parse_jsonish(row["tick_attribution"])
    if not isinstance(tick_attribution, dict):
        tick_attribution = None
    return {
        "degraded": False,
        "source": {"table": "drive_audits", "subject": subject_n, "degraded": False},
        "subject": subject_n,
        "artifact_id": row["artifact_id"],
        "observed_at": observed_dt.isoformat() if observed_dt else None,
        "drive_pressures": _pressures_dict(row["drive_pressures"]),
        "active_drives": [str(x) for x in active_drives],
        "active_count": int(row["active_count"] or 0),
        "dominant_drive": row["dominant_drive"],
        "summary": row["summary"],
        "tick_attribution": tick_attribution,
        "tension_kinds": [str(x) for x in tension_kinds],
        "stale": stale,
    }


async def fetch_window(
    pool: Any,
    *,
    subject: str | None = None,
    hours: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    subject_n = normalize_subject(subject)
    hours_n = normalize_hours(hours)
    window_start, window_end = _window_bounds(hours_n, now=now)
    base_source = {"table": "drive_audits", "subject": subject_n, "hours": hours_n}
    if pool is None:
        return {
            **_pool_missing_payload(**base_source),
            "subject": subject_n,
            "hours": hours_n,
        }
    try:
        async with pool.acquire() as conn:
            hist_rows = await conn.fetch(
                f"""
                SELECT active_count, COUNT(*)::bigint AS n
                FROM drive_audits
                WHERE {WINDOW_PRED}
                GROUP BY active_count
                """,
                subject_n,
                window_start,
                window_end,
            )
            dominant_rows = await conn.fetch(
                f"""
                SELECT dominant_drive, COUNT(*)::bigint AS n
                FROM drive_audits
                WHERE {WINDOW_PRED}
                GROUP BY dominant_drive
                """,
                subject_n,
                window_start,
                window_end,
            )
            bounds = await conn.fetchrow(
                f"""
                SELECT COUNT(*)::bigint AS row_count,
                       MIN(COALESCE(observed_at, created_at)) AS oldest_ts,
                       MAX(COALESCE(observed_at, created_at)) AS newest_ts
                FROM drive_audits
                WHERE {WINDOW_PRED}
                """,
                subject_n,
                window_start,
                window_end,
            )
            pressure_rows = await conn.fetch(
                f"""
                SELECT drive_pressures
                FROM drive_audits
                WHERE {WINDOW_PRED}
                ORDER BY COALESCE(observed_at, created_at) DESC
                LIMIT 5000
                """,
                subject_n,
                window_start,
                window_end,
            )
            attr_rows = await conn.fetch(
                f"""
                SELECT tick_attribution, COALESCE(observed_at, created_at) AS observed_at
                FROM drive_audits
                WHERE {WINDOW_PRED}
                ORDER BY COALESCE(observed_at, created_at) DESC
                LIMIT 5000
                """,
                subject_n,
                window_start,
                window_end,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("drives_analytics_window_failed error=%s", exc)
        return {
            **_query_failed_payload(exc, **base_source),
            "subject": subject_n,
            "hours": hours_n,
        }

    active_count_hist = _gate.parse_postgres_histogram_rows(
        [(row["active_count"], row["n"]) for row in hist_rows]
    )
    dominant_counts = parse_dominant_rows(
        [(row["dominant_drive"], row["n"]) for row in dominant_rows]
    )
    sums = {k: 0.0 for k in DRIVE_KEYS}
    n_pressure = 0
    for row in pressure_rows:
        pressures = _pressures_dict(row["drive_pressures"])
        n_pressure += 1
        for key in DRIVE_KEYS:
            sums[key] += pressures[key]
    mean_pressures = {
        k: (sums[k] / n_pressure if n_pressure else 0.0) for k in DRIVE_KEYS
    }
    oldest = bounds["oldest_ts"] if bounds else None
    newest = bounds["newest_ts"] if bounds else None
    row_count = int(bounds["row_count"] or 0) if bounds else 0
    coverage = coverage_meta(
        requested_hours=hours_n,
        oldest=oldest if isinstance(oldest, datetime) else None,
        newest=newest if isinstance(newest, datetime) else None,
        row_count=row_count,
    )
    attribution = aggregate_tick_attribution(
        [
            {
                "tick_attribution": _parse_jsonish(row["tick_attribution"]),
                "observed_at": row["observed_at"],
            }
            for row in attr_rows
        ]
    )
    kpis = build_window_kpis(
        active_count_hist=active_count_hist,
        dominant_counts=dominant_counts,
        mean_pressures=mean_pressures,
        coverage=coverage,
    )
    return {
        "degraded": False,
        "source": {**base_source, "degraded": False},
        "subject": subject_n,
        "hours": hours_n,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "kpis": kpis,
        "dominant_counts": dominant_counts,
        "mean_pressures": mean_pressures,
        "attribution": attribution,
        "coverage": coverage,
    }


async def fetch_series(
    pool: Any,
    *,
    subject: str | None = None,
    hours: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    subject_n = normalize_subject(subject)
    hours_n = normalize_hours(hours)
    window_start, window_end = _window_bounds(hours_n, now=now)
    base_source = {"table": "drive_audits", "subject": subject_n, "hours": hours_n}
    if pool is None:
        return {
            **_pool_missing_payload(**base_source),
            "subject": subject_n,
            "hours": hours_n,
            "tick_rate": [],
            "pressures": {k: [] for k in DRIVE_KEYS},
            "coverage": coverage_meta(
                requested_hours=hours_n, oldest=None, newest=None, row_count=0
            ),
        }
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT COALESCE(observed_at, created_at) AS ts, drive_pressures
                FROM drive_audits
                WHERE {WINDOW_PRED}
                ORDER BY COALESCE(observed_at, created_at) ASC
                LIMIT {SERIES_ROW_CAP}
                """,
                subject_n,
                window_start,
                window_end,
            )
            bounds = await conn.fetchrow(
                f"""
                SELECT COUNT(*)::bigint AS row_count,
                       MIN(COALESCE(observed_at, created_at)) AS oldest_ts,
                       MAX(COALESCE(observed_at, created_at)) AS newest_ts
                FROM drive_audits
                WHERE {WINDOW_PRED}
                """,
                subject_n,
                window_start,
                window_end,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("drives_analytics_series_failed error=%s", exc)
        return {
            **_query_failed_payload(exc, **base_source),
            "subject": subject_n,
            "hours": hours_n,
            "tick_rate": [],
            "pressures": {k: [] for k in DRIVE_KEYS},
            "coverage": coverage_meta(
                requested_hours=hours_n, oldest=None, newest=None, row_count=0
            ),
        }

    timestamps: list[datetime] = []
    pressure_series: dict[str, list[dict[str, Any]]] = {k: [] for k in DRIVE_KEYS}
    for row in rows:
        ts = row["ts"]
        if not isinstance(ts, datetime):
            continue
        ts_u = _as_utc(ts)
        timestamps.append(ts_u)
        pressures = _pressures_dict(row["drive_pressures"])
        iso = ts_u.isoformat()
        for key in DRIVE_KEYS:
            pressure_series[key].append({"t": iso, "v": pressures[key]})

    # Bucket width scales with window; keep series readable under 240 points.
    bucket_sec = max(60, int((hours_n * 3600) / 120))
    tick_rate = downsample_series(
        bucket_tick_rates(timestamps, window_start=window_start, bucket_sec=bucket_sec),
        max_points=240,
    )
    pressures_out = {
        k: downsample_series(points, max_points=240) for k, points in pressure_series.items()
    }
    oldest = bounds["oldest_ts"] if bounds else None
    newest = bounds["newest_ts"] if bounds else None
    row_count = int(bounds["row_count"] or 0) if bounds else 0
    coverage = coverage_meta(
        requested_hours=hours_n,
        oldest=oldest if isinstance(oldest, datetime) else None,
        newest=newest if isinstance(newest, datetime) else None,
        row_count=row_count,
    )
    return {
        "degraded": False,
        "source": {**base_source, "degraded": False},
        "subject": subject_n,
        "hours": hours_n,
        "tick_rate": tick_rate,
        "pressures": pressures_out,
        "coverage": coverage,
    }


def _resolve_concept_store_path() -> tuple[str, bool]:
    """Return (path, store_path_is_fallback_default)."""
    from orion.spark.concept_induction.settings import DEFAULT_CONCEPT_STORE_PATH

    env_path = os.getenv("CONCEPT_STORE_PATH", "").strip()
    if env_path:
        return env_path, False
    return str(DEFAULT_CONCEPT_STORE_PATH), True


def fetch_divergence_sync(*, subject: str | None = None, audit_pressures: dict[str, float] | None = None) -> dict[str, Any]:
    """Compare drive_state.v1 concept store vs audit pressures (fail-open)."""
    subject_n = normalize_subject(subject)
    store_path, is_fallback = _resolve_concept_store_path()
    notes = [
        "autonomy_state_v2_note: frozen/historical — not a live second signal",
    ]
    drive_state_pressures: dict[str, float | None] = {k: None for k in DRIVE_KEYS}
    audit = {k: float((audit_pressures or {}).get(k, 0.0) or 0.0) for k in DRIVE_KEYS}
    degraded = False
    error: str | None = None
    try:
        # Load via same helper as scripts/drive_state_divergence_audit.py
        audit_mod_path = REPO_ROOT / "scripts" / "drive_state_divergence_audit.py"
        mod_name = "_orion_drive_state_divergence_audit_for_hub"
        existing = sys.modules.get(mod_name)
        if existing is None:
            spec = importlib.util.spec_from_file_location(mod_name, audit_mod_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load {audit_mod_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
            existing = module
        drive_raw, drive_error = existing.load_drive_state_v1(store_path, subject_n)
        if drive_error:
            degraded = True
            error = drive_error
            notes.append(drive_error)
        elif drive_raw is None:
            degraded = True
            error = "drive_state.v1 unavailable"
            notes.append(error)
        else:
            raw_pressures = drive_raw.get("pressures") or {}
            for key in DRIVE_KEYS:
                try:
                    drive_state_pressures[key] = float(raw_pressures.get(key))
                except (TypeError, ValueError):
                    drive_state_pressures[key] = None
    except Exception as exc:  # noqa: BLE001
        degraded = True
        error = str(exc) or exc.__class__.__name__
        notes.append(error)
        logger.warning("drives_analytics_divergence_failed error=%s", exc)

    if is_fallback:
        notes.append(
            f"CONCEPT_STORE_PATH unset; using DEFAULT_CONCEPT_STORE_PATH={store_path!r}"
        )
        degraded = True

    deltas: dict[str, float | None] = {}
    abs_vals: list[float] = []
    for key in DRIVE_KEYS:
        a = drive_state_pressures.get(key)
        b = audit.get(key)
        if a is None or b is None:
            deltas[key] = None
            continue
        delta = float(a) - float(b)
        deltas[key] = delta
        abs_vals.append(abs(delta))
    return {
        "degraded": degraded,
        "error": error,
        "source": {
            "store_path": store_path,
            "store_path_is_fallback_default": is_fallback,
            "table": "drive_audits",
            "subject": subject_n,
            "degraded": degraded,
            "error": error,
        },
        "subject": subject_n,
        "store_path": store_path,
        "store_path_is_fallback_default": is_fallback,
        "drive_state_pressures": drive_state_pressures,
        "audit_pressures": audit,
        "deltas": deltas,
        "max_abs_delta": max(abs_vals) if abs_vals else 0.0,
        "autonomy_state_v2_note": "frozen/historical — not a live second signal",
        "notes": notes,
    }


def fetch_goal_alignment_sync(
    *,
    subject: str | None = None,
    pressures: dict[str, float] | None = None,
    saturated: bool = False,
    stale: bool = False,
) -> dict[str, Any]:
    """Per-drive goal match + funnel posture for coloring (fail-open)."""
    subject_n = normalize_subject(subject)
    pressure_map = {k: float((pressures or {}).get(k, 0.0) or 0.0) for k in DRIVE_KEYS}
    funnel = {
        "proposed": 0,
        "active": 0,
        "planned": 0,
        "executing": 0,
        "completed": 0,
        "archived": 0,
    }
    notes: list[str] = []
    active_goals: list[dict[str, Any]] = []
    goals_available = False
    degraded = False
    try:
        from orion.autonomy.repository import build_autonomy_repository

        repo = build_autonomy_repository(backend="graph", timeout_sec=3.0, goals_limit=12)
        lookup = repo.get_latest(subject_n)
        state = getattr(lookup, "state", None)
        availability = str(getattr(lookup, "availability", "") or "")
        if state is None or availability in {"unavailable", "empty"}:
            degraded = True
            notes.append("goals unavailable")
        else:
            goals_available = True
            headlines = list(getattr(state, "goal_headlines", None) or [])
            for goal in headlines:
                status = str(getattr(goal, "proposal_status", None) or "proposed").lower()
                if status in funnel:
                    funnel[status] += 1
                elif status in {"active", "promoted"}:
                    funnel["active"] += 1
                else:
                    funnel["proposed"] += 1
                active_goals.append(
                    {
                        "artifact_id": getattr(goal, "artifact_id", None),
                        "drive_origin": getattr(goal, "drive_origin", None),
                        "proposal_status": getattr(goal, "proposal_status", None) or "proposed",
                        "goal_statement": getattr(goal, "goal_statement", None),
                    }
                )
    except Exception as exc:  # noqa: BLE001
        degraded = True
        notes.append("goals unavailable")
        notes.append(str(exc) or exc.__class__.__name__)
        logger.warning("drives_analytics_goal_alignment_failed error=%s", exc)

    matching_origins = {
        str(g.get("drive_origin") or "").strip().lower()
        for g in active_goals
        if g.get("drive_origin")
    }
    per_drive: dict[str, Any] = {}
    for key in DRIVE_KEYS:
        has_match = key in matching_origins
        # Without goals, elevated pressure falls to yellow/neutral — never invent green.
        color = align_color_for_drive(
            pressure=pressure_map[key],
            has_matching_goal=has_match if goals_available else False,
            saturated=saturated,
            starved=False,
            stale=stale,
        )
        if not goals_available and pressure_map[key] > 0 and color == "yellow":
            # Spec: degrade coloring to yellow/neutral with goals-unavailable note.
            pass
        per_drive[key] = {
            "pressure": pressure_map[key],
            "has_matching_goal": bool(has_match and goals_available),
            "color_align": color,
        }

    return {
        "degraded": degraded or (not goals_available),
        "goals_available": goals_available,
        "source": {
            "subject": subject_n,
            "degraded": degraded or (not goals_available),
            "error": None if goals_available else "goals unavailable",
        },
        "subject": subject_n,
        "active_goals": active_goals,
        "per_drive": per_drive,
        "funnel": funnel,
        "notes": notes,
    }


async def fetch_goal_alignment(
    pool: Any,
    *,
    subject: str | None = None,
) -> dict[str, Any]:
    """Goal alignment using latest audit pressures when pool available."""
    subject_n = normalize_subject(subject)
    pressures = {k: 0.0 for k in DRIVE_KEYS}
    saturated = False
    stale = False
    if pool is not None:
        snap = await fetch_snapshot(pool, subject=subject_n)
        if not snap.get("degraded"):
            pressures = dict(snap.get("drive_pressures") or pressures)
            stale = bool(snap.get("stale"))
        window = await fetch_window(pool, subject=subject_n, hours=24)
        if not window.get("degraded"):
            verdict = ((window.get("kpis") or {}).get("gate_verdict_drive_only"))
            saturated = verdict == SATURATED
    return fetch_goal_alignment_sync(
        subject=subject_n,
        pressures=pressures,
        saturated=saturated,
        stale=stale,
    )


async def fetch_divergence(
    pool: Any,
    *,
    subject: str | None = None,
) -> dict[str, Any]:
    subject_n = normalize_subject(subject)
    audit_pressures = {k: 0.0 for k in DRIVE_KEYS}
    if pool is not None:
        snap = await fetch_snapshot(pool, subject=subject_n)
        if not snap.get("degraded"):
            audit_pressures = dict(snap.get("drive_pressures") or audit_pressures)
    return fetch_divergence_sync(subject=subject_n, audit_pressures=audit_pressures)
