"""Postgres / store adapters for Hub Drives Analytics.

I/O only — degrade never raise. Pure helpers live in ``drives_analytics``.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from orion.spark.concept_induction.drives import DRIVE_KEYS

from .drives_analytics import (
    SATURATED,
    SUBJECT_ALLOWLIST,
    aggregate_tick_attribution,
    align_color_for_drive,
    build_window_kpis,
    coverage_meta,
    downsample_series,
    drive_economy_verdict_from_drive_stats,
    bucket_tick_rates,
    normalize_hours,
    normalize_subject,
    parse_dominant_rows,
    resolve_repo_root,
    _as_utc,
    _gate,
)

logger = logging.getLogger("orion.hub.drives_analytics_queries")


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
            # Full-window SQL means (not a recent-5000 sample).
            mean_row = await conn.fetchrow(
                f"""
                SELECT
                  AVG(NULLIF(drive_pressures->>'coherence', '')::double precision) AS coherence,
                  AVG(NULLIF(drive_pressures->>'continuity', '')::double precision) AS continuity,
                  AVG(NULLIF(drive_pressures->>'capability', '')::double precision) AS capability,
                  AVG(NULLIF(drive_pressures->>'relational', '')::double precision) AS relational,
                  AVG(NULLIF(drive_pressures->>'predictive', '')::double precision) AS predictive,
                  AVG(NULLIF(drive_pressures->>'autonomy', '')::double precision) AS autonomy
                FROM drive_audits
                WHERE {WINDOW_PRED}
                  AND drive_pressures IS NOT NULL
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
    mean_pressures = {k: 0.0 for k in DRIVE_KEYS}
    if mean_row is not None:
        for key in DRIVE_KEYS:
            val = mean_row[key]
            if val is not None:
                try:
                    mean_pressures[key] = float(val)
                except (TypeError, ValueError):
                    mean_pressures[key] = 0.0
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
        repo_root = resolve_repo_root()
        if repo_root is None:
            raise FileNotFoundError("Orion repo root not found for drive_state_divergence_audit")
        audit_mod_path = repo_root / "scripts" / "drive_state_divergence_audit.py"
        if not audit_mod_path.is_file():
            raise FileNotFoundError(f"missing {audit_mod_path}")
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
            notes.append(
                "funnel counts are from active goal headlines only; "
                "completed/archived are excluded by the autonomy repository read path"
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
        "funnel_scope": "active_headlines_only",
        "notes": notes,
    }


async def fetch_drive_rail_verdict(
    pool: Any,
    *,
    subject: str | None = None,
    hours: int | None = 24,
    now: datetime | None = None,
) -> str:
    """Cheap hist+dominant query for saturation coloring (no full window payload)."""
    from .drives_analytics import apply_dominant_counts, drive_stats_from_histogram

    subject_n = normalize_subject(subject)
    hours_n = normalize_hours(hours)
    if pool is None:
        return "UNMEASURABLE"
    window_start, window_end = _window_bounds(hours_n, now=now)
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
    except Exception as exc:  # noqa: BLE001
        logger.warning("drives_analytics_rail_verdict_failed error=%s", exc)
        return "UNMEASURABLE"
    hist = _gate.parse_postgres_histogram_rows(
        [(row["active_count"], row["n"]) for row in hist_rows]
    )
    dominant_counts = parse_dominant_rows(
        [(row["dominant_drive"], row["n"]) for row in dominant_rows]
    )
    stats = apply_dominant_counts(drive_stats_from_histogram(hist), dominant_counts)
    return drive_economy_verdict_from_drive_stats(stats)


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
        verdict = await fetch_drive_rail_verdict(pool, subject=subject_n, hours=24)
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
