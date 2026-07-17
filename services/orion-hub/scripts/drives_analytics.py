"""Pure helpers for Hub Drives Analytics (no I/O).

KPI / saturation math imports from ``scripts/analysis/measure_autonomy_gate.py``
via file-path load so Hub's ``scripts`` package does not shadow the repo
``scripts/`` tree. Thresholds are re-exported for unit parity checks.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from orion.spark.concept_induction.drives import DRIVE_KEYS

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
