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

_GATE_MODULE_NAME = "_orion_measure_autonomy_gate_for_hub_drives"


def _repo_root_candidates() -> list[Path]:
    """Resolve repo roots usable from a monorepo checkout *or* Hub Docker.

    Hub image layout is ``/app/scripts/drives_analytics.py`` (service root = ``/app``),
    so ``Path(__file__).parents[3]`` is ``/`` and is *not* the Orion repo.
    Compose mounts the repo at ``ORION_REPO_ROOT`` (default ``/repo``) and
    ``/mnt/scripts/Orion-Sapienform``.
    """
    seen: set[str] = set()
    out: list[Path] = []

    def _add(path: Path | None) -> None:
        if path is None:
            return
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            resolved = path
        key = str(resolved)
        if key in seen:
            return
        seen.add(key)
        out.append(resolved)

    for key in ("ORION_REPO_ROOT", "HUB_AGENT_CLAUDE_WORKSPACE"):
        raw = os.getenv(key, "").strip()
        if raw:
            _add(Path(raw))
    here = Path(__file__).resolve()
    # Monorepo: services/orion-hub/scripts/<file> -> parents[3] = repo root
    if len(here.parents) >= 4:
        _add(here.parents[3])
    _add(Path("/repo"))
    _add(Path("/mnt/scripts/Orion-Sapienform"))
    return out


def _gate_script_candidates() -> list[Path]:
    paths: list[Path] = []
    # Image-vendored copy (Dockerfile); does not require the repo volume mount.
    paths.append(Path(__file__).resolve().parents[1] / "vendor" / "measure_autonomy_gate.py")
    for root in _repo_root_candidates():
        paths.append(root / "scripts" / "analysis" / "measure_autonomy_gate.py")
    return paths


def _build_embedded_gate_fallback():
    """Minimal copy of measure_autonomy_gate pure surface when the file is absent.

    Kept in sync with scripts/analysis/measure_autonomy_gate.py thresholds + helpers
    used by this module. Prefer the real file; this exists so Hub never 500s on
    import when the repo volume / vendor copy is missing.
    """
    from dataclasses import dataclass, field, replace
    from types import SimpleNamespace
    from typing import Iterable, Optional

    SATURATION_DOMINANT_SHARE = 0.90
    SATURATION_MIN_ACTIVE = 5
    SATURATION_ALL_ACTIVE_FRAC = 0.75
    COACTIVATION_MIN_FRAC = 0.10
    UNMEASURABLE = "UNMEASURABLE"
    SATURATED = "SATURATED"

    @dataclass
    class DriveStats:
        record_count: int = 0
        coactivation_frac: float = 0.0
        concurrent_active_hist: dict[int, int] = field(default_factory=dict)
        all_active_frac: float = 0.0
        dominant_counts: dict[str, int] = field(default_factory=dict)
        top_dominant_share: float = 0.0

    def drive_stats_from_histogram(hist: dict[int, int]) -> DriveStats:
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
        out: dict[str, int] = {}
        for row in rows:
            try:
                key = row[0]
                audits = int(row[1])
            except (TypeError, ValueError, IndexError, OverflowError):
                continue
            if audits < 0 or key is None:
                continue
            out[str(key)] = audits
        return out

    def apply_dominant_counts(stats: DriveStats, dominant_counts: dict[str, int]) -> DriveStats:
        share = 0.0
        if dominant_counts and stats.record_count > 0:
            share = max(dominant_counts.values()) / stats.record_count
        return replace(
            stats,
            dominant_counts=dict(dominant_counts),
            top_dominant_share=share,
        )

    def parse_postgres_histogram_rows(rows: Iterable[Any]) -> dict[int, int]:
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

    def _as_utc_local(ts: datetime) -> datetime:
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    def retention_caveat(
        source_name: str, oldest_available: Optional[datetime], window_start: datetime
    ) -> Optional[str]:
        if oldest_available is None:
            return None
        oldest = _as_utc_local(oldest_available)
        start = _as_utc_local(window_start)
        if oldest <= start:
            return None
        gap_hours = (oldest - start).total_seconds() / 3600.0
        return (
            f"{source_name} retention only covers back to {oldest.isoformat()}, "
            f"{gap_hours:.1f}h short of the requested window start ({start.isoformat()}) "
            "-- classification/metrics for the uncovered period reflect zero rows, "
            "not measured absence"
        )

    def is_undefined_table_error(exc: BaseException) -> bool:
        if getattr(exc, "pgcode", None) == "42P01":
            return True
        if type(exc).__name__ == "UndefinedTable":
            return True
        msg = str(exc).lower()
        return "relation" in msg and "does not exist" in msg

    return SimpleNamespace(
        DriveStats=DriveStats,
        drive_stats_from_histogram=drive_stats_from_histogram,
        parse_dominant_rows=parse_dominant_rows,
        apply_dominant_counts=apply_dominant_counts,
        parse_postgres_histogram_rows=parse_postgres_histogram_rows,
        retention_caveat=retention_caveat,
        is_undefined_table_error=is_undefined_table_error,
        SATURATION_DOMINANT_SHARE=SATURATION_DOMINANT_SHARE,
        SATURATION_MIN_ACTIVE=SATURATION_MIN_ACTIVE,
        SATURATION_ALL_ACTIVE_FRAC=SATURATION_ALL_ACTIVE_FRAC,
        COACTIVATION_MIN_FRAC=COACTIVATION_MIN_FRAC,
        UNMEASURABLE=UNMEASURABLE,
        SATURATED=SATURATED,
        _embedded_fallback=True,
    )


def _load_measure_autonomy_gate():
    """Load measure_autonomy_gate without colliding with Hub ``scripts``."""
    existing = sys.modules.get(_GATE_MODULE_NAME)
    if existing is not None:
        return existing
    for path in _gate_script_candidates():
        if not path.is_file():
            continue
        try:
            spec = importlib.util.spec_from_file_location(_GATE_MODULE_NAME, path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[_GATE_MODULE_NAME] = module
            spec.loader.exec_module(module)
            module._embedded_fallback = False  # type: ignore[attr-defined]
            module._loaded_from = str(path)  # type: ignore[attr-defined]
            return module
        except Exception as exc:  # noqa: BLE001
            sys.modules.pop(_GATE_MODULE_NAME, None)
            logger.warning("measure_autonomy_gate_load_failed path=%s error=%s", path, exc)
    logger.warning(
        "measure_autonomy_gate_missing; using embedded fallback thresholds "
        "(set ORION_REPO_ROOT or vendor the script)"
    )
    fallback = _build_embedded_gate_fallback()
    sys.modules[_GATE_MODULE_NAME] = fallback  # type: ignore[assignment]
    return fallback


_gate = _load_measure_autonomy_gate()

# Re-export gate pure helpers + thresholds (do not fork when real module loads).
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


def resolve_repo_root() -> Path | None:
    """Best-effort Orion repo root for loading sibling scripts (divergence audit)."""
    for root in _repo_root_candidates():
        if (root / "scripts" / "analysis" / "measure_autonomy_gate.py").is_file():
            return root
        if (root / "scripts" / "drive_state_divergence_audit.py").is_file():
            return root
    return None


# Back-compat name used by older helpers; prefer resolve_repo_root().
REPO_ROOT = resolve_repo_root() or (
    Path(__file__).resolve().parents[3]
    if len(Path(__file__).resolve().parents) >= 4
    else Path(__file__).resolve().parents[1]
)

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
    verdict = drive_economy_verdict_from_drive_stats(stats)
    return {
        "record_count": stats.record_count,
        "coactivation_frac": stats.coactivation_frac,
        "all_active_frac": stats.all_active_frac,
        "top_dominant_share": stats.top_dominant_share,
        "dominant_counts": dict(stats.dominant_counts),
        "concurrent_active_hist": dict(stats.concurrent_active_hist),
        "mean_pressures": {k: float(mean_pressures.get(k, 0.0)) for k in DRIVE_KEYS},
        "gate_verdict_drive_only": verdict,
        "coverage": dict(coverage or {}),
        # Explicit: Hub strip never claims full offline-gate GO.
        "gate_note": (
            "GO_DRIVE_ONLY is drive-rail only; full GO still needs resource_pressure"
            if verdict == GO_DRIVE_ONLY
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


# Re-export I/O adapters so callers can keep `from scripts import drives_analytics`.
def __getattr__(name: str):  # pragma: no cover - thin re-export
    if name.startswith("fetch_") or name in {
        "STALE_AFTER_SEC",
        "SERIES_ROW_CAP",
        "fetch_goal_alignment_sync",
        "fetch_divergence_sync",
        "fetch_drive_rail_verdict",
    }:
        from . import drives_analytics_queries as _queries

        return getattr(_queries, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
