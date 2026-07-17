from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
hub_scripts_pkg = HUB_ROOT / "scripts" / "__init__.py"
if (
    "scripts" not in sys.modules
    or not str(getattr(sys.modules.get("scripts"), "__file__", "")).startswith(str(HUB_ROOT))
):
    spec = importlib.util.spec_from_file_location(
        "scripts",
        str(hub_scripts_pkg),
        submodule_search_locations=[str(HUB_ROOT / "scripts")],
    )
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules["scripts"] = module
        spec.loader.exec_module(module)

from scripts import drives_analytics as da


def test_normalize_hours_clamps_to_allowlist() -> None:
    assert da.normalize_hours(24) == 24
    assert da.normalize_hours(99) == 24  # default
    assert da.normalize_hours(168) == 168


def test_normalize_subject_defaults() -> None:
    assert da.normalize_subject(None) == "orion"
    assert da.normalize_subject("  ") == "orion"
    assert da.normalize_subject("juniper") == "juniper"


def test_coverage_meta_reports_short_history() -> None:
    now = datetime(2026, 7, 16, 12, tzinfo=timezone.utc)
    oldest = now - timedelta(hours=3)
    meta = da.coverage_meta(
        requested_hours=24, oldest=oldest, newest=now, row_count=10
    )
    assert meta["row_count"] == 10
    assert meta["coverage_hours"] == 3.0
    assert "retention_note" in meta and meta["retention_note"]


def test_align_color_regimes() -> None:
    assert da.align_color_for_drive(
        pressure=0.8, has_matching_goal=True, saturated=False, starved=False, stale=False
    ) == "green"
    assert da.align_color_for_drive(
        pressure=0.8, has_matching_goal=False, saturated=False, starved=False, stale=False
    ) == "yellow"
    assert da.align_color_for_drive(
        pressure=0.9, has_matching_goal=True, saturated=True, starved=False, stale=False
    ) == "red"
    assert da.align_color_for_drive(
        pressure=0.0, has_matching_goal=False, saturated=False, starved=False, stale=False
    ) == "neutral"
    assert da.align_color_for_drive(
        pressure=0.5, has_matching_goal=True, saturated=False, starved=True, stale=False
    ) == "red"
    assert da.align_color_for_drive(
        pressure=0.5, has_matching_goal=True, saturated=False, starved=False, stale=True
    ) == "red"


def test_aggregate_tick_attribution_skips_nulls() -> None:
    rows = [
        {"tick_attribution": {"predictive": 0.5}, "observed_at": "2026-07-16T10:00:00+00:00"},
        {"tick_attribution": None, "observed_at": "2026-07-16T09:00:00+00:00"},
    ]
    out = da.aggregate_tick_attribution(rows)
    assert out["attributed_row_count"] == 1
    assert out["null_attribution_row_count"] == 1
    assert out["per_drive"]["predictive"] == 0.5
    assert out["first_attributed_at"] == "2026-07-16T10:00:00+00:00"


def test_saturation_thresholds_match_autonomy_gate() -> None:
    # Hub ``scripts`` shadows repo ``scripts/``; load gate the same way the helper does.
    gate = da._load_measure_autonomy_gate()
    assert da.SATURATION_DOMINANT_SHARE == gate.SATURATION_DOMINANT_SHARE
    assert da.SATURATION_ALL_ACTIVE_FRAC == gate.SATURATION_ALL_ACTIVE_FRAC
    assert da.COACTIVATION_MIN_FRAC == gate.COACTIVATION_MIN_FRAC
    assert da.SATURATION_DOMINANT_SHARE == 0.90


def test_drive_economy_verdict_never_claims_full_go() -> None:
    empty = da.drive_stats_from_histogram({})
    assert da.drive_economy_verdict_from_drive_stats(empty) == "UNMEASURABLE"

    low = da.drive_stats_from_histogram({1: 100})
    assert da.drive_economy_verdict_from_drive_stats(low) == "NO-GO"

    healthy = da.apply_dominant_counts(
        da.drive_stats_from_histogram({2: 50, 3: 50}),
        {"predictive": 40, "relational": 40, "coherence": 20},
    )
    assert da.drive_economy_verdict_from_drive_stats(healthy) == "GO_DRIVE_ONLY"

    mono = da.apply_dominant_counts(
        da.drive_stats_from_histogram({2: 100}),
        {"predictive": 96},
    )
    assert da.drive_economy_verdict_from_drive_stats(mono) == "SATURATED"


def test_downsample_and_bucket_tick_rates() -> None:
    points = [{"t": i, "v": float(i)} for i in range(10)]
    assert len(da.downsample_series(points, max_points=4)) == 4
    assert da.downsample_series(points, max_points=100) == points

    start = datetime(2026, 7, 16, 12, tzinfo=timezone.utc)
    stamps = [start + timedelta(seconds=s) for s in (0, 10, 70)]
    buckets = da.bucket_tick_rates(stamps, window_start=start, bucket_sec=60)
    assert buckets[0] == {"t": start.isoformat(), "count": 2}
    assert buckets[1]["count"] == 1


def test_gate_script_candidates_include_vendor_and_orion_repo_root(monkeypatch, tmp_path) -> None:
    """Hub Docker parents[3] is `/`; resolution must prefer ORION_REPO_ROOT / vendor."""
    import scripts.drives_analytics as da_live

    gate_dir = tmp_path / "scripts" / "analysis"
    gate_dir.mkdir(parents=True)
    gate_file = gate_dir / "measure_autonomy_gate.py"
    gate_file.write_text(
        "# stub\nSATURATION_DOMINANT_SHARE = 0.90\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ORION_REPO_ROOT", str(tmp_path))
    monkeypatch.delenv("HUB_AGENT_CLAUDE_WORKSPACE", raising=False)
    candidates = da_live._gate_script_candidates()
    assert any(p.resolve() == gate_file.resolve() for p in candidates)
    # Vendor path under Hub service root always listed first.
    assert candidates[0].name == "measure_autonomy_gate.py"
    assert "vendor" in candidates[0].parts


def test_embedded_gate_fallback_preserves_thresholds() -> None:
    import scripts.drives_analytics as da_live

    fallback = da_live._build_embedded_gate_fallback()
    assert fallback.SATURATION_DOMINANT_SHARE == 0.90
    assert fallback.COACTIVATION_MIN_FRAC == 0.10
    stats = fallback.apply_dominant_counts(
        fallback.drive_stats_from_histogram({2: 100}),
        {"predictive": 96},
    )
    assert stats.top_dominant_share == 0.96
