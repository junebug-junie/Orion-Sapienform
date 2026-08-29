"""Unit tests for scripts/analyze_cabinet_ambient_correlation.py (v2 analysis)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "analyze_cabinet_ambient_correlation.py"


def _load():
    spec = importlib.util.spec_from_file_location("analyze_cabinet_ambient_correlation", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load()


def _tick(ts: str, rms: float, activity: float, fan: float, cpu: float) -> mod.Tick:
    return mod.Tick(
        t=mod.parse_db_timestamp(ts),
        rms=rms,
        activity=activity,
        fields={
            "measurements.fan_pct_max": fan,
            "pressures.fan": fan / 100.0,
            "pressures.cpu": cpu,
            "pressures.thermal": cpu * 0.9,
            "pressures.power": 0.5,
            "measurements.chassis_watts": 300.0,
            "measurements.disk_bytes_per_sec": 1e6,
            "measurements.temp_c_max": 40.0,
            "pressures.cabinet_climate_activity": 0.1,
            "pressures.cabinet_proximity_activity": 0.0,
        },
    )


def test_first_differences_handles_none():
    assert mod.first_differences([1.0, None, 3.0]) == [None, None]


def test_floor_stats_reports_rms_band():
    ticks = [_tick("2026-08-26 02:00:00+00", 4000.0, 0.2, 40.0, 0.3) for _ in range(5)]
    floor = mod.floor_stats(ticks)
    assert floor is not None
    assert floor.rms_median == 4000.0
    assert floor.rms_within_10pct_of_median_pct == 100.0


def test_delta_coupling_uses_changes_not_levels():
    ticks = [
        _tick("2026-08-26 02:00:00+00", 1000.0, 0.1, 40.0, 0.2),
        _tick("2026-08-26 02:00:30+00", 2000.0, 0.2, 40.0, 0.2),
        _tick("2026-08-26 02:01:00+00", 3000.0, 0.3, 40.0, 0.2),
        _tick("2026-08-26 02:01:30+00", 4000.0, 0.4, 40.0, 0.2),
    ]
    rows = mod.delta_coupling(ticks, max_lag_ticks=0)
    fan_row = next(r for r in rows if r.target == "measurements.fan_pct_max")
    cpu_row = next(r for r in rows if r.target == "pressures.cpu")
    assert fan_row.target_stdev == pytest.approx(0.0)
    assert fan_row.r_lag0 is None  # Δfan is flat — undefined correlation
    assert cpu_row.target_stdev == pytest.approx(0.0)


def test_spike_forensics_flags_fan_move():
    ticks = [
        _tick("2026-08-26 02:00:00+00", 4000.0, 0.1, 40.0, 0.2),
        _tick("2026-08-26 02:00:30+00", 4200.0, 0.9, 50.0, 0.25),
    ]
    floor = mod.floor_stats(ticks)
    spikes = mod.spike_forensics(ticks, floor, top_n=1)
    assert len(spikes) == 1
    assert "fan moved" in spikes[0].notes


def test_render_report_mentions_delta_coupling():
    floor = mod.FloorStats(
        n=2,
        rms_min=1.0,
        rms_max=2.0,
        rms_mean=1.5,
        rms_median=1.5,
        rms_stdev=0.5,
        rms_cv=0.33,
        rms_within_10pct_of_median_pct=100.0,
        activity_p10=0.1,
        activity_p50=0.2,
        activity_p90=0.3,
        fan_pct_min=30.0,
        fan_pct_max=40.0,
        fan_pct_stdev=1.0,
    )
    text = mod.render_report(
        node="athena",
        window_hours=24,
        grain_sec=30,
        span=(
            datetime(2026, 8, 26, 2, 0, tzinfo=timezone.utc),
            datetime(2026, 8, 26, 3, 0, tzinfo=timezone.utc),
        ),
        floor=floor,
        coupling=[
            mod.CouplingRow(
                target="measurements.fan_pct_max",
                n=10,
                target_stdev=0.5,
                r_lag0=0.0,
                best_lag_ticks=0,
                best_r=0.0,
            )
        ],
        spikes=[],
    )
    assert "ΔRMS coupling" in text
    assert "Level Pearson" in text
