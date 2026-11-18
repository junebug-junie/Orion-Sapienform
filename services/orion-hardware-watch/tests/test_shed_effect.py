"""D11 reducer (orion/hardware_watch/shed_effect.py): delta after a shed, against a matched control."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.hardware_watch.rules import TempPoint
from orion.hardware_watch.shed_effect import ShedEpisode, control_deltas, shed_effect, summarize

T0 = datetime(2026, 10, 5, 0, 0, tzinfo=timezone.utc)


def series(fn, hours=6, step=30):
    return [TempPoint(T0 + timedelta(seconds=s), fn(s)) for s in range(0, hours * 3600, step)]


def test_flat_cabinet_gives_zero_delta_and_zero_effect():
    cab = series(lambda s: 31.0)
    ep = ShedEpisode("test", "e1", "cabinet_hot", T0 + timedelta(hours=3), T0 + timedelta(hours=3, minutes=40))
    row = shed_effect(ep, cab, series(lambda s: 300.0), [ep])
    assert row["delta_c_15m"] == 0.0 and row["effect_c_15m"] == 0.0 and row["control_n_15m"] > 10


def test_a_cooling_shed_against_a_flat_control_reads_negative():
    start = 3 * 3600
    cab = series(lambda s: 31.0 - (0.04 * (s - start) / 60 if start <= s <= start + 1800 else 0.0))
    gpu = series(lambda s: 100.0 if start <= s <= start + 2400 else 600.0)
    ep = ShedEpisode("test", "e1", "cabinet_hot", T0 + timedelta(seconds=start), T0 + timedelta(seconds=start + 2400))
    row = shed_effect(ep, cab, gpu, [ep])
    assert row["effect_c_15m"] < -0.5 and row["gpu_w_drop"] > 400


def test_control_excludes_shed_windows_and_needs_a_matching_temperature():
    cab = series(lambda s: 31.0)
    ep = ShedEpisode("test", "e1", "x", T0 + timedelta(hours=1), T0 + timedelta(hours=5))
    # grid runs 00:30..05:44; the episode blocks 00:30..05:15 -> only 05:20..05:40 remain
    assert len(control_deltas(cab, [ep], near_c=31.0, band_c=0.5, minutes=15)) == 5
    assert len(control_deltas(cab, [], near_c=31.0, band_c=0.5, minutes=15)) == 63   # 00:30..05:40 every 5 min
    assert control_deltas(cab, [], near_c=25.0, band_c=0.5, minutes=15) == []


def test_no_readings_is_none_not_zero():
    ep = ShedEpisode("test", "e1", "x", T0, T0 + timedelta(minutes=30))
    row = shed_effect(ep, [], [], [ep])
    assert row["delta_c_15m"] is None and row["effect_c_15m"] is None and row["gpu_w_drop"] is None
    assert summarize([row]) == {"episodes": 1, "measured_15m": 0, "mean_effect_c_15m": None, "enough_to_judge": False}
