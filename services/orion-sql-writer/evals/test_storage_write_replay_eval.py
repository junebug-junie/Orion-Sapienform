"""Eval: real write history replayed through the storage-write organ.

The fixture is a live pull (``storage_write_replay.py --live --write-fixture``,
2026-10-02): per-minute committed rows from each family's own table and lost
writes from ``bus_fallback_log``, classified by the writer's own classifier.
Every minute goes through the real emitter and the real reducer.

What it pins:
- calm reads as a measured 0.0, every minute, never as absent;
- the 2026-09-26 home-cooling serialization burst (863 rows, 0 landed until the
  fix at 04:52) reads 1.0 within two minutes and returns to 0.0 within the
  600 s span after the fix;
- the 2026-09-07 cockpit validation burst (182 rejects mixed with 191 commits)
  is visible and names the failure class.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("storage_write_replay_eval", HERE / "storage_write_replay.py")
replay = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = replay
_spec.loader.exec_module(replay)

DATA = json.loads(replay.FIXTURE.read_text())
PERIODS = {p["label"]: p for p in DATA["periods"]}


def _timeline(label):
    return replay.replay_period(PERIODS[label])


def test_calm_period_is_measured_zero_every_minute():
    tl = _timeline("calm_recent_3h")
    assert len(tl) >= 170
    assert all(t["pressure"] == 0.0 for t in tl), [t for t in tl if t["pressure"] != 0.0][:3]
    # real traffic behind the zero: hundreds of writes in every 10-minute span
    assert min(t["attempted"] for t in tl[10:]) > 1000


def test_home_cooling_burst_saturates_fast_and_clears_after_the_fix():
    tl = {t["minute"]: t for t in _timeline("home_cooling_serialization_2026-09-26")}
    assert tl["2026-09-26T03:38:00Z"]["pressure"] is None  # no writes yet: not measured, not calm
    assert tl["2026-09-26T03:39:00Z"]["pressure"] > 0.0  # first lost rows
    assert tl["2026-09-26T03:40:00Z"]["pressure"] == 1.0
    held = [t for m, t in tl.items() if "2026-09-26T03:40:00Z" <= m <= "2026-09-26T04:51:00Z"]
    assert held and all(t["pressure"] == 1.0 for t in held)
    # fix landed 04:52; the 600 s span empties of failures by 05:02
    assert 0.0 < tl["2026-09-26T04:55:00Z"]["pressure"] < 1.0
    assert tl["2026-09-26T05:02:00Z"]["pressure"] == 0.0
    assert tl["2026-09-26T05:29:00Z"]["pressure"] == 0.0


def test_cockpit_validation_burst_is_visible_and_classified():
    period = PERIODS["cockpit_validation_2026-09-07"]
    classes = {
        cls
        for fams in period["minutes"].values()
        for fam, counts in fams.items()
        if fam == "cockpit_turn_sighting"
        for cls in counts
    }
    assert "validation" in classes and "committed" in classes
    tl = _timeline("cockpit_validation_2026-09-07")
    nonzero = [t for t in tl if t["pressure"]]
    assert nonzero and max(t["pressure"] for t in nonzero) == 1.0
    # a lone reject never fires on its own (2-failure hysteresis)
    for t in tl:
        if t["pressure"]:
            assert t["failed"] >= 2
