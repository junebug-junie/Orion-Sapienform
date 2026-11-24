"""Replay eval mechanics (#3821): the tick replay decays like curiosity does, and the joint rebuild
folds through the real window."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_curiosity_seed_bar_replay_eval as ev  # noqa: E402

T0 = datetime(2026, 11, 24, 0, 0, tzinfo=timezone.utc)


def test_tick_seeds_decay_and_horizon():
    # one write of 1.0: decayed >= 2/3 while age <= 600 s -> ticks at 0..600 s = 11 ticks
    per_day, fresh, seeded = ev.tick_seeds([(T0, 1.0)], 2.0 / 3.0, T0, T0 + timedelta(hours=1))
    assert per_day == {"2026-11-24": 11}
    assert len(fresh) == 1 and len(seeded) == 1


def test_calm_writes_never_seed():
    writes = [(T0 + timedelta(minutes=5 * k), 0.0) for k in range(12)]
    per_day, fresh, seeded = ev.tick_seeds(writes, 0.55, T0, T0 + timedelta(hours=1))
    assert per_day == {} and not seeded and fresh


def test_joint_proxy_folds_one_surprising_outcome():
    rows = [("family_a_room", 5, T0, T0 + timedelta(minutes=5), T0 + timedelta(minutes=10), 3.0, 4, 1)]
    out = ev.joint_proxy(rows, "family_a_room")
    assert len(out) == 1 and out[0][1] > 0.9  # z ~ 3 -> value ~ 1
    assert ev.joint_proxy(rows, "family_a") == []


def test_eval_constants_match_live_curiosity():
    """The replay mirrors the live decay horizon and tick interval; fail if either drifts."""
    from orion.substrate.pressure import PressureConfig

    assert ev.HORIZON_S == PressureConfig().prediction_error_decay_horizon_seconds
    settings_src = (Path(__file__).resolve().parents[1] / "app" / "settings.py").read_text()
    block = settings_src.split("endogenous_curiosity_tick_interval_sec", 1)[1][:120]
    assert f"{ev.TICK.total_seconds():.1f}" in block
