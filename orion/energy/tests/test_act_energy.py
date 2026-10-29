"""Per-act energy arithmetic (orion/energy/act_energy.py).

Fixture values are real: circe gpu2 around the reverie image of 2026-10-29 16:40:56 UTC
(orion_biometrics raw samples, ~31 s apart) and the power-intent settlement for the same image
(mean 58.13 W, baseline 42.28 W, 60.74 s -> 0.2674 Wh).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.energy.act_energy import (
    card_energy,
    heat_c,
    idle_floor,
    incremental_joules,
    usd,
    watts_at,
)

T0 = datetime(2026, 10, 29, 16, 40, 0, tzinfo=timezone.utc)


def _at(sec: float) -> datetime:
    return T0 + timedelta(seconds=sec)


# circe gpu2, live: 16:40:20.75 41.41 W, 16:40:52.03 41.61 W, 16:41:23.29 216.11 W, 16:41:54.64 48.81 W
GPU2 = [(_at(20.751), 41.41), (_at(52.025), 41.61), (_at(83.287), 216.11), (_at(114.642), 48.81)]


def test_incremental_joules_is_the_power_intent_rule():
    # The settled image: (58.126 - 42.28) W x 60.742 s.
    j = incremental_joules(58.12589285714286, 42.28, 60.741578)
    assert j / 3600.0 == pytest.approx(0.26736, rel=1e-3)
    assert incremental_joules(30.0, 42.0, 10.0) == 0.0  # below baseline: nothing, never negative
    assert incremental_joules(-1.0, 0.0, 10.0) is None
    assert incremental_joules(50.0, 40.0, 0.0) is None
    assert incremental_joules(float("nan"), 40.0, 10.0) is None


def test_idle_floor_is_a_low_quantile_and_needs_history():
    values = [40.0] * 95 + [200.0] * 5
    assert idle_floor(values) == pytest.approx(40.0)
    assert idle_floor(values[:30]) is None  # half an hour of samples is not a floor
    assert idle_floor([float("nan")] * 100) is None


def test_watts_at_interpolates_and_refuses_holes():
    assert watts_at(GPU2, _at(52.025)) == pytest.approx(41.61)
    mid = watts_at(GPU2, _at(67.656))  # halfway between 41.61 and 216.11
    assert mid == pytest.approx((41.61 + 216.11) / 2, rel=1e-3)
    assert watts_at(GPU2, _at(200.0)) == pytest.approx(48.81)  # tail held within 90 s
    assert watts_at(GPU2, _at(300.0)) is None  # 185 s after the last sample: unknown, not 0
    hole = [(_at(0), 40.0), (_at(400), 40.0)]
    assert watts_at(hole, _at(200)) is None
    assert watts_at([], _at(0)) is None


def test_card_energy_alone_on_the_card():
    # Constant 140 W over 60 s on a card idling at 40 W: 100 W x 60 s = 1.6667 Wh.
    flat = [(_at(0), 140.0), (_at(30), 140.0), (_at(60), 140.0), (_at(90), 140.0)]
    got = card_energy(flat, idle_w=40.0, start=_at(10), end=_at(70))
    assert got.wh == pytest.approx(100.0 * 60.0 / 3600.0, rel=1e-6)
    assert got.mean_share == pytest.approx(1.0)
    assert got.samples_in_window == 2


def test_lease_shares_conserve_the_card_energy():
    """Two leases overlapping half the time: their shares sum to the card's above-idle energy."""
    flat = [(_at(0), 140.0), (_at(30), 140.0), (_at(60), 140.0), (_at(90), 140.0), (_at(120), 140.0)]
    a = (_at(0), _at(60))
    b = (_at(30), _at(90))
    ea = card_energy(flat, idle_w=40.0, start=a[0], end=a[1], others=[b])
    eb = card_energy(flat, idle_w=40.0, start=b[0], end=b[1], others=[a])
    whole = card_energy(flat, idle_w=40.0, start=_at(0), end=_at(90))
    assert ea.wh + eb.wh == pytest.approx(whole.wh, rel=1e-6)
    assert ea.wh == pytest.approx(100.0 * (30 + 15) / 3600.0, rel=1e-6)
    assert ea.mean_share == pytest.approx(0.75)


def test_card_energy_is_absent_without_readings():
    assert card_energy([], idle_w=40.0, start=_at(0), end=_at(60)) is None
    assert card_energy(GPU2, idle_w=40.0, start=_at(500), end=_at(560)) is None
    assert card_energy(GPU2, idle_w=40.0, start=_at(60), end=_at(60)) is None


def test_card_energy_on_the_live_reverie_lease():
    """The orion-thought diffusion lease 163c99b3 (16:40:56.08 -> 16:41:45.62) on circe gpu2,
    idle floor 40.8 W (p05 of gpu2 over the trailing 24 h). One 30 s sample caught the 216 W
    peak, so this one act reads high against the 1 Hz power-intent window (0.267 Wh): a 30 s
    cadence is an estimate per act, not a meter. The family check is
    services/orion-sql-writer/evals/act_energy_vs_power_intent.py."""
    got = card_energy(GPU2, idle_w=40.8, start=_at(56.077), end=_at(105.616))
    assert got is not None and got.samples_in_window == 1
    assert 1.0 < got.wh < 2.0


def test_heat_and_money():
    # circe 0.3111 C per 100 W of hourly-mean watts; 1 Wh in an hour raises the hour's mean by 1 W.
    assert heat_c(1.0, 0.3111) == pytest.approx(0.003111)
    assert heat_c(-1.0, 0.3111) == 0.0
    assert usd(1000.0, 0.1254564295292) == pytest.approx(0.12545643)
