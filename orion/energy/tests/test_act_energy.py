"""Per-act energy arithmetic (orion/energy/act_energy.py).

Fixture values are real: circe gpu2 around the reverie image of 2026-10-29 16:40:56 UTC
(orion_biometrics raw samples, ~31 s apart) and the power-intent settlement for the same image
(mean 58.13 W, baseline 42.28 W, 60.74 s -> 0.2674 Wh).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.energy.act_energy import (
    HostFloors,
    HostSeries,
    LeaseSpan,
    attribute_host_window,
    card_energy,
    machine_energy,
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


# --- whole machine above rest ---------------------------------------------------------------------

def _flat(w: float, start: float = -60.0, end: float = 3700.0, step: float = 30.0):
    n = int((end - start) / step) + 1
    return [(_at(start + i * step), w) for i in range(n)]


# circe, live hourly means (orion_biometrics_summary): rest 2026-10-29 20-21Z chassis ~600 W,
# GPU total ~210 W, CPU ~52 W; busy from 22Z chassis ~1150 W, GPU ~430 W, CPU ~125 W.
# CPU floor here = 52 W so the CPU part matches the 2026-10-30 report (live code reads the row's
# fitted cpu_idle_watts instead; the arithmetic is the same).
CIRCE_REST = HostFloors(chassis_w=600.0, card_idle_w={0: 52.5, 1: 52.5, 2: 52.5, 3: 52.5})
CIRCE_BUSY = HostSeries(chassis=_flat(1150.0), cpu_above=_flat(125.0 - 52.0),
                        cards={i: _flat(107.5) for i in range(4)})
DAY = {T0.date(): CIRCE_REST}


def _all_four_cards(series, metered=True):
    return {i: [LeaseSpan(_at(0), _at(3600), metered)] for i in range(4)}


def test_regression_gpu_only_missed_about_half_of_circe_busy_hour():
    """Regression (2026-10-30): the GPU-only meter charged 220 W of a 550 W chassis rise.

    One act holding all four cards alone for the busy hour: GPU above idle 4 x 55 = 220 W, CPU above
    rest 73 W, chassis above rest 550 W. The act gets its cards' share of the chassis rise:
    220 x 550 / 293 = 413 W. The CPU's 137 W share is not the act's: it stays unattributed."""
    whole = [machine_energy(CIRCE_BUSY, DAY, card=i, start=_at(0), end=_at(3600), step_s=10.0) for i in range(4)]
    act_wh = sum(w.wh for w in whole)
    gpu_wh = sum(w.wh_gpu for w in whole)
    assert gpu_wh == pytest.approx(220.0, rel=1e-6)            # the old meter: 40% of the rise
    assert act_wh == pytest.approx(220.0 * 550.0 / 293.0, rel=1e-6)
    hour = attribute_host_window(CIRCE_BUSY, DAY, _all_four_cards(CIRCE_BUSY), start=_at(0), end=_at(3600),
                                 step_s=10.0)
    assert hour.above_rest_wh == pytest.approx(550.0, rel=1e-6) and hour.covered_s == pytest.approx(3600.0)
    assert hour.rest_wh == pytest.approx(600.0)
    assert hour.metered_wh == pytest.approx(act_wh, rel=1e-9)  # the two code paths agree
    assert hour.unattributed_wh == pytest.approx(73.0 * 550.0 / 293.0, rel=1e-6)  # the CPU share
    assert hour.metered_wh + hour.unattributed_wh == pytest.approx(hour.above_rest_wh, rel=1e-9)  # closure


def test_shares_of_every_lease_close_on_the_measured_total():
    """Two leases share card 2 (one has no act), card 0 runs unleased, a sponsored container burns
    CPU: every watt above rest lands exactly once."""
    series = HostSeries(chassis=_flat(900.0), cpu_above=_flat(28.0),
                        cards={0: _flat(90.0), 1: _flat(52.5), 2: _flat(200.0), 3: _flat(52.5)},
                        sponsored_cpu=_flat(20.0))
    a = (_at(0), _at(600))
    b = (_at(300), _at(900))
    ea = machine_energy(series, DAY, card=2, start=a[0], end=a[1], others=[b], step_s=1.0)
    eb = machine_energy(series, DAY, card=2, start=b[0], end=b[1], others=[a], step_s=1.0)
    hour = attribute_host_window(series, DAY, {2: [LeaseSpan(*a, True), LeaseSpan(*b, False)]},
                                 start=_at(0), end=_at(900), step_s=1.0)
    assert hour.metered_wh == pytest.approx(ea.wh, rel=1e-9)
    assert hour.leased_unmetered_wh == pytest.approx(eb.wh, rel=1e-9)
    load = (90.0 - 52.5) + (200.0 - 52.5) + 28.0
    assert hour.above_rest_wh == pytest.approx(300.0 * 900.0 / 3600.0, rel=1e-9)
    assert hour.sponsored_wh == pytest.approx(300.0 * 20.0 / load * 900.0 / 3600.0, rel=1e-9)
    unleased = 300.0 * ((90.0 - 52.5) + 8.0) / load * 900.0 / 3600.0
    assert hour.unattributed_wh == pytest.approx(eb.wh + unleased, rel=1e-9)
    assert hour.sponsored_wh + hour.metered_wh + hour.unattributed_wh == pytest.approx(hour.above_rest_wh)


def test_whole_machine_is_absent_without_a_chassis_or_cpu_reading_or_a_floor():
    no_chassis = HostSeries(chassis=[], cpu_above=CIRCE_BUSY.cpu_above, cards=CIRCE_BUSY.cards)
    assert machine_energy(no_chassis, DAY, card=2, start=_at(0), end=_at(60)) is None
    no_cpu = HostSeries(chassis=CIRCE_BUSY.chassis, cpu_above=[], cards=CIRCE_BUSY.cards)
    assert machine_energy(no_cpu, DAY, card=2, start=_at(0), end=_at(60)) is None
    assert machine_energy(CIRCE_BUSY, {}, card=2, start=_at(0), end=_at(60)) is None
    # Another card on the host has no reading: the split's denominator is unknown, so no number.
    hole = HostSeries(chassis=CIRCE_BUSY.chassis, cpu_above=CIRCE_BUSY.cpu_above, cards={2: CIRCE_BUSY.cards[2]})
    assert machine_energy(hole, DAY, card=2, start=_at(0), end=_at(60)) is None
    # The hourly split still measures the chassis, and says the split was unknown.
    hour = attribute_host_window(no_cpu, DAY, _all_four_cards(no_cpu), start=_at(0), end=_at(60))
    assert hour.above_rest_wh == pytest.approx(550.0 / 60.0) and hour.split_unknown_s == pytest.approx(60.0)
    assert hour.metered_wh == 0.0 and hour.unattributed_wh == pytest.approx(hour.above_rest_wh)
    # No chassis at all: nothing covered, nothing invented.
    hour = attribute_host_window(no_chassis, DAY, {}, start=_at(0), end=_at(60))
    assert hour.covered_s == 0.0 and hour.above_rest_wh == 0.0 and hour.rest_wh == 0.0


def test_nothing_above_rest_charges_nothing_and_chassis_below_rest_is_zero():
    calm = HostSeries(chassis=_flat(590.0), cpu_above=_flat(0.0), cards={i: _flat(52.5) for i in range(4)})
    got = machine_energy(calm, DAY, card=2, start=_at(0), end=_at(60))
    assert got.wh == 0.0 and got.wh_gpu == 0.0
    # Card busy but the wall meter reads at rest (meter lag): GPU part counts, whole machine 0.
    lag = HostSeries(chassis=_flat(590.0), cpu_above=_flat(0.0), cards={**calm.cards, 2: _flat(152.5)})
    got = machine_energy(lag, DAY, card=2, start=_at(0), end=_at(60))
    assert got.wh == 0.0 and got.wh_gpu == pytest.approx(100.0 * 60 / 3600.0)


def test_floors_are_keyed_by_utc_day_across_midnight():
    t = datetime(2026, 10, 29, 23, 59, 0, tzinfo=timezone.utc)
    pts = [(t + timedelta(seconds=30 * i), 800.0) for i in range(-2, 8)]
    series = HostSeries(chassis=pts, cpu_above=[(p[0], 0.0) for p in pts], cards={})
    floors = {t.date(): HostFloors(600.0, {}), (t + timedelta(days=1)).date(): HostFloors(700.0, {})}
    hour = attribute_host_window(series, floors, {}, start=t, end=t + timedelta(seconds=120))
    assert hour.above_rest_wh == pytest.approx((200.0 * 60 + 100.0 * 60) / 3600.0)
    hour = attribute_host_window(series, {t.date(): HostFloors(600.0, {})}, {}, start=t, end=t + timedelta(seconds=120))
    assert hour.covered_s == pytest.approx(60.0)  # no floors for the second day: skipped, not 0 W


def test_many_leases_sweep_matches_brute_force():
    """The pointer sweep over sorted leases equals testing every lease at every step."""
    series = HostSeries(chassis=_flat(900.0), cpu_above=_flat(10.0), cards={2: _flat(200.0)})
    spans = [LeaseSpan(_at(i * 7), _at(i * 7 + 11), i % 3 == 0) for i in range(100)]
    floors = {T0.date(): HostFloors(600.0, {2: 52.5})}
    hour = attribute_host_window(series, floors, {2: list(reversed(spans))}, start=_at(0), end=_at(720))
    ratio = 300.0 / (147.5 + 10.0)
    expect_m = expect_u = 0.0
    for k in range(720):
        t = _at(k + 0.5)
        held = [x.metered for x in spans if x.start <= t < x.end]
        if held:
            w = 147.5 * ratio / 3600.0
            expect_m += w * sum(held) / len(held)
            expect_u += w * (1 - sum(held) / len(held))
    assert hour.metered_wh == pytest.approx(expect_m, rel=1e-9)
    assert hour.leased_unmetered_wh == pytest.approx(expect_u, rel=1e-9)


def test_near_zero_load_cannot_take_more_than_the_ratio_cap():
    """Review finding: rest sits under calm chassis, so a card 1 W above idle at near-zero load took
    the whole surplus (live per-sample ratio up to 78). The cap leaves the excess unattributed."""
    from orion.energy.act_energy import RATIO_CAP

    series = HostSeries(chassis=_flat(700.0), cpu_above=_flat(0.0), cards={2: _flat(53.5)})
    floors = {T0.date(): HostFloors(600.0, {2: 52.5})}
    got = machine_energy(series, floors, card=2, start=_at(0), end=_at(60))
    assert got.wh == pytest.approx(1.0 * RATIO_CAP * 60.0 / 3600.0)
    hour = attribute_host_window(series, floors, {2: [LeaseSpan(_at(0), _at(60), True)]}, start=_at(0), end=_at(60))
    assert hour.metered_wh == pytest.approx(got.wh)
    assert hour.unattributed_wh == pytest.approx((100.0 - RATIO_CAP) * 60.0 / 3600.0)
