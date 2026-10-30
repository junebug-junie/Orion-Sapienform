"""Organ 2 Budget (shadow) and the strain dial: one price per constraint, every tick."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.physiology.body import (
    BodyInputs,
    EnergyHour,
    EnergyInput,
    EnergyPrice,
    HeatInput,
    HostWattsInput,
    JuniperInput,
    LaneInput,
)
from orion.physiology.budget import BudgetConfig, price_state, strain_dial
from orion.physiology.heat_config import parse_heat_sensitivity
from orion.schemas.physiology import (
    PhysiologyConstraintReadingV1 as R,
    PhysiologyStateV1,
    parse_physiology_state,
)

NOW = datetime(2026, 10, 27, 3, 30, tzinfo=timezone.utc)   # 21:30 MDT on 10-26

# The 10-27 fit, as --write-config wrote it.
HEAT_DOC = {
    "schema": "heat_sensitivity.v1", "fitted_at": "2026-10-27T03:33:28+00:00",
    "window": ["2026-09-26T00:00:00Z", "2026-10-27T00:00:00Z"],
    "hosts": {
        "athena": {"point": 0.6863, "ci95": [0.2336, 1.0596], "horizon": "2h", "n": 707,
                   "fitted_at": "2026-10-27T03:33:28+00:00", "regressor": "compute_watts",
                   "baseline_watts_p10": 223.6, "fit_command": "fit"},
        "circe": {"point": 0.2985, "ci95": [0.1448, 0.4491], "horizon": "2h", "n": 707,
                  "fitted_at": "2026-10-27T03:33:28+00:00", "regressor": "compute_watts",
                  "baseline_watts_p10": 263.1, "fit_command": "fit"},
    },
}
HEAT_CFG = parse_heat_sensitivity(HEAT_DOC)
CFG = BudgetConfig(heat_config=HEAT_CFG, heat_config_error=None)


def heat(level):
    return R(constraint="heat", unit="c", level=level, budget=31.0, budget_kind="soft_line")


def lane(role, *, held, lease, slots, lent=0.0, queued=0):
    return R(constraint=f"lane:{role}", unit="slot_s", level=held, budget=slots * 3600.0, budget_kind="capacity",
             detail={"lease_s": lease, "slots": slots, "window_sec": 3600.0, "lent_overlap_s": lent,
                     "deferrable_queued": queued, "deferrable_waiting_sec": 21600.0})


def body(watts=None, energy=None):
    return BodyInputs(heat=HeatInput(True), energy=energy or EnergyInput(False), lanes=LaneInput(False),
                      juniper=JuniperInput(False),
                      host_watts=HostWattsInput(True, watts if watts is not None else {"athena": 244.5, "circe": 464.6, "hecate": 0.0}))


def run(readings, *, b=None, carry=None, now=NOW, cfg=CFG):
    st = PhysiologyStateV1(tick_id=f"physiology:{now.isoformat()}", generated_at=now, constraints=readings)
    out, carry = price_state(st, b or body(), carry, now, cfg)
    return {c.constraint: c for c in out.constraints}, out, carry


# --- heat ----------------------------------------------------------------------------------------

def test_heat_live_shape_under_the_line_is_slack_and_zero():
    """Live 10-27 03:21 watts (athena 244.5 W, circe 464.6 W compute: Orion's share ~0.75 C), on a
    cabinet 1 C cooler than that evening's 31.04 C: room ~29.3 C, under 31, so slack."""
    by, out, _ = run([heat(30.04)])
    p = by["heat"].price
    assert p.value == 0.0 and p.bind_state == "slack"
    assert p.use == pytest.approx(0.6863 * (244.5 - 223.6) / 100 + 0.2985 * (464.6 - 263.1) / 100, abs=1e-3)
    assert "hecate asleep" in p.why
    assert out.strain.value == 0.0


def test_heat_room_over_the_line_without_orion_is_exogenous_and_ramps_not_jumps():
    by, _, carry = run([heat(33.0)])          # room ~32.3 C
    p = by["heat"].price
    assert p.bind_state == "exogenous_bind" and "not my load" in p.why
    assert 0 < p.value <= 0.25 * 120 / 3600 + 1e-9    # one 120 s tick of the bounded ramp
    # Hysteresis: the room dips to 30.8 C (under 31, above 30.5): still bound.
    by2, _, _ = run([heat(31.0 + 0.55)], carry=carry, now=NOW + timedelta(seconds=120))
    assert by2["heat"].price.bind_state == "exogenous_bind"


def test_heat_orion_share_over_the_headroom_binds_and_climbs():
    """use > budget exactly when the cabinet is over the line while the room
    without Orion is under it: then Orion's own load is what crosses the line."""
    w = {"athena": 223.6 + 100.0, "circe": 263.1 + 400.0, "hecate": 0.0}   # share 0.686 + 1.194 = 1.88 C
    by, _, carry = run([heat(30.8)], b=body(w))                            # under the line
    p = by["heat"].price
    assert p.budget == pytest.approx(31.0 - (30.8 - p.use), abs=1e-3)
    assert p.use < p.budget and p.bind_state == "slack" and p.value == 0.0
    by2, _, _ = run([heat(31.5)], b=body(w), carry=carry, now=NOW + timedelta(seconds=120))  # over the line
    p2 = by2["heat"].price
    assert p2.use > p2.budget and p2.bind_state == "binding" and p2.value > 0


def test_heat_reads_the_raw_cabinet_temperature_no_sensor_offset():
    """Regression (2026-10-29): the cabinet sensor never moved, so the ~2 C rise after the 10-17
    vision bring-up is real heat. The heat price used to lower every reading after 10-17 00:35 by
    2 C ("cabinet 28.9 C (sensor offset -2.0 C ...) -> 26.9 C"). It must compare the raw reading.
    The fixture has no offsets, so the budget arithmetic here is a contract check, not the guard
    that would have caught the old code (only the ``why`` format differs on it). The real guards
    are the parser refusal and the checked-in-yaml tests below."""
    by, _, _ = run([heat(28.9)])
    p = by["heat"].price
    assert "sensor offset" not in p.why and "-> 26.9" not in p.why
    assert p.why.startswith("cabinet 28.9 C; ")
    room = 28.9 - p.use
    assert p.budget == pytest.approx(31.0 - room, abs=1e-6)


def test_heat_live_10_27_reading_is_orion_s_own_load_crossing_the_line():
    """The live 10-27 03:21 reading, raw: cabinet 31.04 C is over the 31 C line, the room without
    Orion (~30.3 C) is under it. So Orion's own load is what crosses the line: binding, not
    exogenous. With the old 2 C offset this read as slack at 29.04 C. (Documents the behaviour
    change; with this offset-free fixture the old code would also pass.)"""
    by, _, _ = run([heat(31.04)])
    p = by["heat"].price
    assert p.use > p.budget and p.bind_state == "binding"


def test_heat_config_with_measurement_offsets_is_refused_not_applied():
    """The offset mechanism is gone. A document that still carries one is malformed (heat reads
    no_reading with the reason), never silently applied and never silently ignored."""
    bad = dict(HEAT_DOC, measurement_offsets=[{"series": "athena.cabinet_temp_c",
                                               "since": "2026-10-17T00:35:00+00:00", "offset_c": 2.0}])
    with pytest.raises(ValueError, match="measurement_offsets"):
        parse_heat_sensitivity(bad)
    assert parse_heat_sensitivity(dict(HEAT_DOC, measurement_offsets=[])).hosts.keys() == {"athena", "circe"}


def test_checked_in_heat_config_has_no_measurement_offsets():
    from pathlib import Path

    from orion.physiology.heat_config import load_heat_sensitivity

    path = Path(__file__).resolve().parents[3] / "config" / "physiology" / "heat_sensitivity.yaml"
    cfg, err = load_heat_sensitivity(path)
    assert err is None and cfg is not None and {"athena", "circe"} <= set(cfg.hosts)
    assert "measurement_offsets" not in path.read_text()


def test_heat_unknowns_are_no_reading_never_zero():
    # No config at all.
    by, out, _ = run([heat(30.0)], cfg=BudgetConfig(heat_config=None, heat_config_error="missing:heat_sensitivity.yaml"))
    assert by["heat"].price.bind_state == "no_reading"
    assert out.strain.no_reading == ["heat"]
    # An entry older than 30 days.
    old = dict(HEAT_DOC, hosts={**HEAT_DOC["hosts"], "circe": {**HEAT_DOC["hosts"]["circe"],
                                                               "fitted_at": "2026-09-20T00:00:00+00:00"}})
    by, _, _ = run([heat(30.0)], cfg=BudgetConfig(heat_config=parse_heat_sensitivity(old), heat_config_error=None))
    assert by["heat"].price.bind_state == "no_reading" and "stale:circe" in by["heat"].price.why
    # hecate awake with no fitted sensitivity: unknown share, not 0.
    by, _, _ = run([heat(30.0)], b=body({"athena": 230.0, "circe": 300.0, "hecate": 400.0}))
    assert by["heat"].price.bind_state == "no_reading" and "hecate" in by["heat"].price.why
    # A fitted host with no watts this hour.
    by, _, _ = run([heat(30.0)], b=body({"athena": None, "circe": 300.0}))
    assert by["heat"].price.bind_state == "no_reading"


def test_no_reading_holds_the_last_price():
    carry = {"heat": {"price": 0.4, "bind": False, "ceiling_since": None, "at": (NOW - timedelta(seconds=120)).isoformat()}}
    by, _, out = run([R(constraint="heat", unit="c", level=None, no_reading="stale:cabinet")], carry=carry)
    assert by["heat"].price.value == 0.4 and by["heat"].price.bind_state == "no_reading"
    assert out["heat"]["price"] == 0.4


# --- energy --------------------------------------------------------------------------------------

def _energy(days, kwh_per_h, rate=0.1213):
    start = datetime(2026, 10, 1, 6, 0, tzinfo=timezone.utc)
    hours = [EnergyHour(start + timedelta(hours=i), kwh_per_h) for i in range(int(days * 24))]
    return EnergyInput(True, month_start=start, hours=hours, prices=[EnergyPrice(start, rate)])


def test_energy_floor_over_the_envelope_is_exogenous_bind():
    """Orion runs ~2 kWh/h always on: $100 is spent by the always-on floor alone (Fact 1)."""
    e = _energy(26, 2.0)
    to_date = 26 * 24 * 2.0 * 0.1213
    reading = R(constraint="energy", unit="usd", level=round(to_date, 4), budget=100.0, budget_kind="envelope")
    by, out, _ = run([reading], b=body(energy=e))
    p = by["energy"].price
    assert p.bind_state == "exogenous_bind"
    # Wording (2026-10-30): the always-on floor is Orion's own body at rest, never "not by what I do".
    assert "not by what I do" not in p.why and "stays switched on" not in p.why
    assert "my body at rest costs about $177/month, more than the $100 envelope by itself" in p.why
    assert "my activity above rest adds about $" in p.why and "(pace $0/month)" in p.why
    # The report's money.floor_usd still parses the unchanged mechanism text.
    assert "the always-on floor alone ($0.243/h) reaches $" in p.why
    assert p.use > 100.0 and p.value <= 0.25 * 120 / 3600 + 1e-9
    assert out.strain.exogenous_bind == ["energy"] and out.strain.value is None


def test_energy_under_a_generous_envelope_is_slack():
    e = _energy(10, 0.5)
    reading = R(constraint="energy", unit="usd", level=round(10 * 24 * 0.5 * 0.1213, 4), budget=500.0, budget_kind="envelope")
    by, _, _ = run([reading], b=body(energy=e), now=datetime(2026, 10, 11, 6, 0, tzinfo=timezone.utc))
    assert by["energy"].price.bind_state == "slack" and by["energy"].price.value == 0.0


def test_energy_without_an_envelope_or_a_day_of_hours_is_no_reading():
    reading = R(constraint="energy", unit="usd", level=5.0)
    by, _, _ = run([reading], b=body(energy=_energy(10, 1.0)))
    assert by["energy"].price.bind_state == "no_reading"
    reading = R(constraint="energy", unit="usd", level=1.0, budget=100.0, budget_kind="envelope")
    by, _, _ = run([reading], b=body(energy=_energy(0.5, 1.0)), now=datetime(2026, 10, 1, 18, 0, tzinfo=timezone.utc))
    assert by["energy"].price.bind_state == "no_reading"


# --- lanes, juniper, claude ----------------------------------------------------------------------

def test_lane_prices_lease_demand_not_the_waiting_backlog():
    """Live 10-27: lane:agent full (3,600 of 3,600) with 5 holds waiting > 6 h. No gap borrowers,
    so demand = capacity: the price does not rise. Hunger is shown, not priced."""
    by, _, _ = run([lane("agent", held=3599.97, lease=3599.97, slots=1, queued=5)])
    p = by["lane:agent"].price
    assert p.value == 0.0 and p.bind_state == "slack" and "5 deferrable waiting (hunger, not priced)" in p.why
    # Gap borrowers push demand over the seat: the price climbs.
    by, _, _ = run([lane("agent", held=3600.0, lease=3993.7, slots=1, lent=393.7)])
    assert by["lane:agent"].price.bind_state == "binding" and by["lane:agent"].price.value > 0


def test_swapped_out_seat_that_served_its_leases_while_loaded_is_slack_and_decays():
    """Patch C1 fix (b). Live 10-27 05:42: lane:diffusion at exogenous_bind 0.095 from a painting
    that ran 04:31 while the seat was loaded; gpu2 was swapped back to agent-gpu2 since, so the
    pool read 0 slots. That is Orion's own finished work, not the world pushing back: no price
    rise, no strain, and an old price decays."""
    r = R(constraint="lane:diffusion", unit="slot_s", level=54.7, budget=0.0, budget_kind="capacity",
          detail={"lease_s": 54.7, "slots": 0, "window_sec": 3600.0, "served_while_loaded": True})
    carry = {"lane:diffusion": {"price": 0.095254, "bind": True, "ceiling_since": None,
                                "at": (NOW - timedelta(seconds=120)).isoformat()}}
    by, out, _ = run([r], carry=carry)
    p = by["lane:diffusion"].price
    assert p.bind_state == "slack" and p.value < 0.095254 and "ran while it was loaded" in p.why
    # The old (buggy) price only decays: the dial never rises from it.
    assert out.strain.exogenous_bind == [] and out.strain.value <= 0.095254
    carry2 = {"lane:diffusion": {"price": 0.0, "bind": False, "ceiling_since": None,
                                 "at": (NOW - timedelta(seconds=120)).isoformat()}}
    by, out, _ = run([r], carry=carry2)
    assert by["lane:diffusion"].price.value == 0.0 and out.strain.value == 0.0


def test_open_lease_on_a_zero_slot_seat_is_no_reading_never_strain():
    r = R(constraint="lane:diffusion", unit="slot_s", level=300.0, budget=0.0, budget_kind="capacity",
          detail={"lease_s": 300.0, "slots": 0, "window_sec": 3600.0, "held_without_live_slot": True})
    by, out, _ = run([r])
    p = by["lane:diffusion"].price
    assert p.bind_state == "no_reading" and p.value == 0.0
    assert out.strain.no_reading == ["lane:diffusion"] and out.strain.exogenous_bind == []


def test_juniper_cap_filled_by_orions_own_outreach_is_self_spent_not_strain():
    """Patch C1 fix (a). Live 10-27 05:42: strain named attention:juniper (4 of 4 sends, all Orion's
    own outreach). Orion spending its own budget is not the world pushing back: the price stays
    (Choice charges it), the dial skips it and lists it apart."""
    r = R(constraint="attention:juniper", unit="messages", level=4.0, budget=4.0, budget_kind="daily_cap")
    late = datetime(2026, 10, 27, 5, 42, tzinfo=timezone.utc)           # 23:42 MDT
    carry = {"attention:juniper": {"price": 0.3, "bind": False, "ceiling_since": None,
                                   "at": (late - timedelta(seconds=120)).isoformat()}}
    by, out, _ = run([r, lane("agent", held=600.0, lease=600.0, slots=1)], carry=carry, now=late)
    assert by["attention:juniper"].price.value > 0.0
    assert out.strain.self_spent == ["attention:juniper"]
    assert out.strain.constraint is None and out.strain.value == 0.0 and "my own spending" in out.strain.why


def test_juniper_day_end_pace_over_the_cap_binds():
    r = R(constraint="attention:juniper", unit="messages", level=3.0, budget=4.0, budget_kind="daily_cap")
    noon = datetime(2026, 10, 26, 18, 0, tzinfo=timezone.utc)            # 12:00 MDT
    by, _, _ = run([r], now=noon)
    assert by["attention:juniper"].price.use == pytest.approx(6.0) and by["attention:juniper"].price.bind_state == "binding"


def test_claude_limited_is_exogenous_clear_is_slack_unknown_is_no_reading():
    lim = R(constraint="claude", unit="limit_events", level=2.0, state="limited", budget=0.0, budget_kind="not_limited")
    clear = R(constraint="claude", unit="limit_events", level=0.0, state="clear", budget=0.0, budget_kind="not_limited")
    unk = R(constraint="claude", unit="limit_events", level=0.0, state="unknown", budget=0.0, budget_kind="not_limited")
    assert run([lim])[0]["claude"].price.bind_state == "exogenous_bind"
    assert run([clear])[0]["claude"].price.bind_state == "slack"
    assert run([unk])[0]["claude"].price.bind_state == "no_reading"


# --- lifecycle: ramp, return to rest, stuck check, strain ------------------------------------------

def test_lifecycle_binds_then_returns_to_zero_within_one_horizon():
    """R3 on the live tick: a lane over-asked for 30 min, then slack. The price rises (bounded
    steps), then reaches 0 within one horizon (1 h) of clearing."""
    carry, t, prices = None, NOW, []
    for i in range(15):
        by, _, carry = run([lane("agent", held=3600.0, lease=7200.0, slots=1, lent=3600.0)], carry=carry, now=t)
        prices.append(by["lane:agent"].price.value)
        t += timedelta(seconds=120)
    peak = prices[-1]
    assert peak > 0 and all(b - a <= 1.0 * 120 / 3600 + 1e-5 for a, b in zip(prices, prices[1:]))
    cleared = None
    for i in range(40):
        by, _, carry = run([lane("agent", held=1000.0, lease=1000.0, slots=1)], carry=carry, now=t)
        t += timedelta(seconds=120)
        if by["lane:agent"].price.value == 0.0:
            cleared = (i + 1) * 120
            break
    assert cleared is not None and cleared <= 3600


def test_a_price_pinned_at_its_ceiling_unflagged_past_a_horizon_is_flagged_stuck():
    carry = {"lane:agent": {"price": 1.0, "bind": False, "ceiling_since": (NOW - timedelta(hours=2)).timestamp(),
                            "at": (NOW - timedelta(seconds=120)).isoformat()}}
    by, out, _ = run([lane("agent", held=3600.0, lease=9000.0, slots=1, lent=5400.0)], carry=carry)
    assert by["lane:agent"].price.bind_state == "stuck_at_ceiling"
    assert "price_stuck_at_ceiling:lane:agent" in out.warnings and out.strain.stuck == ["lane:agent"]


def test_strain_is_the_largest_live_price_over_its_ceiling_exogenous_listed_apart():
    def priced(name, value, state):
        return R(constraint=name, unit="x", level=1.0,
                 price={"value": value, "ceiling": 1.0, "bind_state": state, "why": ""})
    s = strain_dial([priced("heat", 0.9, "exogenous_bind"), priced("lane:agent", 0.3, "binding"),
                     priced("lane:fast", 0.1, "slack"), priced("claude", 0.5, "no_reading")])
    assert (s.value, s.constraint, s.exogenous_bind, s.no_reading) == (0.3, "lane:agent", ["heat"], ["claude"])
    rest = strain_dial([priced("lane:fast", 0.0, "slack")])
    assert rest.value == 0.0 and rest.constraint is None
    assert strain_dial([]).value is None


def test_priced_state_round_trips_the_schema_and_the_stored_row_keeps_the_prices():
    by, out, _ = run([heat(30.04), lane("agent", held=3599.97, lease=3599.97, slots=1)])
    again = parse_physiology_state(out.model_dump_json())
    assert again is not None and again.strain is not None
    assert {c.constraint: c.price.bind_state for c in again.constraints} == {"heat": "slack", "lane:agent": "slack"}
    # sql-writer persists ``constraints`` only; the dial is recomputable from them.
    assert strain_dial(again.constraints) == again.strain


# --- review fixes -------------------------------------------------------------------------------

def test_a_constraint_absent_for_a_tick_keeps_its_price():
    """Review: the pool state unread -> only the bare 'lane' placeholder. lane:fast must come back
    at its old price, not 0 (unknown is not reset)."""
    _, _, carry = run([lane("fast", held=3600.0, lease=9000.0, slots=1, lent=5400.0)],
                      carry={"lane:fast": {"price": 0.9, "bind": False, "ceiling_since": None,
                                           "at": (NOW - timedelta(seconds=120)).isoformat()}})
    t1 = NOW + timedelta(seconds=120)
    by, out, carry = run([R(constraint="lane", unit="slot_s", level=None, no_reading="read_failed:gpu_pool_events")],
                         carry=carry, now=t1)
    assert carry["lane:fast"]["price"] > 0.9
    assert "lane" in out.strain.no_reading            # unknown lanes are named, not hidden as calm
    by, _, _ = run([lane("fast", held=1000.0, lease=1000.0, slots=1)], carry=carry, now=t1 + timedelta(seconds=120))
    assert by["lane:fast"].price.value > 0.85         # resumed from where it was, decaying


def test_juniper_one_early_message_is_not_a_days_pace_and_dst_elapsed_is_real():
    r = R(constraint="attention:juniper", unit="messages", level=1.0, budget=4.0, budget_kind="daily_cap")
    early = datetime(2026, 10, 26, 7, 6, tzinfo=timezone.utc)                 # 01:06 MDT
    by, _, _ = run([r], now=early)
    assert by["attention:juniper"].price.use == 1.0 and by["attention:juniper"].price.bind_state == "slack"
    # 2026-11-01 (DST ends): 12:00 MST is 13 real hours after local midnight (06:00Z).
    dst = datetime(2026, 11, 1, 19, 0, tzinfo=timezone.utc)
    by, _, _ = run([R(constraint="attention:juniper", unit="messages", level=13.0, budget=40.0, budget_kind="daily_cap")], now=dst)
    assert by["attention:juniper"].price.use == pytest.approx(13.0 * 24 / 13.0)


def test_heat_config_near_its_age_limit_warns_before_going_no_reading():
    _, out, _ = run([heat(30.0)], now=datetime(2026, 11, 21, tzinfo=timezone.utc))
    assert any(w.startswith("heat_sensitivity_stale_in_days:athena") for w in out.warnings)
    by, out, _ = run([heat(30.0)], now=datetime(2026, 11, 27, tzinfo=timezone.utc))
    assert "heat_sensitivity_stale:athena" in out.warnings and by["heat"].price.bind_state == "no_reading"


def test_energy_zero_envelope_is_no_reading_and_the_filling_hour_is_left_out():
    zero = R(constraint="energy", unit="usd", level=5.0, budget=0.0, budget_kind="envelope")
    assert run([zero], b=body(energy=_energy(10, 1.0)))[0]["energy"].price.bind_state == "no_reading"
    # 24 complete hours + the hour still filling at a tenth of the rate: the floor is not dragged down.
    start = datetime(2026, 10, 1, 6, 0, tzinfo=timezone.utc)
    hours = [EnergyHour(start + timedelta(hours=i), 1.0) for i in range(24)] + [EnergyHour(start + timedelta(hours=24), 0.1)]
    e = EnergyInput(True, month_start=start, hours=hours, prices=[EnergyPrice(start, 0.1)])
    reading = R(constraint="energy", unit="usd", level=2.41, budget=1000.0, budget_kind="envelope")
    by, _, _ = run([reading], b=body(energy=e), now=start + timedelta(hours=24, minutes=10))
    assert "$0.100/h" in by["energy"].price.why and "floor alone ($0.100/h)" in by["energy"].price.why


def test_claude_unknown_shows_no_budget_not_zero():
    unk = R(constraint="claude", unit="limit_events", level=0.0, state="unknown", budget=0.0, budget_kind="not_limited")
    assert run([unk])[0]["claude"].price.budget is None
