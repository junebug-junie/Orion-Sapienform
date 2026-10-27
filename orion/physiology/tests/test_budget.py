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
    "measurement_offsets": [{"series": "athena.cabinet_temp_c", "since": "2026-10-17T00:35:00+00:00",
                             "offset_c": 2.0, "change_event": "cabinet sensor geometry changed (vision bring-up)",
                             "source": "#2780"}],
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
    """Live 10-27 03:21: cabinet 31.04 C raw (= 29.04 C after #2780's 2 C sensor offset), athena
    244.5 W, circe 464.6 W compute: Orion's share ~0.75 C, room ~28.3 C, far under 31."""
    by, out, _ = run([heat(31.04)])
    p = by["heat"].price
    assert p.value == 0.0 and p.bind_state == "slack"
    assert p.use == pytest.approx(0.6863 * (244.5 - 223.6) / 100 + 0.2985 * (464.6 - 263.1) / 100, abs=1e-3)
    assert "sensor offset -2.0 C" in p.why and "hecate asleep" in p.why
    assert out.strain.value == 0.0


def test_heat_room_over_the_line_without_orion_is_exogenous_and_ramps_not_jumps():
    by, _, carry = run([heat(35.0)])          # 33 C after the offset, room ~32.3 C
    p = by["heat"].price
    assert p.bind_state == "exogenous_bind" and "not my load" in p.why
    assert 0 < p.value <= 0.25 * 120 / 3600 + 1e-9    # one 120 s tick of the bounded ramp
    # Hysteresis: the room dips to 30.8 C (under 31, above 30.5): still bound.
    by2, _, _ = run([heat(31.0 + 2.0 + 0.55)], carry=carry, now=NOW + timedelta(seconds=120))
    assert by2["heat"].price.bind_state == "exogenous_bind"


def test_heat_orion_share_over_the_headroom_binds_and_climbs():
    """use > budget exactly when the cabinet (offset removed) is over the line while the room
    without Orion is under it: then Orion's own load is what crosses the line."""
    w = {"athena": 223.6 + 100.0, "circe": 263.1 + 400.0, "hecate": 0.0}   # share 0.686 + 1.194 = 1.88 C
    by, _, carry = run([heat(32.8)], b=body(w))                            # 30.8 C: under the line
    p = by["heat"].price
    assert p.budget == pytest.approx(31.0 - (30.8 - p.use), abs=1e-3)
    assert p.use < p.budget and p.bind_state == "slack" and p.value == 0.0
    by2, _, _ = run([heat(33.5)], b=body(w), carry=carry, now=NOW + timedelta(seconds=120))  # 31.5 C
    p2 = by2["heat"].price
    assert p2.use > p2.budget and p2.bind_state == "binding" and p2.value > 0


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
    assert p.bind_state == "exogenous_bind" and "not by what I do" in p.why
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


def test_lane_with_no_live_slot_still_holding_is_exogenous():
    r = R(constraint="lane:diffusion", unit="slot_s", level=546.5, budget=0.0, budget_kind="capacity",
          detail={"lease_s": 546.5, "slots": 0, "window_sec": 3600.0, "held_without_live_slot": True})
    by, _, _ = run([r])
    assert by["lane:diffusion"].price.bind_state == "exogenous_bind"


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
    by, out, _ = run([heat(31.04), lane("agent", held=3599.97, lease=3599.97, slots=1)])
    again = parse_physiology_state(out.model_dump_json())
    assert again is not None and again.strain is not None
    assert {c.constraint: c.price.bind_state for c in again.constraints} == {"heat": "slack", "lane:agent": "slack"}
    # sql-writer persists ``constraints`` only; the dial is recomputable from them.
    assert strain_dial(again.constraints) == again.strain
