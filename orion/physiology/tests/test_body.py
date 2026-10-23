"""Organ 1 Body: pure reducer tests. Every missing or stale source is no_reading, never 0."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.physiology.body import (
    BodyConfig,
    BodyInputs,
    ClaudeObservation,
    EnergyHour,
    EnergyInput,
    EnergyPrice,
    HeatInput,
    JuniperInput,
    LaneInput,
    LeaseEvent,
    build_physiology_state,
    claude_reading,
    energy_reading,
    heat_reading,
    juniper_reading,
    lane_readings,
)
from orion.schemas.physiology import PhysiologyStateV1, parse_physiology_state

NOW = datetime(2026, 10, 23, 18, 0, tzinfo=timezone.utc)
CFG = BodyConfig()
MONTH = datetime(2026, 10, 1, 6, 0, tzinfo=timezone.utc)   # 00:00 America/Denver


# --- heat ---------------------------------------------------------------------------------------

def test_heat_fresh_reads_cabinet_against_soft_line():
    r = heat_reading(HeatInput(read_ok=True, temp_c=29.5, thermal_state="elevated", observed_at=NOW), CFG)
    assert r.level == 29.5 and r.budget == 31.0 and r.budget_kind == "soft_line" and r.no_reading is None
    assert r.detail["over_line_c"] == -1.5


def test_heat_line_is_configurable():
    r = heat_reading(HeatInput(read_ok=True, temp_c=30.0, thermal_state="normal"), BodyConfig(heat_soft_line_c=29.0))
    assert r.budget == 29.0 and r.detail["over_line_c"] == 1.0


def test_heat_failed_read_is_no_reading_not_zero():
    r = heat_reading(HeatInput(read_ok=False), CFG)
    assert r.level is None and r.no_reading == "read_failed:cabinet"


def test_heat_stale_cabinet_is_no_reading_even_with_a_last_temp():
    r = heat_reading(HeatInput(read_ok=True, temp_c=28.0, thermal_state="unknown"), CFG)
    assert r.level is None and r.no_reading == "stale:cabinet" and r.detail["last_temp_c"] == 28.0


def test_heat_no_points():
    assert heat_reading(HeatInput(read_ok=True, temp_c=None, thermal_state="unknown"), CFG).no_reading == "no_points:cabinet"


# --- energy -------------------------------------------------------------------------------------

def _hours(n, kwh=1.0, last=NOW - timedelta(minutes=30)):
    last = last.replace(minute=0, second=0, microsecond=0)
    return [EnergyHour(last - timedelta(hours=i), kwh) for i in range(n)]


def test_energy_prices_each_hour_at_its_own_rate_and_carries_the_newest_rate_forward():
    hours = _hours(4)                                    # 14:00..17:00
    prices = [EnergyPrice(hours[3].hour_start, 0.10), EnergyPrice(hours[2].hour_start, 0.20)]
    r = energy_reading(EnergyInput(read_ok=True, month_start=MONTH, hours=hours, prices=prices), NOW, CFG)
    # 0.10 + 0.20 + two later hours carried at 0.20
    assert abs(r.level - 0.70) < 1e-9 and r.unit == "usd"
    assert r.detail["priced_kwh"] == 2.0 and r.detail["carried_rate_kwh"] == 2.0
    assert r.budget is None and r.budget_kind is None              # no envelope set
    assert r.source_at == hours[0].hour_start


def test_energy_rate_from_before_month_seeds_carry_forward():
    hours = _hours(2)
    r = energy_reading(EnergyInput(read_ok=True, month_start=MONTH, hours=hours,
                                   prices=[EnergyPrice(MONTH - timedelta(days=3), 0.12)]), NOW, CFG)
    assert abs(r.level - 0.24) < 1e-9 and r.detail["carried_rate_kwh"] == 2.0


def test_energy_envelope_is_the_line_when_set():
    r = energy_reading(EnergyInput(read_ok=True, month_start=MONTH, hours=_hours(1),
                                   prices=[EnergyPrice(MONTH, 0.1)]), NOW, BodyConfig(energy_monthly_envelope_usd=40.0))
    assert r.budget == 40.0 and r.budget_kind == "envelope"


def test_energy_no_reading_cases():
    def reason(inp):
        r = energy_reading(inp, NOW, CFG)
        assert r.level is None
        return r.no_reading

    assert reason(EnergyInput(read_ok=False, failed_source="energy_cost_accrued")) == "read_failed:energy_cost_accrued"
    assert reason(EnergyInput(read_ok=True, month_start=MONTH)) == "no_rows:orion_energy_hourly"
    stale = _hours(2, last=NOW - timedelta(hours=5))
    assert reason(EnergyInput(read_ok=True, month_start=MONTH, hours=stale,
                              prices=[EnergyPrice(MONTH, 0.1)])) == "stale:orion_energy_hourly"
    assert reason(EnergyInput(read_ok=True, month_start=MONTH, hours=_hours(2))) == "no_rate:energy_cost_accrued"


def test_energy_hours_before_month_start_are_ignored():
    hours = _hours(1) + [EnergyHour(MONTH - timedelta(hours=1), 100.0)]
    r = energy_reading(EnergyInput(read_ok=True, month_start=MONTH, hours=hours,
                                   prices=[EnergyPrice(MONTH - timedelta(days=1), 0.1)]), NOW, CFG)
    assert abs(r.level - 0.1) < 1e-9


# --- lanes --------------------------------------------------------------------------------------

def _close(lease, role, at, held_s, event="released"):
    return LeaseEvent(lease_id=lease, role=role, event=event, at=at, held_ms=held_s * 1000.0)


def test_lane_slot_seconds_clip_to_the_window_and_union_expired_then_released():
    events = [
        _close("a", "fast", NOW - timedelta(minutes=10), 60),
        # held 2 h, released 30 min ago: only 30 min of it inside the window
        _close("b", "fast", NOW - timedelta(minutes=30), 7200),
        # one lease, expired then released: not double counted
        _close("c", "metacog", NOW - timedelta(minutes=5), 40, event="expired"),
        _close("c", "metacog", NOW - timedelta(minutes=4), 100),
        # an open grant 20 min ago holds up to now
        LeaseEvent(lease_id="d", role="agent", event="granted", at=NOW - timedelta(minutes=20)),
    ]
    out = {r.constraint: r for r in lane_readings(LaneInput(True, events, {"fast": 4, "metacog": 2, "agent": 1,
                                                                            "diffusion": 1}), NOW, CFG)}
    assert out["lane:fast"].level == 60 + 1800
    assert out["lane:fast"].budget == 4 * 3600.0 and out["lane:fast"].budget_kind == "capacity"
    assert out["lane:metacog"].level == 100.0          # union of [t-40,t] and [t+60-100, t+60]
    assert out["lane:agent"].level == 1200.0
    # A lane with capacity and no lease was read fresh: a real 0, not no_reading.
    assert out["lane:diffusion"].level == 0.0 and out["lane:diffusion"].no_reading is None


def test_lane_capacity_unread_keeps_the_level_without_a_line():
    out = lane_readings(LaneInput(True, [_close("a", "fast", NOW, 10)], None), NOW, CFG)
    assert out[0].level == 10.0 and out[0].budget is None and out[0].detail["capacity"] == "pool_state_unread"


def test_lane_events_read_failed_is_no_reading_per_known_lane():
    out = lane_readings(LaneInput(False, [], {"fast": 4}), NOW, CFG)
    assert [(r.constraint, r.level, r.no_reading) for r in out] == [("lane:fast", None, "read_failed:gpu_pool_events")]
    out = lane_readings(LaneInput(False, [], None), NOW, CFG)
    assert out[0].constraint == "lane" and out[0].no_reading == "read_failed:gpu_pool_events"


def test_lane_close_without_held_ms_is_not_invented():
    ev = LeaseEvent(lease_id="x", role="fast", event="released", at=NOW, held_ms=None)
    out = lane_readings(LaneInput(True, [ev], {"fast": 1}), NOW, CFG)
    assert out[0].level == 0.0


# --- attention:juniper --------------------------------------------------------------------------

def test_juniper_counts_sends_against_the_daily_cap():
    r = juniper_reading(JuniperInput(True, sent_today=3, cap_counted_today=2, newest_decision_at=NOW), NOW, CFG)
    assert r.level == 3.0 and r.budget == 4.0 and r.budget_kind == "daily_cap" and r.detail["cap_counted_today"] == 2


def test_juniper_zero_sent_is_a_real_zero_only_when_the_log_is_fresh():
    assert juniper_reading(JuniperInput(True, 0, 0, NOW - timedelta(minutes=1)), NOW, CFG).level == 0.0
    stale = juniper_reading(JuniperInput(True, 0, 0, NOW - timedelta(hours=2)), NOW, CFG)
    assert stale.level is None and stale.no_reading == "stale:endogenous_outreach_decisions"
    assert juniper_reading(JuniperInput(True, 0, 0, None), NOW, CFG).no_reading == "no_rows:endogenous_outreach_decisions"
    assert juniper_reading(JuniperInput(False), NOW, CFG).no_reading == "read_failed:endogenous_outreach_decisions"


# --- claude -------------------------------------------------------------------------------------

def _obs(kind="session_limit", state="clear", observed=True, n=0, age_s=60):
    return ClaudeObservation(kind=kind, state=state, observed=observed, event_count=n,
                             received_at=NOW - timedelta(seconds=age_s))


def test_claude_no_observation_is_unknown_not_clear():
    r = claude_reading((), NOW, CFG)
    assert r.level is None and r.state == "unknown" and r.no_reading == "no_observation"


def test_claude_unobserved_and_stale():
    assert claude_reading((_obs(observed=False, state="unknown"),), NOW, CFG).no_reading == "unobserved:claude_limit"
    assert claude_reading((_obs(age_s=4 * 3600),), NOW, CFG).no_reading == "stale:claude_limit"


def test_claude_any_kind_limited_is_limited_and_counts_events():
    r = claude_reading((_obs(n=2), _obs(kind="weekly_limit", state="limited", n=1)), NOW, CFG)
    assert r.state == "limited" and r.level == 3.0 and r.unit == "limit_events"
    assert r.detail["weekly_limit_state"] == "limited"


def test_claude_one_kind_unread_beside_clear_is_unknown():
    r = claude_reading((_obs(), _obs(kind="weekly_limit", observed=False, state="unknown")), NOW, CFG)
    assert r.state == "unknown" and r.level == 0.0


# --- whole state --------------------------------------------------------------------------------

def test_build_state_has_every_constraint_and_round_trips():
    inputs = BodyInputs(
        heat=HeatInput(True, 30.0, "elevated", NOW),
        energy=EnergyInput(True, MONTH, _hours(2), [EnergyPrice(MONTH, 0.1)]),
        lanes=LaneInput(True, [], {"fast": 4}),
        juniper=JuniperInput(True, 1, 1, NOW),
        claude=())
    st = build_physiology_state(NOW, inputs, CFG, warnings=["w"])
    assert [c.constraint for c in st.constraints] == ["heat", "energy", "lane:fast", "attention:juniper", "claude"]
    assert st.tick_id == f"physiology:{NOW.isoformat()}" and st.warnings == ["w"]
    again = parse_physiology_state(st.model_dump_json())
    assert isinstance(again, PhysiologyStateV1) and again == st
    assert parse_physiology_state(b"{not json") is None and parse_physiology_state(None) is None


def test_no_reading_always_means_level_none():
    inputs = BodyInputs(heat=HeatInput(False), energy=EnergyInput(False), lanes=LaneInput(False),
                        juniper=JuniperInput(False), claude=())
    st = build_physiology_state(NOW, inputs, CFG)
    assert st.constraints and all(c.no_reading and c.level is None for c in st.constraints)
