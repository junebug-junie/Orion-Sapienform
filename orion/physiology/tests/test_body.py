"""Organ 1 Body: pure reducer tests. Every missing or stale source is no_reading, never 0."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.physiology.body import (
    BacklogInput,
    BacklogSnapshot,
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
    occupied_seconds,
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

def _close(lease, role, at, held_s, event="released", hold=None):
    return LeaseEvent(lease_id=lease, role=role, event=event, at=at, held_ms=held_s * 1000.0, hold_lease_id=hold)


def _lanes(events, capacity, newest=NOW):
    out, warnings = lane_readings(LaneInput(True, events, capacity, newest_event_at=newest), NOW, CFG)
    return {r.constraint: r for r in out}, warnings


def test_lane_slot_seconds_clip_to_the_window_and_merge_one_leases_closes():
    events = [
        _close("a", "fast", NOW - timedelta(minutes=10), 60),
        # held 2 h, released 30 min ago: only 30 min of it inside the window
        _close("b", "fast", NOW - timedelta(minutes=30), 7200),
        # one lease, expired then released: overlapping spans merge, not double counted
        _close("c", "metacog", NOW - timedelta(minutes=5), 40, event="expired"),
        _close("c", "metacog", NOW - timedelta(minutes=4), 100),
        # an open lease (from gpu_pool_leases) granted 20 min ago holds up to now
        LeaseEvent(lease_id="d", role="agent", event="granted", at=NOW - timedelta(minutes=20)),
    ]
    out, warnings = _lanes(events, {"fast": 4, "metacog": 2, "agent": 1, "diffusion": 1})
    assert out["lane:fast"].level == 60 + 1800
    assert out["lane:fast"].budget == 4 * 3600.0 and out["lane:fast"].budget_kind == "capacity"
    assert out["lane:metacog"].level == 100.0
    assert out["lane:agent"].level == 1200.0
    # A lane with capacity and no lease, read from a fresh feed: a real 0, not no_reading.
    assert out["lane:diffusion"].level == 0.0 and out["lane:diffusion"].no_reading is None
    assert warnings == ["lane_open_leases:1"]


def test_lane_retried_lease_does_not_count_its_wait():
    # held 60 s, retried at t-10m; re-granted and held 30 s more, released at t-2m.
    events = [_close("r", "fast", NOW - timedelta(minutes=10), 60, event="retried"),
              _close("r", "fast", NOW - timedelta(minutes=2), 30)]
    out, _ = _lanes(events, {"fast": 1})
    assert out["lane:fast"].level == 90.0


def test_lane_child_calls_inside_a_hold_are_not_counted_twice():
    events = [_close("hold", "agent", NOW - timedelta(minutes=1), 600),
              _close("child", "agent", NOW - timedelta(minutes=2), 120, hold="hold")]
    out, warnings = _lanes(events, {"agent": 1})
    assert out["lane:agent"].level == 600.0
    assert "lane_child_leases_in_hold_slots:1" in warnings


def test_lane_close_without_held_ms_closes_the_lease_with_no_time():
    events = [LeaseEvent(lease_id="x", role="fast", event="cancelled", at=NOW, held_ms=None)]
    out, _ = _lanes(events, {"fast": 1})
    assert out["lane:fast"].level == 0.0


def test_lane_zero_slot_role_has_a_zero_line_not_no_line():
    out, _ = _lanes([], {"agent-gpu2": 0})
    assert out["lane:agent-gpu2"].budget == 0.0 and out["lane:agent-gpu2"].budget_kind == "capacity"


def test_lane_capacity_unread_keeps_the_level_without_a_line():
    out, _ = _lanes([_close("a", "fast", NOW, 10)], None)
    assert out["lane:fast"].level == 10.0 and out["lane:fast"].budget is None
    assert out["lane:fast"].detail["capacity"] == "pool_state_unread"


def test_lane_no_reading_cases():
    def reasons(inp):
        out, _ = lane_readings(inp, NOW, CFG)
        assert all(r.level is None for r in out)
        return [(r.constraint, r.no_reading) for r in out]

    assert reasons(LaneInput(False, [], {"fast": 4})) == [("lane:fast", "read_failed:gpu_pool_events")]
    assert reasons(LaneInput(False, [], None)) == [("lane", "read_failed:gpu_pool_events")]
    # The events feed went quiet: every lane is stale, never 0.
    assert reasons(LaneInput(True, [], {"fast": 4}, newest_event_at=NOW - timedelta(hours=1))) == [
        ("lane:fast", "stale:gpu_pool_events")]
    assert reasons(LaneInput(True, [], {"fast": 4}, newest_event_at=None)) == [("lane:fast", "no_rows:gpu_pool_events")]
    # Fresh feed, nothing held, pool state unread: the lanes do not vanish silently.
    assert reasons(LaneInput(True, [], None, newest_event_at=NOW)) == [("lane", "read_failed:gpu_pool_state")]


def _ts(hms: str) -> datetime:
    h, m, sec = hms.split(":")
    return datetime(2026, 10, 23, int(h), int(m), tzinfo=timezone.utc) + timedelta(seconds=float(sec))


def test_lane_lease_replaced_on_another_seat_is_charged_per_seat_live_2026_10_23():
    """The live over-count (lane:agent read 5,985.9 slot-s in a 3,600 s window on 1 slot,
    tick 2026-10-23 10:14:12). Real span shapes from gpu_pool_events/gpu_pool_leases:

    * f4126f56 hold on agent, released 09:40:15.760 after 2,085.868 s;
    * 720191b9 hold on agent, released 10:14:07.759 after 2,031.875 s;
    * 3e2aaebf hold, ABORTED twice on agent-gpu2 (09:41:27.480 after 1,718.778 s, 10:05:16.052
      after 755.122 s), then re-granted on agent at 10:14:07.766 and still open.

    The old reducer kept one role per lease (the last row it saw: the open row, agent), so both
    agent-gpu2 spans landed on lane:agent: 1,635.3 + 755.1 = 2,390.4 s extra."""
    now = _ts("10:14:12.168727")
    ev = [
        LeaseEvent("f4126f56", "agent", "released", _ts("09:40:15.760344"), 2085.867939e3),
        LeaseEvent("720191b9", "agent", "released", _ts("10:14:07.758557"), 2031.87485e3),
        LeaseEvent("3e2aaebf", "agent-gpu2", "aborted", _ts("09:41:27.479759"), 1718.777638e3),
        LeaseEvent("3e2aaebf", "agent-gpu2", "aborted", _ts("10:05:16.052205"), 755.121794e3),
        LeaseEvent("3e2aaebf", "agent", "granted", _ts("10:14:07.766452")),
    ]
    out, warnings = lane_readings(LaneInput(True, ev, {"agent": 1, "agent-gpu2": 2}, newest_event_at=now), now, CFG)
    by = {r.constraint: r for r in out}
    agent, gpu2 = by["lane:agent"], by["lane:agent-gpu2"]
    assert agent.level <= agent.budget == 3600.0
    # agent: 09:14:12.17..09:40:15.76 + 09:40:15.88..10:14:07.76 + the re-grant's 4.4 s open.
    assert abs(agent.level - (1563.59 + 2031.87 + 4.40)) < 0.5
    # agent-gpu2 gets its own 2,390 s, which the old reducer gave to agent.
    assert abs(gpu2.level - 2390.4) < 0.5
    assert "lane_leases_held_on_two_seats:1" in warnings


def test_lane_gap_lending_is_reported_apart_not_counted_as_held():
    """Live 10-23 06:09: a cortex-exec request (system priority, no hold_lease_id) ran 123 s in the
    gap of an idle background hold on the 1-slot agent lane (scheduler H3). Both leases are real;
    the slot was occupied once."""
    now = _ts("06:30:00")
    ev = [LeaseEvent("b8964820", "agent", "granted", _ts("03:10:21.633747")),       # still held
          LeaseEvent("54397698", "agent", "released", _ts("06:11:30.35888"), 123.132188e3),
          LeaseEvent("df6083a6", "agent", "released", _ts("06:24:42.562854"), 270.567544e3)]
    out, _ = lane_readings(LaneInput(True, ev, {"agent": 1}, newest_event_at=now), now, CFG)
    agent = out[0]
    assert agent.level == 3600.0 == agent.budget
    assert abs(agent.detail["lease_s"] - (3600 + 123.132 + 270.568)) < 0.01
    assert abs(agent.detail["lent_overlap_s"] - (123.132 + 270.568)) < 0.01


def test_lane_projection_still_granted_after_a_close_is_not_held_to_now():
    """Review finding: the pool's projection row lags its own close event. The close (newer
    fact) wins: the lease held 600 s, not the whole window."""
    ev = [LeaseEvent("L", "fast", "granted", NOW - timedelta(minutes=60)),
          _close("L", "fast", NOW - timedelta(minutes=50), 600)]
    out, warnings = _lanes(ev, {"fast": 1})
    assert out["lane:fast"].level == 600.0 and not any(w.startswith("lane_open_leases") for w in warnings)


def test_lane_two_slots_count_two_concurrent_holds_and_cap_the_third():
    now = NOW
    ev = [LeaseEvent(x, "agent-gpu2", "granted", NOW - timedelta(minutes=30)) for x in ("a", "b", "c")]
    out, _ = lane_readings(LaneInput(True, ev, {"agent-gpu2": 2}, newest_event_at=now), now, CFG)
    assert out[0].level == 3600.0 and out[0].detail["lent_overlap_s"] == 1800.0


def test_lane_zero_slot_seat_with_only_closed_leases_served_them_while_loaded():
    """lane:diffusion read 546 s against capacity 0 live: a swap seat (diffusion / agent-gpu2 share
    gpu2) that ran a painting while loaded and has been swapped out since. Uncapped, never hidden,
    and NOT flagged as held without a slot (Patch C1: live 10-27 04:31 painting, then unloaded)."""
    ev = [_close("d", "diffusion", NOW - timedelta(minutes=1), 546.479)]
    out, _ = _lanes(ev, {"diffusion": 0})
    d = out["lane:diffusion"]
    assert abs(d.level - 546.479) < 1e-6 and d.detail["served_while_loaded"] is True
    assert "held_without_live_slot" not in d.detail and "lent_overlap_s" not in d.detail


def test_lane_zero_slot_seat_with_an_open_lease_is_really_held_without_a_live_slot():
    ev = [LeaseEvent("d", "diffusion", "granted", NOW - timedelta(minutes=5))]
    out, _ = lane_readings(LaneInput(True, ev, {"diffusion": 0}, newest_event_at=NOW), NOW, CFG)
    d = {r.constraint: r for r in out}["lane:diffusion"]
    assert d.detail["held_without_live_slot"] is True and "served_while_loaded" not in d.detail


def test_occupied_seconds_end_and_start_at_the_same_instant_is_one_slot():
    t = NOW
    spans = [(t - timedelta(seconds=100), t - timedelta(seconds=50)), (t - timedelta(seconds=50), t)]
    assert occupied_seconds(spans, t - timedelta(hours=1), t, 1) == (100.0, 100.0, 2)


# --- deferrable backlog (hunger) -----------------------------------------------------------------

# config/gpu_pool.yaml ``classes`` (subset), as the state RPC returns it.
CLASSES = {"agent": ["agent", "agent-gpu2", "agent-deep", "chat"], "memory_distill": ["agent"],
           "diffusion": ["diffusion"], "world": ["world"]}


def _backlog(depth, *, at=NOW - timedelta(seconds=5), last_empty=None, host="circe"):
    return BacklogInput(True, [BacklogSnapshot(host, at, depth)],
                        last_empty_at={(host, wc): last_empty for wc in (depth or {})})


def _hungry(bl, capacity, classes=CLASSES):
    out, warnings = lane_readings(LaneInput(True, [], capacity, newest_event_at=NOW, class_roles=classes),
                                  NOW, CFG, bl)
    return {r.constraint: r for r in out}, warnings


def test_backlog_sits_beside_every_seat_its_class_may_use():
    """Live 10-23 23:58: deferrable_depth {"agent": 4} (curiosity readings, background holds). The
    agent class may run on agent, agent-gpu2, agent-deep and chat, so it sits beside each of them;
    a seat it cannot use reads a real 0."""
    bl = _backlog({"agent": 4}, last_empty=NOW - timedelta(minutes=26))
    by, warnings = _hungry(bl, {"agent": 1, "agent-gpu2": 2, "fast": 4})
    for lane in ("lane:agent", "lane:agent-gpu2"):
        assert by[lane].detail["deferrable_queued"] == 4
        assert by[lane].detail["deferrable_waiting_sec"] == 26 * 60.0
    assert by["lane:fast"].detail["deferrable_queued"] == 0
    # Hunger is not strain: the level (held slot-seconds) does not move.
    assert by["lane:agent"].level == 0.0 and not warnings


def test_backlog_classes_add_on_a_shared_seat():
    bl = _backlog({"agent": 2, "memory_distill": 3})
    by, _ = _hungry(bl, {"agent": 1, "agent-gpu2": 2})
    assert by["lane:agent"].detail["deferrable_queued"] == 5
    assert by["lane:agent-gpu2"].detail["deferrable_queued"] == 2


def test_backlog_non_empty_for_the_whole_lookback_reads_capped():
    by, _ = _hungry(_backlog({"agent": 2}, last_empty=None), {"agent": 1})
    assert by["lane:agent"].detail["deferrable_waiting_sec"] == 6 * 3600.0
    assert by["lane:agent"].detail["deferrable_waiting_capped"] is True


def test_backlog_unknown_is_never_zero():
    def reason(bl, classes=CLASSES):
        by, _ = _hungry(bl, {"agent": 1}, classes)
        assert "deferrable_queued" not in by["lane:agent"].detail
        return by["lane:agent"].detail["deferrable_no_reading"]

    assert reason(None) == "not_read"
    assert reason(BacklogInput(False)) == "read_failed:gpu_pool_state_history"
    assert reason(BacklogInput(True)) == "no_rows:gpu_pool_state_history"
    assert reason(_backlog({"agent": 1}, at=NOW - timedelta(minutes=5))) == "stale:gpu_pool_state_history"
    # A snapshot from before #2873 (deferrable_depth NULL) is unknown, not "nothing waiting".
    assert reason(_backlog(None)) == "unrecorded:deferrable_depth"
    # The pool's class map was not read: the seat cannot be told, so it is not a 0.
    assert reason(_backlog({"agent": 1}), classes=None) == "unread:gpu_pool_classes"


def test_backlog_snapshot_slightly_ahead_of_our_clock_is_fresh():
    by, _ = _hungry(_backlog({"agent": 1}, at=NOW + timedelta(seconds=3)), {"agent": 1})
    assert by["lane:agent"].detail["deferrable_queued"] == 1


def test_backlog_stale_host_is_named_not_silently_dropped():
    bl = BacklogInput(True, [BacklogSnapshot("circe", NOW - timedelta(seconds=5), {"agent": 1}),
                             BacklogSnapshot("hecate", NOW - timedelta(minutes=4), {"agent": 7})],
                      last_empty_at={("circe", "agent"): NOW - timedelta(minutes=1)})
    by, warnings = _hungry(bl, {"agent": 1})
    assert by["lane:agent"].detail["deferrable_queued"] == 1
    assert by["lane:agent"].detail["deferrable_hosts_stale"] == "hecate"
    assert "lane_backlog_hosts_stale:hecate" in warnings


def test_backlog_class_with_no_seat_on_the_pool_is_a_warning_not_lost():
    _, warnings = _hungry(_backlog({"world": 3}), {"agent": 1})
    assert "lane_deferrable_unplaced:world:3" in warnings


def test_backlog_detail_survives_the_schema_and_the_stored_row():
    bl = _backlog({"agent": 4}, last_empty=NOW - timedelta(minutes=1))
    inputs = BodyInputs(heat=HeatInput(False), energy=EnergyInput(False),
                        lanes=LaneInput(True, [], {"agent": 1}, newest_event_at=NOW, class_roles=CLASSES),
                        juniper=JuniperInput(False), backlog=bl)
    state = build_physiology_state(NOW, inputs)
    again = parse_physiology_state(state.model_dump_json())
    lane = [c for c in again.constraints if c.constraint == "lane:agent"][0]
    assert lane.detail["deferrable_queued"] == 4


# --- attention:juniper --------------------------------------------------------------------------

def test_juniper_counts_cap_counted_sends_against_the_daily_cap():
    r = juniper_reading(JuniperInput(True, sent_today=3, cap_counted_today=2, newest_decision_at=NOW), NOW, CFG)
    # Level and line count the same thing; every send that reached Juniper is in detail.
    assert r.level == 2.0 and r.budget == 4.0 and r.budget_kind == "daily_cap"
    assert r.detail["sent_today_all_sources"] == 3


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


def test_claude_both_kinds_clear_is_clear():
    r = claude_reading((_obs(), _obs(kind="weekly_limit")), NOW, CFG)
    assert r.state == "clear" and r.level == 0.0


def test_claude_one_kind_unread_beside_clear_is_unknown():
    r = claude_reading((_obs(), _obs(kind="weekly_limit", observed=False, state="unknown")), NOW, CFG)
    assert r.state == "unknown" and r.level == 0.0


def test_claude_weekly_not_yet_received_after_restart_is_unknown_not_clear():
    r = claude_reading((_obs(),), NOW, CFG)
    assert r.state == "unknown" and r.detail["weekly_limit_state"] == "unread"
    stale_weekly = claude_reading((_obs(), _obs(kind="weekly_limit", age_s=4 * 3600)), NOW, CFG)
    assert stale_weekly.state == "unknown" and stale_weekly.detail["weekly_limit_state"] == "unread"


# --- whole state --------------------------------------------------------------------------------

def test_build_state_has_every_constraint_and_round_trips():
    inputs = BodyInputs(
        heat=HeatInput(True, 30.0, "elevated", NOW),
        energy=EnergyInput(True, MONTH, _hours(2), [EnergyPrice(MONTH, 0.1)]),
        lanes=LaneInput(True, [], {"fast": 4}, newest_event_at=NOW),
        juniper=JuniperInput(True, 1, 1, NOW),
        claude=())
    st = build_physiology_state(NOW, inputs, CFG, warnings=["w"])
    assert [c.constraint for c in st.constraints] == ["heat", "energy", "lane:fast", "attention:juniper", "claude"]
    assert st.tick_id == f"physiology:{NOW.isoformat()}" and st.warnings == ["w"]
    again = parse_physiology_state(st.model_dump_json())
    assert isinstance(again, PhysiologyStateV1) and again == st
    assert parse_physiology_state(b"{not json") is None and parse_physiology_state(None) is None


def test_reading_contract_level_xor_no_reading():
    import pytest
    from orion.schemas.physiology import PhysiologyConstraintReadingV1 as R

    with pytest.raises(ValueError):
        R(constraint="heat", unit="c", level=None)
    with pytest.raises(ValueError):
        R(constraint="heat", unit="c", level=0.0, no_reading="stale:cabinet")


def test_unpriced_energy_is_flagged_as_partial():
    hours = _hours(2) + [EnergyHour(MONTH, 5.0)]     # month's first hour precedes every rate
    inputs = BodyInputs(heat=HeatInput(False), lanes=LaneInput(False), juniper=JuniperInput(False),
                        energy=EnergyInput(True, MONTH, hours, [EnergyPrice(MONTH + timedelta(hours=1), 0.1)]))
    st = build_physiology_state(NOW, inputs, CFG)
    assert "energy_partial:unpriced_kwh" in st.warnings


def test_no_reading_always_means_level_none():
    inputs = BodyInputs(heat=HeatInput(False), energy=EnergyInput(False), lanes=LaneInput(False),
                        juniper=JuniperInput(False), claude=())
    st = build_physiology_state(NOW, inputs, CFG)
    assert st.constraints and all(c.no_reading and c.level is None for c in st.constraints)
