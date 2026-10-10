"""Arousal reducer (Temporal Self rev 4, R3): absence, hysteresis, restart lifecycle."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.regulation.arousal import classify_arousal, gpu_queue_evidence, level_seconds
from orion.regulation.juniper_turns import JUNIPER_IDLE_MINUTES_SQL, JUNIPER_TURN_PREDICATE, is_juniper_turn
from orion.schemas.regulation import (
    ArousalInputsV1,
    RegulationStateV1,
    parse_regulation_state,
)
from orion.schemas.registry import resolve

T0 = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)


def inputs(at=T0, *, minutes=120.0, turn_ok=True, reflex=None, thermal="normal", cab_ok=True,
           gpu_age=2.0, depth=0, sustained=0.0) -> ArousalInputsV1:
    return ArousalInputsV1(
        observed_at=at, juniper_turn_read_ok=turn_ok, minutes_since_juniper_turn=minutes,
        cabinet_read_ok=cab_ok, cabinet_reflex=reflex, cabinet_thermal_state=thermal, cabinet_temp_c=31.0,
        gpu_state_age_sec=gpu_age, gpu_queue_depth=depth, gpu_queue_sustained_sec=sustained)


def step(prev, **kw):
    return classify_arousal(prev, inputs(**kw))


# --- absence -> unknown ---------------------------------------------------------------------


@pytest.mark.parametrize("kw,stale", [
    (dict(turn_ok=False, minutes=None), "stale:juniper_turns"),
    (dict(cab_ok=False, reflex=None, thermal=None), "stale:cabinet"),
    (dict(reflex="cabinet_unknown", thermal="unknown"), "stale:cabinet"),
    (dict(gpu_age=None, depth=None, sustained=None), "stale:gpu_state"),
    (dict(gpu_age=31.0), "stale:gpu_state"),
])
def test_any_stale_input_reads_unknown_never_idle(kw, stale):
    r = step(None, **kw)
    assert r.arousal_level == "unknown"
    assert stale in r.reasons


def test_everything_missing_is_unknown():
    r = classify_arousal(None, ArousalInputsV1(observed_at=T0))
    assert r.arousal_level == "unknown"


def test_no_juniper_turn_ever_is_a_fresh_idle():
    r = step(None, minutes=None, turn_ok=True)
    assert r.arousal_level == "idle"


def test_disabled_is_unknown():
    r = classify_arousal(None, inputs(), enabled=False)
    assert r.arousal_level == "unknown" and r.reasons == ["disabled"]


# --- engaged / idle -------------------------------------------------------------------------


def test_turn_inside_45_minutes_is_engaged_and_outside_is_idle():
    assert step(None, minutes=44.9).arousal_level == "engaged"
    assert step(None, minutes=45.0).arousal_level == "idle"


def test_a_turn_moves_idle_to_engaged_immediately():
    idle = step(None, minutes=300)
    r = step(idle, at=T0 + timedelta(seconds=5), minutes=0.0)
    assert r.arousal_level == "engaged" and r.since == T0 + timedelta(seconds=5)


def test_since_holds_while_level_holds():
    a = step(None, minutes=100)
    b = step(a, at=T0 + timedelta(seconds=120), minutes=102)
    assert b.arousal_level == "idle" and b.since == T0


# --- strain and hysteresis ------------------------------------------------------------------


def test_cabinet_hot_is_immediate_strain_even_with_other_inputs_stale():
    r = step(None, reflex="cabinet_hot", thermal="hot", gpu_age=None, depth=None, turn_ok=False)
    assert r.arousal_level == "strained" and "cabinet_hot" in r.reasons and r.strain_latched


def test_gpu_queue_needs_five_sustained_minutes():
    assert step(None, depth=5, sustained=299.0).arousal_level == "idle"
    r = step(None, depth=5, sustained=300.0)
    assert r.arousal_level == "strained" and "gpu_queue_sustained" in r.reasons


def test_strain_clears_only_after_ten_clear_minutes():
    r = step(None, depth=5, sustained=400.0)
    t = T0
    for _ in range(5):                      # clear from 120 s: 0..480 s of clear time, still strained
        t += timedelta(seconds=120)
        r = step(r, at=t, depth=0)
        assert r.arousal_level == "strained", r.reasons
    t += timedelta(seconds=120)             # 720 s: 600 s of clear time
    r = step(r, at=t, depth=0)
    assert r.arousal_level == "idle" and not r.strain_latched and "strain_cleared" in r.reasons


def test_queue_back_above_floor_restarts_the_clear_clock():
    r = step(None, depth=5, sustained=400.0)
    r = step(r, at=T0 + timedelta(seconds=120), depth=0)
    assert r.strain_clear_since == T0 + timedelta(seconds=120)
    r = step(r, at=T0 + timedelta(seconds=240), depth=3, sustained=10.0)
    assert r.arousal_level == "strained" and r.strain_clear_since is None


def test_stale_time_never_earns_clear_time():
    r = step(None, depth=5, sustained=400.0)
    r = step(r, at=T0 + timedelta(seconds=120), depth=0)
    r = step(r, at=T0 + timedelta(seconds=240), gpu_age=None, depth=None, sustained=None)
    assert r.arousal_level == "unknown" and r.strain_latched and r.strain_clear_since is None
    r = step(r, at=T0 + timedelta(seconds=360), depth=0)
    assert r.arousal_level == "strained" and r.strain_clear_since == T0 + timedelta(seconds=360)


def test_cabinet_hot_resets_the_clear_clock():
    r = step(None, depth=5, sustained=400.0)
    r = step(r, at=T0 + timedelta(seconds=120), depth=0)
    r = step(r, at=T0 + timedelta(seconds=240), depth=0, reflex="cabinet_hot", thermal="hot")
    assert r.strain_clear_since is None and r.since == T0


# --- lifecycle across a restart -------------------------------------------------------------


def test_short_restart_keeps_since_and_latch():
    r = step(None, depth=5, sustained=400.0)
    r = step(r, at=T0 + timedelta(seconds=120), depth=0)
    # Round-trip through JSON, as the checkpoint and Redis do.
    restored = RegulationStateV1(generated_at=r.observed_at, arousal=r).model_dump_json()
    prev = parse_regulation_state(restored).arousal
    r2 = step(prev, at=T0 + timedelta(seconds=300), depth=0)
    assert r2.arousal_level == "strained" and r2.since == T0
    assert r2.strain_clear_since == T0 + timedelta(seconds=120)


def test_long_outage_drops_the_previous_reading():
    r = step(None, depth=5, sustained=400.0)
    r2 = step(r, at=T0 + timedelta(hours=4), depth=0)
    assert r2.arousal_level == "idle" and r2.since == T0 + timedelta(hours=4)
    assert "prev_dropped:gap" in r2.reasons


def test_future_stamped_previous_is_dropped():
    r = step(None, at=T0 + timedelta(minutes=10), minutes=1)
    r2 = step(r, at=T0, minutes=100)
    assert r2.since == T0 and r2.arousal_level == "idle"


# --- GPU evidence ---------------------------------------------------------------------------


def snaps(depths, start=T0, every=5):
    return [(start + timedelta(seconds=i * every), d) for i, d in enumerate(depths)]


def test_gpu_evidence_sustained_run_and_age():
    s = snaps([{}] + [{"agent": 2}] * 61)                    # 61 snapshots at depth 2 = 300 s
    age, depth, sus = gpu_queue_evidence(s, T0 + timedelta(seconds=305 + 3))
    assert (age, depth, sus) == (3.0, 2, 300.0)


def test_gpu_evidence_single_stuck_lease_is_below_the_floor():
    s = snaps([{"diffusion": 1}] * 200)
    assert gpu_queue_evidence(s, T0 + timedelta(seconds=1000))[2] == 0.0


def test_gpu_evidence_gap_breaks_the_run():
    s = snaps([{"agent": 3}] * 10) + snaps([{"agent": 3}] * 10, start=T0 + timedelta(seconds=100))
    _, _, sus = gpu_queue_evidence(s, T0 + timedelta(seconds=150))
    assert sus == 45.0


def test_gpu_evidence_none_and_malformed():
    assert gpu_queue_evidence([], T0) == (None, None, None)
    assert gpu_queue_evidence([(T0, {"agent": -1}), (T0, "x")], T0) == (None, None, None)


def test_level_seconds():
    a = step(None, minutes=1)
    b = step(a, at=T0 + timedelta(minutes=10), minutes=100)
    assert level_seconds([a, b], T0 + timedelta(minutes=30)) == {
        "engaged": 600.0, "idle": 1200.0, "strained": 0.0, "unknown": 0.0}


# --- Juniper turn rule ----------------------------------------------------------------------


@pytest.mark.parametrize("payload,expected", [
    ({"prompt": "hi", "source": "hub_orion"}, True),
    ({"prompt": "Run your dream cycle.", "source": None}, True),
    ({"prompt": "   ", "source": "hub_orion"}, False),
    ({"prompt": None}, False),
    ({"prompt": "", "client_meta": {"unsolicited": True}}, False),
    ({"prompt": "x", "client_meta": {"unsolicited": True}}, False),
    ({"prompt": "x", "client_meta": {"unsolicited": "true"}}, False),
    (None, False),
])
def test_is_juniper_turn(payload, expected):
    assert is_juniper_turn(payload) is expected


def test_idle_sql_filters_to_juniper_turns():
    assert JUNIPER_TURN_PREDICATE in JUNIPER_IDLE_MINUTES_SQL
    assert "unsolicited" in JUNIPER_TURN_PREDICATE and "btrim(prompt)" in JUNIPER_TURN_PREDICATE


def test_registered_in_both_registries():
    from orion.schemas.registry import SCHEMA_REGISTRY

    assert resolve("RegulationStateV1") is RegulationStateV1
    assert SCHEMA_REGISTRY["RegulationStateV1"].model is RegulationStateV1


def test_parse_regulation_state_tolerates_garbage():
    assert parse_regulation_state(None) is None
    assert parse_regulation_state(b"{not json") is None
    assert parse_regulation_state({"schema_version": "regulation.state.v1"}) is None
