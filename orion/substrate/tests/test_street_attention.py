"""Street events -> attention: the gate, the held value, the calm state, Rung 6.

Spec: docs/superpowers/specs/2026-10-20-street-continuity-handoff-design.md
section 6. Pure rules in orion/substrate/street_attention.py.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

import pytest

from orion.attention.world_first import node_candidate, rank_candidates
from orion.schemas.attention_candidate import (
    NO_ATTENTION_CONSUMER_NODE_IDS,
    NO_CURIOSITY_CONSUMER_NODE_IDS,
    STREET_NODE_ID,
    is_world_source_id,
)
from orion.schemas.vision import StreetEventV1
from orion.substrate import mutation_control_surface
from orion.substrate.mutation_control_surface import RuntimeControlSurfaceStore, get_prediction_expectation
from orion.substrate.prediction_error_magnitude import compute_prediction_error_magnitude
from orion.substrate.prediction_expectation import (
    WARMUP_READINGS,
    Expectation,
    assess,
    candidate_grid,
    domain_specs,
    paired_gain,
)
from orion.substrate.street_attention import (
    GATE_P_VALUE,
    P_VALUE_FLOOR,
    STREET_HANDOFF_DOMAIN,
    StreetAttentionState,
    street_gate,
    street_handoff_raw_reading,
)

NOW = datetime(2026, 10, 20, 18, 0, 0, tzinfo=timezone.utc)
T = NOW.timestamp()
UP = {"walkway": T - 5.0, "doorbell": T - 30.0}


def _event(kind: str = "handoff_missed", *, p: float | None = 0.01, streams=("doorbell", "walkway"), **kw) -> StreetEventV1:
    return StreetEventV1(
        event_id=kw.pop("event_id", f"ev-{kind}-{p}"),
        kind=kind,
        stream_ids=list(streams),
        occurred_at=NOW,
        subject_kind=kw.pop("subject_kind", "person"),
        salience=kw.pop("salience", 1.0 if kind == "doorbell_ring" else 0.4),
        p_value=p,
        correlation_id=kw.pop("correlation_id", "corr-1"),
        **kw,
    )


def _gate(event, *, passes=(), cap=4, frames=UP, max_age=600.0):
    return street_gate(
        event,
        now_epoch=T,
        recent_pass_epochs=list(passes),
        max_per_hour=cap,
        last_frame_at=frames,
        camera_up_max_age_sec=max_age,
    )


# --- gate -----------------------------------------------------------------


def test_a_ring_always_passes_even_over_the_cap_and_with_cameras_dark() -> None:
    ring = _event("doorbell_ring", p=None, streams=("doorbell",))
    v = _gate(ring, passes=[T - 10] * 9, cap=4, frames={})
    assert v.passed and "always passes" in v.reason


def test_a_high_p_event_is_refused() -> None:
    v = _gate(_event(p=0.31))
    assert not v.passed and v.reason.startswith("p_value 0.31")


def test_p_exactly_at_the_threshold_is_refused() -> None:
    assert not _gate(_event(p=GATE_P_VALUE)).passed
    assert _gate(_event(p=GATE_P_VALUE - 1e-9)).passed


def test_an_ungraded_non_ring_event_is_refused() -> None:
    v = _gate(_event("known_person_arrived", p=None, person_guess="Juniper"))
    assert not v.passed and v.reason.startswith("no_p_value")


def test_a_down_camera_blocks() -> None:
    v = _gate(_event(), frames={"walkway": T - 5.0, "doorbell": T - 601.0})
    assert not v.passed and v.reason.startswith("camera_down:doorbell")


def test_a_never_seen_camera_blocks() -> None:
    v = _gate(_event(), frames={"walkway": T - 5.0})
    assert not v.passed and v.reason == "camera_unseen:doorbell"


def test_the_hourly_cap_holds_and_releases_after_an_hour() -> None:
    full = [T - 3500.0, T - 1000.0, T - 100.0, T - 1.0]
    v = _gate(_event(), passes=full)
    assert not v.passed and v.reason.startswith("hourly_cap")
    # The oldest pass ages out of the trailing hour: one slot frees up.
    assert _gate(_event(), passes=[T - 3601.0, *full[1:]]).passed


def test_cap_zero_refuses_everything_but_rings() -> None:
    assert not _gate(_event(), cap=0).passed
    assert _gate(_event("doorbell_ring", p=None, streams=("doorbell",)), cap=0).passed


def test_a_known_person_is_named_in_the_reason() -> None:
    v = _gate(_event(person_guess="Juniper"))
    assert v.passed and "known person Juniper" in v.reason


# --- state: held value and calm -------------------------------------------


def _state() -> StreetAttentionState:
    st = StreetAttentionState()
    for stream, ts in UP.items():
        st.frame_seen(stream, ts)
    return st


def test_a_ring_activates_and_the_node_returns_to_a_real_zero_after_the_hold() -> None:
    st = _state()
    v = st.judge(_event("doorbell_ring", p=None, streams=("doorbell",)), now_epoch=T, max_per_hour=4, camera_up_max_age_sec=600)
    assert v.passed
    assert st.held_reading(T + 1, 90.0)[0] == 1.0
    assert st.held_reading(T + 89, 90.0)[0] == 1.0
    value, held = st.held_reading(T + 90, 90.0)
    assert value == 0.0 and held == []  # calm is a measured 0, not a decayed residue


def test_a_refused_event_never_moves_the_node() -> None:
    st = _state()
    v = st.judge(_event(p=0.6), now_epoch=T, max_per_hour=4, camera_up_max_age_sec=600)
    assert not v.passed
    assert st.held_reading(T + 1, 90.0) == (0.0, [])
    assert st.blocked_by_reason == {"p_value": 1}


def test_the_cap_holds_through_the_state() -> None:
    st = _state()
    verdicts = [
        st.judge(_event(event_id=f"e{i}"), now_epoch=T + i, max_per_hour=4, camera_up_max_age_sec=600).passed
        for i in range(6)
    ]
    assert verdicts == [True, True, True, True, False, False]
    assert st.status(T + 6, 90.0)["passes_last_hour"] == 4
    assert st.blocked_by_reason == {"hourly_cap": 2}


def test_rings_use_up_the_budget() -> None:
    st = _state()
    for i in range(4):
        st.judge(_event("doorbell_ring", p=None, streams=("doorbell",), event_id=f"r{i}"), now_epoch=T + i, max_per_hour=4, camera_up_max_age_sec=600)
    assert not st.judge(_event(), now_epoch=T + 5, max_per_hour=4, camera_up_max_age_sec=600).passed


def test_street_visible_needs_a_recent_frame() -> None:
    st = _state()
    assert st.street_visible(["walkway"], T, 600.0)
    assert not st.street_visible(["walkway"], T + 700, 600.0)
    assert not StreetAttentionState().street_visible(["walkway"], T, 600.0)


# --- attention: the node competes as the world, and rests ----------------


def _street_candidate(current: float, *, zeros: int = 2000, ones: int = 10):
    hist = [(NOW - timedelta(seconds=30 * (i + 1)), 0.0) for i in range(zeros)]
    hist += [(NOW - timedelta(hours=30, seconds=30 * i), 1.0) for i in range(ones)]
    mag = compute_prediction_error_magnitude(value=current, observed_at=NOW, history=hist, now=NOW)
    return node_candidate(
        node_id=STREET_NODE_ID,
        label="Street prediction error",
        magnitude=mag,
        observed_at=NOW,
        now=NOW,
        history_values=[v for _, v in hist],
    )


def test_street_is_world_and_admitted_to_attention_but_not_curiosity() -> None:
    assert is_world_source_id(STREET_NODE_ID)
    assert STREET_NODE_ID not in NO_ATTENTION_CONSUMER_NODE_IDS
    assert STREET_NODE_ID in NO_CURIOSITY_CONSUMER_NODE_IDS


def test_a_held_ring_wins_world_first_and_a_zero_rests() -> None:
    ranking = rank_candidates([_street_candidate(1.0)])
    assert ranking.winner is not None and ranking.winner.candidate.source_id == STREET_NODE_ID
    assert ranking.winner.candidate.source_kind == "external"
    calm = rank_candidates([_street_candidate(0.0)])
    assert calm.no_winner
    assert "not busy" in calm.ineligible[0].reason  # judged quiet, not absent


def test_a_stale_street_reading_is_absent_not_calm() -> None:
    hist = [(NOW - timedelta(seconds=30 * (i + 1)), 0.0) for i in range(2000)]
    old = NOW - timedelta(seconds=400)
    mag = compute_prediction_error_magnitude(value=0.0, observed_at=old, history=hist, now=NOW)
    cand = node_candidate(node_id=STREET_NODE_ID, label="street", magnitude=mag, observed_at=old, now=NOW)
    assert cand.absent and "old" in (cand.absent_reason or "")


def test_the_glossary_marks_the_street_a_per_tick_trigger_not_event_written() -> None:
    from orion.attention.world_first import (
        node_prediction_error_event_written,
        node_prediction_error_semantics,
    )

    assert node_prediction_error_semantics(STREET_NODE_ID)[0] == "trigger"
    # Event-written would fade the reading by its age; this node is rewritten
    # every tick, so its age is the tick's, not the event's.
    assert node_prediction_error_event_written(STREET_NODE_ID) is False


# --- Rung 6 street_handoff -------------------------------------------------


def test_raw_reading_is_minus_log_p_for_graded_handoffs_only() -> None:
    assert street_handoff_raw_reading(_event("handoff_met", p=0.5)) == pytest.approx(math.log(2.0))
    assert street_handoff_raw_reading(_event("handoff_late", p=1.0)) == 0.0
    assert street_handoff_raw_reading(_event("handoff_missed", p=0.0)) == pytest.approx(-math.log(P_VALUE_FLOOR))
    assert street_handoff_raw_reading(_event("doorbell_ring", p=None)) is None
    assert street_handoff_raw_reading(_event("overlap_pair", p=0.01)) is None
    assert street_handoff_raw_reading(_event("handoff_met", p=None)) is None


def test_calibrated_p_values_average_one_nat_the_rest_point() -> None:
    r = random.Random(7)
    readings = [street_handoff_raw_reading(_event("handoff_met", p=r.random())) for _ in range(20000)]
    mean = sum(readings) / len(readings)
    var = sum((x - mean) ** 2 for x in readings) / len(readings)
    assert mean == pytest.approx(1.0, abs=0.05)  # E[-ln U] = 1: ordinary, not signal
    assert var == pytest.approx(1.0, abs=0.1)


SPEC = domain_specs()[STREET_HANDOFF_DOMAIN]
T0 = NOW - timedelta(days=5)


def _timed(values):
    return [(T0 + timedelta(minutes=10 * i), v) for i, v in enumerate(values)]


def _calibrated(n: int, seed: int) -> list[float]:
    r = random.Random(seed)
    return [-math.log(max(r.random(), P_VALUE_FLOOR)) for _ in range(n)]


def test_domain_spec_is_registered_with_a_24h_rollback_window() -> None:
    assert SPEC.domain == "street_handoff"
    assert SPEC.target_surface == "prediction_expectation:street_handoff"
    assert SPEC.rollback_window_sec == 24 * 3600
    assert SPEC.default == Expectation(0.2, 0.01)


def test_too_few_handoffs_propose_nothing() -> None:
    assert assess(SPEC, _timed(_calibrated(250, 1)), current=SPEC.default) is None


def test_candidates_stay_inside_domain_bounds() -> None:
    for candidate in candidate_grid(SPEC, SPEC.default):
        assert SPEC.in_bounds(candidate)


def test_going_numb_loses_on_calibrated_handoffs() -> None:
    values = _calibrated(900, 2)
    gain = paired_gain(values, candidate=Expectation(0.2, 100.0), reference=SPEC.default, score_from=WARMUP_READINGS)
    assert gain is not None and gain.upper_bound < 0.0


def test_unset_surface_reads_as_the_default(tmp_path, monkeypatch) -> None:
    store = RuntimeControlSurfaceStore(sql_db_path=str(tmp_path / "control.sqlite3"))
    monkeypatch.setattr(mutation_control_surface, "_CONTROL_SURFACE_STORE", store)
    assert get_prediction_expectation(STREET_HANDOFF_DOMAIN) == SPEC.default


def test_handoff_surprise_z_reads_the_expectation() -> None:
    st = StreetAttentionState()
    assert st.handoff_surprise_z(1.0, alpha=0.2, min_variance=0.01) is None  # no baseline yet
    z = st.handoff_surprise_z(5.0, alpha=0.2, min_variance=0.01)
    assert z is not None and z > 10  # variance floor binds on the 2nd reading
