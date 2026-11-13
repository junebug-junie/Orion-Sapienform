"""Orion Prime reliability (2026-11-03): what wakes Prime, and tries that never reach the model.

Live 10-30..11-03: 254 thoughts, 96 failed, 78 of them "RPC timeout" -- the call waited in the pool
queue for its whole 60 s. Those were recorded as failed thoughts and started the refractory interval.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.prime.state import (
    CONCERN_TTL,
    NO_MODEL_FREE_TRIES,
    QUIET_FLOOR,
    PrimeSnapshotV1,
    PrimeStateV1,
    classify_try,
    no_model_backoff,
    note_model_reached,
    note_no_model,
    observe,
    wants_starved_route,
    wants_to_think,
)

T0 = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
REFRACTORY = timedelta(seconds=300)
SNAP = PrimeSnapshotV1(binding=[], body_ok=True)


# --- what wakes Prime ---------------------------------------------------------------------------


def test_the_quiet_floor_wakes_prime_after_a_quarter_of_a_concerns_life():
    assert QUIET_FLOOR == CONCERN_TTL / 4
    state = PrimeStateV1(seen_binding=[], last_thought_at=T0)
    assert observe(state, SNAP, T0 + QUIET_FLOOR - timedelta(seconds=1)).pending_changes == []
    woke = observe(state, SNAP, T0 + QUIET_FLOOR)
    assert woke.pending_changes == ["quiet: nothing has woken me for 6 hours"]
    # Once: a second quiet tick before the thought adds nothing.
    assert observe(woke, SNAP, T0 + QUIET_FLOOR + timedelta(seconds=15)).pending_changes == woke.pending_changes


def test_the_quiet_floor_never_fires_before_the_first_thought_or_beside_a_real_change():
    assert [c for c in observe(PrimeStateV1(), SNAP, T0).pending_changes if c.startswith("quiet")] == []
    busy = PrimeStateV1(seen_binding=[], last_thought_at=T0, pending_changes=["quiet: nothing has woken me for 6 hours"])
    assert observe(busy, SNAP, T0 + QUIET_FLOOR).pending_changes == busy.pending_changes


# --- classifying a try --------------------------------------------------------------------------


@pytest.mark.parametrize("err, gw, kind", [
    (None, None, "model"),
    ("json_parse_failed", None, "model"),
    ("empty_llm_response", None, "model"),
    # Generation ran out of time on a granted seat: the model's (Prime's) own outcome.
    ("empty_llm_response", {"error": "timeout", "reason": "caller_budget_exhausted"}, "model"),
    ("empty_llm_response", {"error": "gpu_pool_unavailable", "reason": "deadline"}, "no_seat"),
    ("empty_llm_response", {"error": "gpu_pool_unavailable", "reason": "shed:cabinet_hot"}, "no_seat"),
    ("empty_llm_response", {"error": "gpu_pool_recalled", "reason": "lease_lost"}, "no_seat"),
    ("empty_llm_response", {"error": "upstream_http_5xx", "reason": None}, "upstream_5xx"),
    ("RPC timeout waiting on orion:mind:llm:reply:abc", None, "no_reply"),
    ("TimeoutError: RPC timeout waiting on x", None, "no_reply"),
])
def test_classify_try(err, gw, kind):
    assert classify_try(err, gw) == kind


# --- no-model tries -----------------------------------------------------------------------------


def test_a_no_model_try_is_not_a_thought_and_the_next_tick_retries():
    state = PrimeStateV1(pending_changes=["quiet: nothing has woken me for 6 hours"], last_thought_at=T0 - timedelta(hours=1))
    after = note_no_model(state, "no_seat", T0, base=REFRACTORY)
    assert after.last_thought_at == state.last_thought_at          # the refractory clock did not start
    assert after.pending_changes == state.pending_changes          # still something to think about
    assert after.no_model_streak == 1 and after.no_model_tries_total == 1 and after.retry_not_before is None
    assert wants_to_think(after, T0 + timedelta(seconds=15), refractory=REFRACTORY)


def test_only_a_no_seat_try_escalates_to_the_starved_route():
    state = PrimeStateV1()
    assert not wants_starved_route(state)
    assert wants_starved_route(note_no_model(state, "no_seat", T0, base=REFRACTORY))
    assert not wants_starved_route(note_no_model(state, "no_reply", T0, base=REFRACTORY))
    assert not wants_starved_route(note_no_model(state, "upstream_5xx", T0, base=REFRACTORY))
    reached = note_model_reached(note_no_model(state, "no_seat", T0, base=REFRACTORY))
    assert not wants_starved_route(reached) and reached.no_model_streak == 0 and reached.no_model_tries_total == 1


def test_no_hot_loop_consecutive_no_model_tries_back_off_up_to_an_hour():
    assert [no_model_backoff(n, base=REFRACTORY) for n in range(1, NO_MODEL_FREE_TRIES + 1)] == [timedelta(0)] * 3
    assert no_model_backoff(NO_MODEL_FREE_TRIES + 1, base=REFRACTORY) == REFRACTORY
    assert no_model_backoff(NO_MODEL_FREE_TRIES + 2, base=REFRACTORY) == 2 * REFRACTORY
    assert no_model_backoff(50, base=REFRACTORY) == timedelta(hours=1)
    state, now = PrimeStateV1(pending_changes=["x"]), T0
    for _ in range(NO_MODEL_FREE_TRIES + 1):
        state = note_no_model(state, "no_reply", now, base=REFRACTORY)
    assert state.retry_not_before == now + REFRACTORY
    assert not wants_to_think(state, now + REFRACTORY - timedelta(seconds=1), refractory=REFRACTORY)
    assert wants_to_think(state, now + REFRACTORY, refractory=REFRACTORY)


def test_a_shed_answer_s_own_retry_after_is_honoured_from_the_first_try():
    state = note_no_model(PrimeStateV1(pending_changes=["x"]), "no_seat", T0, base=REFRACTORY, retry_after_sec=90)
    assert state.retry_not_before == T0 + timedelta(seconds=90)
    assert not wants_to_think(state, T0 + timedelta(seconds=60), refractory=REFRACTORY)


def test_old_persisted_state_without_the_new_fields_still_loads():
    old = ('{"schema_version":"prime.state.v1","concerns":[],"open_nominations":[],"pending_changes":[],'
           '"recent_outcomes":[],"last_thought_at":null,"last_reading":"","thoughts_total":3,'
           '"seen_active_arc_id":"arc1","seen_binding":[]}')
    state = PrimeStateV1.model_validate_json(old)
    assert state.no_model_streak == 0 and state.last_try_kind is None and state.retry_not_before is None
