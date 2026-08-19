"""Producer-side wiring for `orion.field.significance` (2026-08-18, docs/
superpowers/specs/2026-08-16-level-aware-significance-design.md).

Verified separately, live, against the real database (see this patch's PR
description): `FieldDigesterStore.load_recent_field_json` and the full
`update_significance_pressure` pipeline both confirmed working end-to-end
against real `substrate_field_state` history before this suite was written.
These tests use a fake store -- same reasoning `test_tension_outreach_
trigger.py` already documents for mocking `_engine()` rather than standing up
a real Postgres in this suite.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.schemas.field_state import FieldStateV1

from app.digestion.significance import update_significance_pressure

BASE = datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc)


def _empty_state(tick_id: str, *, generated_at: datetime = BASE) -> FieldStateV1:
    return FieldStateV1(generated_at=generated_at, tick_id=tick_id)


class _FakeStore:
    def __init__(self, payloads: list[dict]) -> None:
        self._payloads = payloads
        self.calls = 0

    def load_recent_field_json(self, *, window_seconds: float) -> list[dict]:
        self.calls += 1
        return self._payloads


class _RaisingStore:
    def load_recent_field_json(self, *, window_seconds: float) -> list[dict]:
        raise RuntimeError("connection refused")


def _loaded_steady_payloads(n: int = 8) -> list[dict]:
    # Alternating pair -> median 0.8125, pstdev 0.0025 -- loaded_steady, same
    # hand-computed fixture shape test_field_significance.py's own headline
    # test uses.
    values = [0.810, 0.815] * (n // 2)
    return [{"node_vectors": {"node:athena": {"memory_pressure": v}}} for v in values]


def test_first_tick_always_recomputes() -> None:
    """No prior `sustained_load_computed_at` -- the throttle has nothing to
    compare against, so it must recompute rather than skip forever."""
    state = _empty_state("tick_0")
    store = _FakeStore(_loaded_steady_payloads())

    state = update_significance_pressure(
        state, store, window_seconds=900.0, check_interval_sec=30.0
    )

    assert store.calls == 1
    assert state.sustained_load_pressure > 0.0
    assert state.sustained_load_computed_at == BASE


def test_within_interval_does_not_recompute() -> None:
    """A tick 10s after the last real computation, with a 30s throttle, must
    carry the existing value forward WITHOUT touching the store."""
    state = _empty_state("tick_0")
    state.sustained_load_pressure = 0.5
    state.sustained_load_computed_at = BASE
    store = _FakeStore(_loaded_steady_payloads())

    later = _empty_state("tick_1", generated_at=BASE + timedelta(seconds=10))
    later.sustained_load_pressure = 0.5
    later.sustained_load_computed_at = BASE
    later = update_significance_pressure(
        later, store, window_seconds=900.0, check_interval_sec=30.0
    )

    assert store.calls == 0
    assert later.sustained_load_pressure == 0.5
    assert later.sustained_load_computed_at == BASE


def test_at_or_past_interval_recomputes() -> None:
    """Exactly at the throttle boundary -- must recompute, not skip. Mirrors
    this repo's own convention of testing exact-boundary behavior explicitly
    rather than only comfortably-inside/outside cases."""
    state = _empty_state("tick_1", generated_at=BASE + timedelta(seconds=30))
    state.sustained_load_pressure = 0.5
    state.sustained_load_computed_at = BASE
    store = _FakeStore(_loaded_steady_payloads())

    state = update_significance_pressure(
        state, store, window_seconds=900.0, check_interval_sec=30.0
    )

    assert store.calls == 1
    assert state.sustained_load_computed_at == BASE + timedelta(seconds=30)


def test_empty_window_gives_zero_and_still_advances_computed_at() -> None:
    """A real, successful computation that finds nothing loaded_steady is
    NOT a failure -- it advances sustained_load_computed_at like any other
    real computation, same as deviation_pressure's own 0.0-on-a-quiet-tick
    convention."""
    state = _empty_state("tick_0")
    store = _FakeStore([])

    state = update_significance_pressure(
        state, store, window_seconds=900.0, check_interval_sec=30.0
    )

    assert store.calls == 1
    assert state.sustained_load_pressure == 0.0
    assert state.sustained_load_computed_at == BASE


def test_store_failure_leaves_value_in_place_and_retries_next_tick() -> None:
    """A DB hiccup must not crash the digestion tick, and must NOT advance
    sustained_load_computed_at -- the honest failure mode is "retry on the
    very next hot tick", not "silently go stale for a full check_interval_
    sec", per this function's own docstring."""
    state = _empty_state("tick_0")
    state.sustained_load_pressure = 0.42
    state.sustained_load_computed_at = None
    store = _RaisingStore()

    state = update_significance_pressure(
        state, store, window_seconds=900.0, check_interval_sec=30.0
    )

    assert state.sustained_load_pressure == 0.42
    assert state.sustained_load_computed_at is None
