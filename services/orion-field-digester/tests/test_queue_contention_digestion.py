"""Producer-side wiring for ``orion.field.queue_contention``.

Uses injectable count readers / prebuilt counts — no Postgres or live
gateway required (same fake-store posture as ``test_digestion_significance``).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.field.queue_contention import SOURCE_DURABLE, SOURCE_GPU_POOL, SOURCE_SEED
from orion.schemas.field_state import FieldStateV1

from app.digestion.queue_contention import (
    default_queue_contention_readers,
    read_queue_contention_counts,
    update_queue_contention_pressure,
)
from app.tensor.update_rules import run_digestion_tick

BASE = datetime(2026, 9, 20, 15, 0, 0, tzinfo=timezone.utc)


def _empty_state(tick_id: str = "tick_qc_0") -> FieldStateV1:
    return FieldStateV1(generated_at=BASE, tick_id=tick_id)


class _NoHistoryStore:
    def load_recent_field_json(self, *, window_seconds: float) -> list[dict]:
        return []


def test_update_writes_score_driver_and_ewma_from_counts() -> None:
    state = _empty_state()
    state.queue_contention_ewma = {SOURCE_DURABLE: 2.0}
    state.queue_contention_ewma_n = {SOURCE_DURABLE: 50}

    state = update_queue_contention_pressure(
        state,
        counts={SOURCE_DURABLE: 10.0},
        alpha=0.0,
        floor=1.0,
    )

    assert state.queue_contention_score == 10.0
    assert state.queue_contention_driver == SOURCE_DURABLE
    assert state.queue_contention_ewma[SOURCE_DURABLE] == 2.0
    assert state.queue_contention_ewma_n[SOURCE_DURABLE] == 51
    assert state.queue_contention_computed_at == BASE


def test_update_max_driver_across_sources() -> None:
    state = _empty_state()
    state.queue_contention_ewma = {
        SOURCE_SEED: 100.0,
        SOURCE_DURABLE: 2.0,
        SOURCE_GPU_POOL: 1.0,
    }
    state.queue_contention_ewma_n = {
        SOURCE_SEED: 20,
        SOURCE_DURABLE: 20,
        SOURCE_GPU_POOL: 20,
    }

    state = update_queue_contention_pressure(
        state,
        counts={
            SOURCE_SEED: 100.0,
            SOURCE_DURABLE: 10.0,
            SOURCE_GPU_POOL: 1.0,
        },
        alpha=0.0,
    )

    assert state.queue_contention_score == 10.0
    assert state.queue_contention_driver == SOURCE_DURABLE


def test_empty_counts_leaves_prior_reading() -> None:
    state = _empty_state()
    state.queue_contention_score = 4.0
    state.queue_contention_driver = SOURCE_SEED
    state.queue_contention_computed_at = None

    state = update_queue_contention_pressure(state, counts={}, alpha=0.01)

    assert state.queue_contention_score == 4.0
    assert state.queue_contention_driver == SOURCE_SEED
    assert state.queue_contention_computed_at is None


def test_read_counts_omits_failing_reader() -> None:
    def _ok() -> float:
        return 3.0

    def _boom() -> float:
        raise RuntimeError("gateway down")

    counts = read_queue_contention_counts(
        readers={SOURCE_DURABLE: _ok, SOURCE_GPU_POOL: _boom}
    )
    assert counts == {SOURCE_DURABLE: 3.0}
    assert SOURCE_GPU_POOL not in counts


def test_default_readers_are_three_sql_counts_with_the_gpu_pool_source() -> None:
    class Store:
        def count_world_pulse_seed_pending(self):
            return 1

        def count_durable_demand_pending(self):
            return 2

        def count_gpu_pool_waiting(self):
            return 7

    readers = default_queue_contention_readers(Store())
    assert set(readers) == {SOURCE_SEED, SOURCE_DURABLE, SOURCE_GPU_POOL}
    assert read_queue_contention_counts(readers=readers) == {SOURCE_SEED: 1.0, SOURCE_DURABLE: 2.0, SOURCE_GPU_POOL: 7.0}

def test_run_digestion_tick_wires_queue_contention_after_significance() -> None:
    state = _empty_state()
    state.queue_contention_ewma = {SOURCE_SEED: 50.0}
    state.queue_contention_ewma_n = {SOURCE_SEED: 10}

    run_digestion_tick(
        state,
        perturbations=[],
        decay_rate=1.0,
        diffusion_rate=0.0,
        staleness_threshold_sec=90.0,
        store=_NoHistoryStore(),
        significance_window_seconds=900.0,
        significance_check_interval_sec=30.0,
        queue_contention_alpha=0.0,
        queue_contention_counts={SOURCE_SEED: 250.0},  # 5x → score 10
    )

    assert state.queue_contention_score == 10.0
    assert state.queue_contention_driver == SOURCE_SEED
    assert state.queue_contention_computed_at == BASE


def test_run_digestion_tick_skips_when_alpha_unset() -> None:
    state = _empty_state()
    run_digestion_tick(
        state,
        perturbations=[],
        decay_rate=1.0,
        diffusion_rate=0.0,
        staleness_threshold_sec=90.0,
        store=_NoHistoryStore(),
        significance_window_seconds=900.0,
        significance_check_interval_sec=30.0,
        # queue_contention_alpha defaults None → skip
    )
    assert state.queue_contention_score == 0.0
    assert state.queue_contention_computed_at is None


def test_run_digestion_tick_uses_injectable_readers() -> None:
    state = _empty_state()
    state.queue_contention_ewma = {SOURCE_GPU_POOL: 1.0}
    state.queue_contention_ewma_n = {SOURCE_GPU_POOL: 5}

    run_digestion_tick(
        state,
        perturbations=[],
        decay_rate=1.0,
        diffusion_rate=0.0,
        staleness_threshold_sec=90.0,
        store=_NoHistoryStore(),
        significance_window_seconds=900.0,
        significance_check_interval_sec=30.0,
        queue_contention_alpha=0.0,
        queue_contention_readers={SOURCE_GPU_POOL: lambda: 5.0},
    )

    assert state.queue_contention_score == 10.0
    assert state.queue_contention_driver == SOURCE_GPU_POOL


# --- Oldest-wait component (2026-09-25) -------------------------------------

from orion.field.queue_contention import OLDEST_WAIT_SUFFIX  # noqa: E402

from app.digestion.queue_contention import (  # noqa: E402
    default_queue_contention_age_readers,
    read_queue_contention_oldest_waits,
)


def test_stuck_queue_through_full_tick_names_oldest_wait() -> None:
    """Count flat at its baseline, oldest item 10 days old -> nonzero, driver names staleness."""
    state = _empty_state()
    state.queue_contention_ewma = {SOURCE_SEED: 144.0}
    state.queue_contention_ewma_n = {SOURCE_SEED: 1176}

    run_digestion_tick(
        state,
        perturbations=[],
        decay_rate=0.0,
        diffusion_rate=0.0,
        staleness_threshold_sec=60.0,
        store=_NoHistoryStore(),
        significance_window_seconds=900.0,
        significance_check_interval_sec=30.0,
        queue_contention_alpha=0.01,
        queue_contention_readers={SOURCE_SEED: lambda: 144.0},
        queue_contention_age_readers={SOURCE_SEED: lambda: 10 * 86400.0},
    )
    assert state.queue_contention_score == 10.0
    assert state.queue_contention_driver == SOURCE_SEED + OLDEST_WAIT_SUFFIX
    assert state.queue_contention_computed_at == BASE


def test_empty_queue_through_full_tick_is_zero() -> None:
    state = _empty_state()
    state.queue_contention_ewma = {SOURCE_GPU_POOL: 0.0}
    state.queue_contention_ewma_n = {SOURCE_GPU_POOL: 100}
    state.queue_contention_score = 7.0
    state.queue_contention_driver = SOURCE_GPU_POOL + OLDEST_WAIT_SUFFIX

    run_digestion_tick(
        state,
        perturbations=[],
        decay_rate=0.0,
        diffusion_rate=0.0,
        staleness_threshold_sec=60.0,
        store=_NoHistoryStore(),
        significance_window_seconds=900.0,
        significance_check_interval_sec=30.0,
        queue_contention_alpha=0.01,
        queue_contention_readers={SOURCE_GPU_POOL: lambda: 0.0},
        queue_contention_age_readers={SOURCE_GPU_POOL: lambda: 0.0},
    )
    assert state.queue_contention_score == 0.0
    assert state.queue_contention_driver is None


def test_expected_wait_override_threads_through_tick() -> None:
    state = _empty_state()
    run_digestion_tick(
        state,
        perturbations=[],
        decay_rate=0.0,
        diffusion_rate=0.0,
        staleness_threshold_sec=60.0,
        store=_NoHistoryStore(),
        significance_window_seconds=900.0,
        significance_check_interval_sec=30.0,
        queue_contention_alpha=0.01,
        queue_contention_readers={},
        queue_contention_age_readers={SOURCE_GPU_POOL: lambda: 300.0},
        queue_contention_expected_wait_sec={SOURCE_GPU_POOL: 100.0},
    )
    assert state.queue_contention_score == pytest.approx(5.0)
    assert state.queue_contention_driver == SOURCE_GPU_POOL + OLDEST_WAIT_SUFFIX


def test_failed_age_reader_is_omitted_not_zero() -> None:
    def boom() -> float:
        raise RuntimeError("db down")

    ages = read_queue_contention_oldest_waits(
        readers={SOURCE_SEED: boom, SOURCE_DURABLE: lambda: 12.0}
    )
    assert ages == {SOURCE_DURABLE: 12.0}


def test_all_readers_failed_keeps_prior_reading() -> None:
    state = _empty_state()
    state.queue_contention_score = 4.0
    state.queue_contention_driver = SOURCE_SEED + OLDEST_WAIT_SUFFIX
    update_queue_contention_pressure(state, counts={}, alpha=0.01, oldest_wait_sec={})
    assert state.queue_contention_score == 4.0
    assert state.queue_contention_computed_at is None


class _AgeStore:
    def oldest_world_pulse_seed_pending_age_sec(self) -> float:
        return 1.0

    def oldest_durable_demand_pending_age_sec(self) -> float:
        return 2.0

    def oldest_gpu_pool_waiting_age_sec(self) -> float:
        return 3.0


def test_default_age_readers_map_every_source() -> None:
    readers = default_queue_contention_age_readers(_AgeStore())
    assert {k: r() for k, r in readers.items()} == {
        SOURCE_SEED: 1.0,
        SOURCE_DURABLE: 2.0,
        SOURCE_GPU_POOL: 3.0,
    }


class _RecordingConn:
    def __init__(self, sink: list[str], value):
        self._sink, self._value = sink, value

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, clause):
        self._sink.append(" ".join(str(clause).split()))
        value = self._value

        class _R:
            def scalar(self_inner):
                return value

        return _R()


class _RecordingEngine:
    def __init__(self, value):
        self.sql: list[str] = []
        self._value = value

    def connect(self):
        return _RecordingConn(self.sql, self._value)


def _store_with(value):
    from app.store import FieldDigesterStore

    store = FieldDigesterStore.__new__(FieldDigesterStore)
    store._engine = _RecordingEngine(value)
    return store


def test_age_sql_filters_match_the_queue_they_describe() -> None:
    """The SQL itself, not an injected lambda: a wrong table/status/order would pass every other test."""
    store = _store_with(12.5)
    assert store.oldest_world_pulse_seed_pending_age_sec() == 12.5
    assert store.oldest_durable_demand_pending_age_sec() == 12.5
    assert store.oldest_gpu_pool_waiting_age_sec() == 12.5
    seed_sql, durable_sql, pool_sql = store._engine.sql
    assert "FROM world_pulse_read_seed WHERE status = 'pending'" in seed_sql
    # Head of line in CLAIM_SQL's own order.
    assert "ORDER BY priority ASC, attempts ASC, created_at ASC, seed_id ASC LIMIT 1" in seed_sql
    # Durable: legacy pending demands UNION waiting pool holds (stage 4.4); behaviour is pinned on
    # real Postgres in test_durable_waiting_sql_postgres.py.
    assert "FROM durable_resource_demands d WHERE d.status = 'pending'" in durable_sql
    assert "h.kind = 'hold' AND h.status IN ('queued', 'backlogged')" in durable_sql
    assert "min(waiting_since)" in durable_sql
    # Queued only: backlogged leases may wait up to backlog_max_age_sec by design. Requests only:
    # a waiting hold is durable_demand_pending's, never double-counted here.
    assert "FROM gpu_pool_leases WHERE status = 'queued' AND kind = 'request'" in pool_sql
    assert "coalesce(queued_since, created_at)" in pool_sql


def test_seed_age_order_matches_claim_sql() -> None:
    from orion.world_pulse_read.queue import CLAIM_SQL

    assert "ORDER BY priority ASC, attempts ASC, created_at ASC, seed_id ASC" in " ".join(CLAIM_SQL.split())


def test_age_sql_empty_queue_is_zero_and_negative_clamped() -> None:
    assert _store_with(None).oldest_gpu_pool_waiting_age_sec() == 0.0
    assert _store_with(-3.0).oldest_durable_demand_pending_age_sec() == 0.0


def test_durable_waiting_sql_joins_on_the_pool_clients_holder_shape() -> None:
    """The hold side is found by holder text; if durable-runs (4.5) and this reader disagreed on
    it, every resumed run would be counted twice across the cutover."""
    from app.store import DURABLE_WAITING_SQL
    from orion.gpu_pool.client import DURABLE_RUN_HOLDER_PREFIX, durable_run_holder
    from orion.schemas.gpu_pool import TERMINAL_STATUSES

    assert f"h.holder = '{DURABLE_RUN_HOLDER_PREFIX}' || d.run_id" in DURABLE_WAITING_SQL
    assert durable_run_holder("r") == f"{DURABLE_RUN_HOLDER_PREFIX}r"
    for status in TERMINAL_STATUSES:
        assert f"'{status}'" in DURABLE_WAITING_SQL
