"""Producer-side wiring for ``orion.field.queue_contention``.

Uses injectable count readers / prebuilt counts — no Postgres or live
gateway required (same fake-store posture as ``test_digestion_significance``).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.field.queue_contention import SOURCE_DURABLE, SOURCE_GATEWAY, SOURCE_SEED
from orion.schemas.field_state import FieldStateV1

from app.digestion.queue_contention import (
    gateway_waiting_sum,
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
        SOURCE_GATEWAY: 1.0,
    }
    state.queue_contention_ewma_n = {
        SOURCE_SEED: 20,
        SOURCE_DURABLE: 20,
        SOURCE_GATEWAY: 20,
    }

    state = update_queue_contention_pressure(
        state,
        counts={
            SOURCE_SEED: 100.0,
            SOURCE_DURABLE: 10.0,
            SOURCE_GATEWAY: 1.0,
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
        readers={SOURCE_DURABLE: _ok, SOURCE_GATEWAY: _boom}
    )
    assert counts == {SOURCE_DURABLE: 3.0}
    assert SOURCE_GATEWAY not in counts


def test_gateway_waiting_sums_upstreams(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "upstreams": {
                    "a": {"waiting": 2, "inflight": 1},
                    "b": {"waiting": 1, "inflight": 0},
                }
            }

    monkeypatch.setattr(
        "app.digestion.queue_contention.requests.get",
        lambda *a, **k: _Resp(),
    )
    assert gateway_waiting_sum("http://llm-gateway:8210/admission") == 3.0


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
    state.queue_contention_ewma = {SOURCE_GATEWAY: 1.0}
    state.queue_contention_ewma_n = {SOURCE_GATEWAY: 5}

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
        queue_contention_readers={SOURCE_GATEWAY: lambda: 5.0},
    )

    assert state.queue_contention_score == 10.0
    assert state.queue_contention_driver == SOURCE_GATEWAY
