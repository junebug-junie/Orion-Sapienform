"""Producer-side update for ``orion.field.queue_contention``.

See ``orion/schemas/field_state.py``'s ``queue_contention_*`` docstrings and
``docs/superpowers/specs/2026-09-20-queue-contention-metric-gate.md``.

Runs inline on every hot digester tick (cheap count reads + O(1) EWMA),
unlike ``update_significance_pressure`` which throttles a real window query.
Count readers are injectable callables so unit tests never need Postgres;
production readers are three SQL counts (reading-seed queue, durable demands,
GPU pool waiting leases) plus three matching oldest-wait reads (2026-09-25:
seconds the oldest waiting item has waited, 0.0 for an empty queue). A failing
reader fails open: that source key is omitted rather than inventing 0 (which
would falsely look like calm against a warm EWMA or a stuck queue).
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

from orion.field.queue_contention import (
    SOURCE_DURABLE,
    SOURCE_GPU_POOL,
    SOURCE_SEED,
    score_queue_contention,
)
from orion.schemas.field_state import FieldStateV1

logger = logging.getLogger("orion.field.digester.queue_contention")


def read_queue_contention_counts(
    *,
    readers: Mapping[str, Callable[[], float]],
) -> dict[str, float]:
    """Invoke injectable readers; omit any key whose reader raises.

    Fail-open per source: a down gateway must not crash the digestion tick
    and must not write a fabricated calm 0 that would drag the EWMA.
    """
    return _read_fail_open(readers, what="count")


def read_queue_contention_oldest_waits(
    *,
    readers: Mapping[str, Callable[[], float]],
) -> dict[str, float]:
    """Oldest-wait sibling of :func:`read_queue_contention_counts`; same fail-open rule.

    A failed age read must be omitted, not written as 0.0: 0.0 means "nothing
    has waited", which is exactly the false calm this component exists to catch.
    """
    return _read_fail_open(readers, what="oldest_wait")


def _read_fail_open(
    readers: Mapping[str, Callable[[], float]], *, what: str
) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, reader in readers.items():
        try:
            values[key] = float(reader())
        except Exception:  # noqa: BLE001 - one bad source must not kill the tick
            logger.warning(
                "queue_contention_%s_reader_failed key=%s", what, key, exc_info=True
            )
    return values


def default_queue_contention_age_readers(store) -> dict[str, Callable[[], float]]:
    """Production oldest-wait readers: seconds the oldest waiting item has waited (0.0 if empty)."""

    def _seed() -> float:
        return float(store.oldest_world_pulse_seed_pending_age_sec())

    def _durable() -> float:
        return float(store.oldest_durable_demand_pending_age_sec())

    def _gpu_pool() -> float:
        return float(store.oldest_gpu_pool_waiting_age_sec())

    return {
        SOURCE_SEED: _seed,
        SOURCE_DURABLE: _durable,
        SOURCE_GPU_POOL: _gpu_pool,
    }


def default_queue_contention_readers(store) -> dict[str, Callable[[], float]]:
    """Production readers: three SQL counts."""

    def _seed() -> float:
        return float(store.count_world_pulse_seed_pending())

    def _durable() -> float:
        return float(store.count_durable_demand_pending())

    def _gpu_pool() -> float:
        return float(store.count_gpu_pool_waiting())

    return {
        SOURCE_SEED: _seed,
        SOURCE_DURABLE: _durable,
        SOURCE_GPU_POOL: _gpu_pool,
    }


def update_queue_contention_pressure(
    state: FieldStateV1,
    *,
    counts: Mapping[str, float],
    alpha: float,
    floor: float = 1.0,
    oldest_wait_sec: Mapping[str, float] | None = None,
    expected_wait_sec: Mapping[str, float] | None = None,
) -> FieldStateV1:
    """Write ``queue_contention_*`` fields from ``counts`` + oldest waits + prior EWMA on state.

    Empty ``counts`` AND empty ``oldest_wait_sec`` (every reader failed)
    leaves the prior reading in place and does not advance
    ``queue_contention_computed_at`` — retry next tick rather than stamp a
    quiet fabricated score.

    The oldest waits are not persisted on ``FieldStateV1`` (``extra="forbid"``:
    a new field is a consumer-first migration across every reader). They land
    in ``queue_contention_driver`` (``<source>:oldest_wait``) and the score;
    the raw age is re-derivable from Postgres at any time.
    """
    if not counts and not oldest_wait_sec:
        return state

    reading = score_queue_contention(
        counts,
        prev_ewma=state.queue_contention_ewma,
        prev_n=state.queue_contention_ewma_n,
        alpha=alpha,
        floor=floor,
        oldest_wait_sec=oldest_wait_sec,
        expected_wait_sec=expected_wait_sec,
    )
    if reading.driver is not None and reading.driver != state.queue_contention_driver:
        # One line per driver change, not per tick: the only place the raw age is visible.
        logger.info(
            "queue_contention_driver_changed driver=%s score=%.2f counts=%s oldest_wait_sec=%s",
            reading.driver,
            reading.score,
            reading.raw,
            {k: round(v) for k, v in reading.oldest_wait_sec.items()},
        )
    state.queue_contention_score = reading.score
    state.queue_contention_driver = reading.driver
    state.queue_contention_ewma = reading.ewma
    state.queue_contention_ewma_n = reading.ewma_n
    state.queue_contention_computed_at = state.generated_at
    return state
