"""Producer-side update for ``orion.field.queue_contention``.

See ``orion/schemas/field_state.py``'s ``queue_contention_*`` docstrings and
``docs/superpowers/specs/2026-09-20-queue-contention-metric-gate.md``.

Runs inline on every hot digester tick (cheap count reads + O(1) EWMA),
unlike ``update_significance_pressure`` which throttles a real window query.
Count readers are injectable callables so unit tests never need Postgres;
production readers are three SQL counts (reading-seed queue, durable demands,
GPU pool waiting leases). A failing reader fails open: that source key is
omitted from ``counts`` rather than inventing 0 (which would falsely look like
calm against a warm EWMA).
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
    counts: dict[str, float] = {}
    for key, reader in readers.items():
        try:
            counts[key] = float(reader())
        except Exception:  # noqa: BLE001 - one bad source must not kill the tick
            logger.warning(
                "queue_contention_count_reader_failed key=%s", key, exc_info=True
            )
    return counts


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
) -> FieldStateV1:
    """Write ``queue_contention_*`` fields from ``counts`` + prior EWMA on state.

    Empty ``counts`` (every reader failed) leaves the prior reading in place
    and does not advance ``queue_contention_computed_at`` — retry next tick
    rather than stamp a quiet fabricated score.
    """
    if not counts:
        return state

    reading = score_queue_contention(
        counts,
        prev_ewma=state.queue_contention_ewma,
        prev_n=state.queue_contention_ewma_n,
        alpha=alpha,
        floor=floor,
    )
    state.queue_contention_score = reading.score
    state.queue_contention_driver = reading.driver
    state.queue_contention_ewma = reading.ewma
    state.queue_contention_ewma_n = reading.ewma_n
    state.queue_contention_computed_at = state.generated_at
    return state
