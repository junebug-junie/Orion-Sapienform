"""Temporal Self: a pure reducer that binds Orion's day into one chronology of arcs.

Spec: docs/superpowers/specs/2026-09-26-temporal-self-design.md (PR #2369, rev 4), patch 2.
No I/O, no LLM, no narrative. The patch-3 driver (orion-durable-runs) reads rows, shapes
them with ``sources``/``broadcast``, folds them here, and persists arcs, frame and days.
"""

from orion.temporal_self.arcs import (
    ReducerConfig,
    advance_clock,
    close_day,
    drain_closed_days,
    fold,
    initial_state,
)
from orion.temporal_self.frame import build_frame

__all__ = [
    "ReducerConfig",
    "advance_clock",
    "build_frame",
    "close_day",
    "drain_closed_days",
    "fold",
    "initial_state",
]
