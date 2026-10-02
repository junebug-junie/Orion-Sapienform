"""Storage-write substrate lane projection (orion-sql-writer reporting on itself).

orion-sql-writer is the consumer that turns most bus channels into Postgres rows.
It counts its own write outcomes per table family into fixed windows and
publishes one grammar trace per window (``sql_writer.storage:<writer>:<window_id>``,
services/orion-sql-writer/app/write_health.py). The storage_write reducer
(orion/substrate/storage_write_loop/) folds those windows into one reading,
``write_failure_pressure`` on ``node:substrate.storage_write``, which the field
topology carries into ``capability:storage`` ``reliability_pressure``.

Counts and class names only -- never a payload, never an error message.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---- Wire contract shared by the producer (orion-sql-writer) and the reducer.
# Lives here, not in orion/substrate/, so the sql-writer can import it without
# executing orion/substrate/__init__.py (graph store, materializer).
STORAGE_WRITE_SOURCE_SERVICE = "orion-sql-writer"
STORAGE_WRITE_TRACE_PREFIX = "sql_writer.storage:"
# One atom per table family per window, plus one closing atom with the totals.
ROLE_FAMILY_WINDOW = "storage_write_window_observed"
ROLE_WINDOW_COMPLETED = "storage_writer_window_completed"

STORAGE_WRITE_NODE_ID = "node:substrate.storage_write"
STORAGE_WRITE_CHANNEL = "write_failure_pressure"

# Outcome classes. Exactly one per incoming write.
OUTCOME_COMMITTED = "committed"
# Unique-key hit on an idempotent table: the row is already there. Not a failure.
OUTCOME_DUPLICATE = "duplicate"
# The writer chose not to write (legacy drop, a patch that found nothing to patch).
# Not attempted, not counted in the denominator.
OUTCOME_SKIPPED = "skipped"
# No route for the kind: a config gap, watched by app/fallback_watch.py and
# app/route_coverage.py. Counted for inspection; never part of the reading,
# because it says nothing about whether storage works.
OUTCOME_UNROUTED = "unrouted"

# The write was attempted and the row did not reach its table. Only these count
# toward write_failure_pressure.
WRITE_FAILURE_CLASSES = frozenset(
    {
        # payload failed the pydantic contract (incl. extra="forbid" rejects, enum drift)
        "validation",
        # Postgres refused the row: NOT NULL, FK, CHECK, non-duplicate unique
        "constraint",
        # the row could not be encoded (e.g. datetime inside a JSON column, 2026-09-26)
        "serialization",
        # could not reach Postgres (connection refused, server starting/shutting down)
        "db_unavailable",
        # statement timeout / lock wait / the writer's own persist deadline
        "timeout",
        # any other database error
        "db_error",
        # the writer shed the event before trying (grammar queue full)
        "backpressure",
        # an exception the classifier does not recognise; still a lost write
        "other",
    }
)
ATTEMPT_CLASSES = WRITE_FAILURE_CLASSES | {OUTCOME_COMMITTED, OUTCOME_DUPLICATE}

# Overflow bucket when a window touches more families than the producer reports.
OTHER_FAMILY = "_other"
MAX_FAMILIES_PER_WINDOW = 32


class StorageWriteFamilyStateV1(BaseModel):
    """One table family's latest window, as the writer counted it."""

    model_config = ConfigDict(extra="forbid")

    family: str
    attempted: int = 0
    committed: int = 0
    duplicate: int = 0
    failed: int = 0
    skipped: int = 0
    unrouted: int = 0
    failure_classes: dict[str, int] = Field(default_factory=dict)
    # Write wall time inside the writer (validation + insert + commit), committed
    # and duplicate writes only. None when nothing committed this window.
    commit_p50_ms: int | None = None
    commit_p95_ms: int | None = None


class StorageWriteWindowCountV1(BaseModel):
    """What the rolling reading needs from one window: attempts and failures per family."""

    model_config = ConfigDict(extra="forbid")

    window_id: str
    window_end: datetime
    attempted: dict[str, int] = Field(default_factory=dict)
    failed: dict[str, int] = Field(default_factory=dict)


class StorageWriteProjectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    projection_id: str
    generated_at: datetime
    writer_node: str | None = None
    last_window_id: str | None = None
    last_window_end: datetime | None = None
    last_window_sec: float = 0.0
    # Totals of the latest window.
    attempted: int = 0
    committed: int = 0
    failed: int = 0
    unrouted: int = 0
    grammar_queue_max: int | None = None
    families: dict[str, StorageWriteFamilyStateV1] = Field(default_factory=dict)
    # Event-time rolling span the reading is computed over (oldest first).
    recent_windows: list[StorageWriteWindowCountV1] = Field(default_factory=list)
    # None = not measured (nothing attempted in the span), never a fabricated 0.0.
    write_failure_pressure: float | None = None
    reading: dict[str, Any] | None = None
    evidence_event_ids: list[str] = Field(default_factory=list)


__all__ = [
    "ATTEMPT_CLASSES",
    "MAX_FAMILIES_PER_WINDOW",
    "OTHER_FAMILY",
    "OUTCOME_COMMITTED",
    "OUTCOME_DUPLICATE",
    "OUTCOME_SKIPPED",
    "OUTCOME_UNROUTED",
    "ROLE_FAMILY_WINDOW",
    "ROLE_WINDOW_COMPLETED",
    "STORAGE_WRITE_CHANNEL",
    "STORAGE_WRITE_NODE_ID",
    "STORAGE_WRITE_SOURCE_SERVICE",
    "STORAGE_WRITE_TRACE_PREFIX",
    "StorageWriteFamilyStateV1",
    "StorageWriteProjectionV1",
    "StorageWriteWindowCountV1",
    "WRITE_FAILURE_CLASSES",
]
