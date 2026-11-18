"""The sql-writer reporting on its own writes (storage-write organ, 2026-10-02).

## What this measures

Every incoming write the writer handles ends in exactly one outcome: it was
committed, it hit an idempotent duplicate, or it did not reach its table (and,
for most paths, landed in ``bus_fallback_log`` instead). This module counts
those outcomes per **table family** (the target table) into a fixed window and,
once per window, publishes one grammar trace on ``orion:grammar:event``:

- one ``storage_write_window_observed`` atom per family that saw traffic, with
  attempted / committed / duplicate / failed counts, failure counts by class,
  and write wall-time p50/p95 for committed writes;
- one ``storage_writer_window_completed`` atom with the window totals and the
  grammar queue high-water mark.

The substrate reducer (``orion/substrate/storage_write_loop/``) turns that into
``write_failure_pressure``. Nothing here reads a clock other than the window
boundaries, and nothing here touches the database.

## Why per window, never per write

The writer is also the writer of ``grammar_events``. A per-write atom about a
grammar write would itself be a grammar write: a feedback loop. Two guards:

1. Aggregation: at most ``MAX_FAMILIES_PER_WINDOW + 2`` atoms per window,
   whatever the write volume (live: ~300 table writes and ~265 grammar events a
   minute on 2026-10-02).
2. Self-exclusion: grammar events carrying this organ's own trace prefix are
   never counted (:func:`counts_grammar_event`), so the report does not count
   its own previous report.

## Absence

A silent writer publishes nothing, and the reading downstream expires (field
digester ``EXPIRING_NODE_CHANNELS``) instead of holding a calm value. A window
with no traffic still publishes its closing atom (so "alive and idle" is
distinguishable from "dead" in the ledger), but carries no family atoms, and
the reducer reports "not measured" rather than 0.0 for it.

Known bias: this report is itself a grammar event, so it travels the same
queue and the same database it reports on. When the grammar path (or Postgres)
is failing, the report about that failure is lost along with it, and the
reading goes unmeasured instead of high. Safe (never calm), but grammar-family
and full-outage failures are under-reported. A bus-direct path to the field
would fix it; deliberately not built here.

## Classification

From the exception object or the error text the writer already records on its
fallback row. Text is matched rather than types because several paths only
have the text (``_write_fallback(..., str(e))``), and SQLAlchemy wraps the
DBAPI error, so its message carries the inner class name, e.g.
``(builtins.TypeError) Object of type datetime is not JSON serializable`` (the
2026-09-26 home-cooling burst, 863 rows).
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.schemas.storage_write_projection import (
    ATTEMPT_CLASSES,
    MAX_FAMILIES_PER_WINDOW,
    OTHER_FAMILY,
    OUTCOME_COMMITTED,
    OUTCOME_DUPLICATE,
    OUTCOME_SKIPPED,
    OUTCOME_UNROUTED,
    ROLE_FAMILY_WINDOW,
    ROLE_WINDOW_COMPLETED,
    STORAGE_WRITE_SOURCE_SERVICE,
    STORAGE_WRITE_TRACE_PREFIX,
    WRITE_FAILURE_CLASSES,
)

logger = logging.getLogger("sql-writer.write_health")

# A literal (not the imported constant) so tests/test_grammar_event_producer_catalog.py's
# static scan can resolve this producer; pinned equal to the contract below.
SOURCE_SERVICE = "orion-sql-writer"
assert SOURCE_SERVICE == STORAGE_WRITE_SOURCE_SERVICE

GRAMMAR_FAMILY = "grammar_events"
# Latency samples kept per family per window (bounded; percentiles are over these).
_MAX_LATENCY_SAMPLES = 2048


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

# Order matters: the first match wins. Each entry: (class, lowercase needles).
_CLASS_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # pydantic's own message prefix ("1 validation error for CockpitHopV1").
    ("validation", ("validation error", "normalization failed")),
    ("backpressure", ("grammar queue full",)),
    ("unrouted", ("unknown kind",)),
    (
        "timeout",
        (
            "statement timeout",
            "querycanceled",
            "canceling statement",
            "lock timeout",
            "locknotavailable",
            "persist timeout",
            "batch timeout",
            "timeouterror",
        ),
    ),
    (
        "db_unavailable",
        (
            "could not connect",
            "connection refused",
            "the database system is",
            "server closed the connection",
            "connection to server",
            "terminating connection",
            "could not translate host name",
            "too many clients",
            "remaining connection slots",
        ),
    ),
    (
        "serialization",
        (
            "not json serializable",
            "builtins.typeerror",
            "can't adapt type",
            "cannot adapt type",
            "invalid input syntax",
            "invalid text representation",
            "numeric field overflow",
            "value too long",
            "out of range",
            "dataerror",
            "invalid byte sequence",
        ),
    ),
    (
        "constraint",
        (
            "integrityerror",
            "violates not-null",
            "violates foreign key",
            "violates check constraint",
            "violates unique constraint",
            "notnullviolation",
            "foreignkeyviolation",
            "checkviolation",
        ),
    ),
    (
        "db_error",
        (
            "operationalerror",
            "programmingerror",
            "internalerror",
            "databaseerror",
            "psycopg2",
            "sqlalchemy",
            "undefinedcolumn",
            "undefinedtable",
        ),
    ),
)


# Error text can carry the payload (SQLAlchemy appends "[SQL: ...] [parameters: ...]",
# Postgres adds a DETAIL line with row data, pydantic appends "input_value=...").
# Only the first line, cut before any of these markers, is matched, so a row
# whose content says "timeout" cannot pick its own class.
_PAYLOAD_MARKERS = (
    "[sql:",
    "[parameters:",
    "input_value=",
    "(background on this error",
    # Postgres puts row data in a DETAIL line ("Failing row contains (...)",
    # "Key (id)=(...)") BEFORE SQLAlchemy's [SQL: ...]; stored error text may
    # have its newlines flattened, so the marker is matched on its own too.
    "detail:",
)
_HEAD_CHARS = 400


def _message_head(text: str) -> str:
    # pydantic and Postgres both put the class-bearing summary on the first line.
    lowered = text.lower().split("\n", 1)[0]
    cut = len(lowered)
    for marker in _PAYLOAD_MARKERS:
        idx = lowered.find(marker)
        if 0 <= idx < cut:
            cut = idx
    return lowered[: min(cut, _HEAD_CHARS)]


def classify_write_error(error: BaseException | str | None) -> str:
    """Map an exception or the writer's recorded error text to one outcome class.

    Returns one of ``WRITE_FAILURE_CLASSES`` or ``"unrouted"``; never raises."""
    if error is None:
        return "other"
    if isinstance(error, BaseException):
        if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
            return "timeout"
        text = f"{type(error).__module__}.{type(error).__name__} {error}"
        try:
            from pydantic import ValidationError

            if isinstance(error, ValidationError):
                return "validation"
        except Exception:  # pragma: no cover - pydantic is always present
            pass
    else:
        text = str(error)
    lowered = _message_head(text)
    for cls, needles in _CLASS_RULES:
        if any(n in lowered for n in needles):
            return cls
    return "other"


# ---------------------------------------------------------------------------
# Window recorder
# ---------------------------------------------------------------------------


def _percentile(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


@dataclass
class _FamilyBucket:
    classes: dict[str, int] = field(default_factory=dict)
    latencies_ms: list[float] = field(default_factory=list)

    def add(self, outcome: str, count: int, latency_ms: float | None) -> None:
        self.classes[outcome] = self.classes.get(outcome, 0) + count
        if (
            latency_ms is not None
            and outcome in (OUTCOME_COMMITTED, OUTCOME_DUPLICATE)
            and len(self.latencies_ms) < _MAX_LATENCY_SAMPLES
        ):
            self.latencies_ms.append(max(0.0, float(latency_ms)))

    def merge(self, other: "_FamilyBucket") -> None:
        for k, v in other.classes.items():
            self.classes[k] = self.classes.get(k, 0) + v
        room = _MAX_LATENCY_SAMPLES - len(self.latencies_ms)
        if room > 0:
            self.latencies_ms.extend(other.latencies_ms[:room])

    @property
    def attempted(self) -> int:
        return sum(n for k, n in self.classes.items() if k in ATTEMPT_CLASSES)

    @property
    def failed(self) -> int:
        return sum(n for k, n in self.classes.items() if k in WRITE_FAILURE_CLASSES)

    def summary(self, family: str) -> str:
        lat = sorted(self.latencies_ms)
        p50 = _percentile(lat, 0.5)
        p95 = _percentile(lat, 0.95)
        failures = "|".join(
            f"{k}:{v}" for k, v in sorted(self.classes.items()) if k in WRITE_FAILURE_CLASSES and v
        ) or "none"
        return (
            f"family={family} attempted={self.attempted} "
            f"committed={self.classes.get(OUTCOME_COMMITTED, 0)} "
            f"duplicate={self.classes.get(OUTCOME_DUPLICATE, 0)} "
            f"failed={self.failed} skipped={self.classes.get(OUTCOME_SKIPPED, 0)} "
            f"unrouted={self.classes.get(OUTCOME_UNROUTED, 0)} classes={failures} "
            f"p50_ms={'none' if p50 is None else int(round(p50))} "
            f"p95_ms={'none' if p95 is None else int(round(p95))}"
        )


def _family_key(family: str | None) -> str:
    key = str(family or "").strip().lower()
    # The summary is a whitespace/comma separated kv line; keep the key a single token.
    key = "".join(ch if (ch.isalnum() or ch in "_.-") else "_" for ch in key)
    return key or OTHER_FAMILY


class WriteHealthRecorder:
    """Counts write outcomes into the current window. Thread-safe: writes complete
    on worker threads (``asyncio.to_thread``) and on the event loop."""

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: dict[str, _FamilyBucket] = {}
        self._grammar_queue_max = 0
        self._window_start = self._clock()

    def record(
        self,
        family: str | None,
        outcome: str,
        *,
        count: int = 1,
        latency_ms: float | None = None,
    ) -> None:
        if count <= 0:
            return
        key = _family_key(family)
        with self._lock:
            if key not in self._buckets and len(self._buckets) >= MAX_FAMILIES_PER_WINDOW:
                key = OTHER_FAMILY
            self._buckets.setdefault(key, _FamilyBucket()).add(outcome, int(count), latency_ms)

    def note_grammar_queue_depth(self, depth: int) -> None:
        with self._lock:
            if depth > self._grammar_queue_max:
                self._grammar_queue_max = int(depth)

    def drain(self) -> tuple[float, float, dict[str, _FamilyBucket], int]:
        with self._lock:
            start, end = self._window_start, self._clock()
            buckets, self._buckets = self._buckets, {}
            queue_max, self._grammar_queue_max = self._grammar_queue_max, 0
            self._window_start = end
        return start, end, buckets, queue_max


_recorder: WriteHealthRecorder | None = None
_enabled = False


def get_recorder() -> WriteHealthRecorder:
    global _recorder
    if _recorder is None:
        _recorder = WriteHealthRecorder()
    return _recorder


def set_enabled(enabled: bool) -> None:
    """Called once at startup from the flag. Off: every hook below is a no-op."""
    global _enabled
    _enabled = bool(enabled)


def is_enabled() -> bool:
    return _enabled


def reset_for_tests() -> None:
    global _recorder, _enabled
    _recorder = None
    _enabled = False


def counts_grammar_event(trace_id: str | None) -> bool:
    """This organ's own reports are never counted as grammar writes."""
    return not str(trace_id or "").startswith(STORAGE_WRITE_TRACE_PREFIX)


def record_grammar(
    trace_ids: list[str | None],
    outcome: str,
    *,
    latency_ms: float | None = None,
) -> None:
    """One grammar persist call (a single event or a trace batch) finished."""
    if not _enabled:
        return
    n = sum(1 for t in trace_ids if counts_grammar_event(t))
    if n:
        get_recorder().record(GRAMMAR_FAMILY, outcome, count=n, latency_ms=latency_ms)


def record_grammar_outcome(
    trace_id: str | None,
    counts: dict[str, int],
    *,
    latency_ms: float | None = None,
) -> None:
    """One grammar persist call (all events of one trace) with what the ledger
    handler says actually happened to them: {"committed": n, "duplicate": m,
    "timeout": k, ...}."""
    if not _enabled or not counts_grammar_event(trace_id):
        return
    recorder = get_recorder()
    for cls, n in counts.items():
        if int(n) > 0:
            recorder.record(GRAMMAR_FAMILY, cls, count=int(n), latency_ms=latency_ms)


# ---------------------------------------------------------------------------
# Per-envelope outcome (non-grammar path)
# ---------------------------------------------------------------------------


class EnvelopeOutcome:
    """Collects what happened to one envelope across the helpers it passes through.

    Shared by reference through a contextvar: ``asyncio.to_thread`` copies the
    context, so the worker thread sees the same object and its marks land here.

    The envelope's own table (``family``) decides. A write to another table
    (evidence units, a side log) is secondary: it only counts when the primary
    table was never written. Precedence:

        primary committed > any failure > primary duplicate
        > secondary committed > secondary duplicate > unrouted > skipped

    so an exception raised after the row committed (a post-commit publish that
    lands in the shared fallback handler) does not turn a landed row into a lost
    one, and a reject a helper reports before returning False (cockpit hop,
    harness trace) is not read as an idempotent duplicate. Marks after
    :meth:`close` are ignored (a background task spawned from the handler
    inherits the context and may finish later)."""

    __slots__ = ("family", "failure", "primary", "secondary", "unrouted", "closed", "_lock")

    def __init__(self, family: str) -> None:
        self.family = family
        self.failure: str | None = None
        self.primary: str | None = None  # committed | duplicate
        self.secondary: str | None = None
        self.unrouted = False
        self.closed = False
        self._lock = threading.Lock()

    def mark_written(self, ok: bool, table: str | None = None) -> None:
        result = OUTCOME_COMMITTED if ok else OUTCOME_DUPLICATE
        with self._lock:
            if self.closed:
                return
            # An unrouted kind can still be written somewhere (the evidence-unit
            # adapter path): report it under the table it actually reached.
            if table and self.family == OUTCOME_UNROUTED:
                self.family = table
            if table is None or table == self.family:
                if self.primary != OUTCOME_COMMITTED:
                    self.primary = result
            elif self.secondary != OUTCOME_COMMITTED:
                self.secondary = result

    def mark_failed(self, error: BaseException | str | None) -> None:
        cls = classify_write_error(error)
        with self._lock:
            if self.closed:
                return
            if cls == OUTCOME_UNROUTED:
                self.unrouted = True
            elif self.failure is None:
                self.failure = cls

    def close(self) -> str:
        with self._lock:
            self.closed = True
            if self.primary == OUTCOME_COMMITTED:
                return OUTCOME_COMMITTED
            if self.failure is not None:
                return self.failure
            if self.primary is not None:
                return self.primary
            if self.secondary is not None:
                return self.secondary
            if self.unrouted:
                return OUTCOME_UNROUTED
            return OUTCOME_SKIPPED


_CURRENT: contextvars.ContextVar[EnvelopeOutcome | None] = contextvars.ContextVar(
    "sql_writer_write_outcome", default=None
)


def begin_envelope(family: str) -> tuple[EnvelopeOutcome | None, Any]:
    if not _enabled:
        return None, None
    outcome = EnvelopeOutcome(family)
    return outcome, _CURRENT.set(outcome)


def end_envelope(
    outcome: EnvelopeOutcome | None,
    token: Any,
    *,
    started: float,
    error: BaseException | None = None,
) -> None:
    if outcome is None:
        return
    try:
        if error is not None:
            outcome.mark_failed(error)
        cls = outcome.close()
        latency_ms = (time.perf_counter() - started) * 1000.0
        get_recorder().record(outcome.family, cls, latency_ms=latency_ms)
    finally:
        if token is not None:
            try:
                _CURRENT.reset(token)
            except ValueError:
                _CURRENT.set(None)


def mark_written(ok: bool, table: str | None = None) -> None:
    outcome = _CURRENT.get()
    if outcome is not None:
        outcome.mark_written(ok, table)


def mark_failed(error: BaseException | str | None) -> None:
    outcome = _CURRENT.get()
    if outcome is not None:
        outcome.mark_failed(error)


# ---------------------------------------------------------------------------
# Window -> grammar events
# ---------------------------------------------------------------------------


def _window_id(start_ts: float) -> str:
    return datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _node_token(node: str | None) -> str:
    raw = (node or "").strip().lower().replace(":", "_")
    return "".join(ch if (ch.isalnum() or ch in "_.-") else "_" for ch in raw) or "writer"


def build_window_events(
    *,
    writer_node: str,
    window_start: float,
    window_end: float,
    buckets: dict[str, _FamilyBucket],
    grammar_queue_max: int = 0,
) -> list[GrammarEventV1]:
    node = _node_token(writer_node)
    trace_id = f"{STORAGE_WRITE_TRACE_PREFIX}{node}:{_window_id(window_start)}"
    emitted_at = datetime.fromtimestamp(window_end, tz=timezone.utc)
    dims = ["storage", "persistence"]
    provenance = GrammarProvenanceV1(
        source_service=SOURCE_SERVICE,
        source_component="write_health_window",
        source_trace_id=trace_id,
    )

    def _event(idx: int, role: str, summary: str, text_value: str) -> GrammarEventV1:
        event_id = f"{trace_id}:{idx:02d}:{role}"
        return GrammarEventV1(
            event_id=event_id,
            event_kind="atom_emitted",
            trace_id=trace_id,
            emitted_at=emitted_at,
            observed_at=emitted_at,
            layer="storage",
            dimensions=dims,
            atom=GrammarAtomV1(
                atom_id=event_id,
                trace_id=trace_id,
                atom_type="observation",
                semantic_role=role,
                layer="storage",
                dimensions=dims,
                summary=summary,
                text_value=text_value,
                confidence=1.0,
                salience=0.3,
            ),
            provenance=provenance,
        )

    events = [
        _event(i, ROLE_FAMILY_WINDOW, bucket.summary(family), family)
        for i, (family, bucket) in enumerate(sorted(buckets.items()))
    ]
    attempted = sum(b.attempted for b in buckets.values())
    failed = sum(b.failed for b in buckets.values())
    committed = sum(b.classes.get(OUTCOME_COMMITTED, 0) for b in buckets.values())
    unrouted = sum(b.classes.get(OUTCOME_UNROUTED, 0) for b in buckets.values())
    events.append(
        _event(
            len(events),
            ROLE_WINDOW_COMPLETED,
            (
                f"writer={node} attempted={attempted} committed={committed} failed={failed} "
                f"unrouted={unrouted} families={len(buckets)} grammar_queue_max={int(grammar_queue_max)} "
                f"window_sec={max(0.0, window_end - window_start):.1f}"
            ),
            node,
        )
    )
    return events


async def flush_window(bus: Any, *, writer_node: str, recorder: WriteHealthRecorder | None = None) -> int:
    """Drain the current window and publish it. Returns the number of events sent.
    A publish failure drops the rest of that window (logged): this report must
    never back up into the write path it reports on."""
    from orion.grammar.publish import publish_grammar_event

    recorder = recorder or get_recorder()
    start, end, buckets, queue_max = recorder.drain()
    events = build_window_events(
        writer_node=writer_node,
        window_start=start,
        window_end=end,
        buckets=buckets,
        grammar_queue_max=queue_max,
    )
    sent = 0
    try:
        if bus is None:
            raise RuntimeError("bus not available")
        for event in events:
            await publish_grammar_event(bus, event, source_name=SOURCE_SERVICE)
            sent += 1
    except Exception:  # noqa: BLE001
        logger.warning(
            "storage_write_health_publish_failed window_start=%s dropped=%d of %d",
            start,
            len(events) - sent,
            len(events),
            exc_info=True,
        )
    return sent


async def _wait_window(
    stop: asyncio.Event,
    interval: float,
    recorder: WriteHealthRecorder,
    queue_depth: Callable[[], int] | None,
) -> None:
    """Sleep one window, sampling the grammar queue depth once a second so the
    window carries its high-water mark, not just its depth at flush time."""
    deadline = time.monotonic() + interval
    while not stop.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        if queue_depth is not None:
            try:
                recorder.note_grammar_queue_depth(int(queue_depth()))
            except Exception:  # noqa: BLE001
                pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=min(1.0, remaining))
        except asyncio.TimeoutError:
            pass


async def run_window_publisher(
    bus_getter: Callable[[], Any],
    *,
    writer_node: str,
    window_sec: float,
    queue_depth: Callable[[], int] | None = None,
    stop: asyncio.Event | None = None,
) -> None:
    """Flush one window every ``window_sec`` until ``stop`` is set."""
    recorder = get_recorder()
    stop = stop or asyncio.Event()
    interval = max(5.0, float(window_sec))
    while not stop.is_set():
        await _wait_window(stop, interval, recorder, queue_depth)
        try:
            try:
                bus = bus_getter()
            except Exception:  # noqa: BLE001
                bus = None
            await flush_window(bus, writer_node=writer_node, recorder=recorder)
        except Exception:  # noqa: BLE001
            # Never let one bad window (an event that fails its own schema, a
            # drain bug) end the organ for the life of the process: downstream
            # would read "unmeasured" forever with nothing in the logs but this.
            logger.exception("storage_write_health_window_failed; continuing")
