"""Heartbeat's H1 verdict as a bounded trace on orion:grammar:event.

Heartbeat computes an H1 verdict (redundant / concentrated / mixed) every 30 s
and, before this module, told no one except whoever polled ``/h1``. This
module turns that stream into two kinds of grammar atom:

* ``h1_verdict_transition`` -- RETIRED 2026-11-24 (#3838). The 2026-11-08
  memory-horizon replay (docs/superpowers/specs/2026-11-08-prediction-error-
  suite-rebuild-design.md, "Heartbeat: what it measures today") found the
  verdict class no better than shuffled input (real vs shuffled agreement
  67-68%, seed-vs-seed 69-70%). Publishing settled class changes at
  confidence 1.0 presented noise as a finding, so the atom, its debouncer and
  its per-hour cap are gone.
* ``h1_hourly_summary`` -- once per summary window, also when every H1
  computation in it failed (``h1_ticks=0 h1_failures=N``, confidence 0 -- an
  absence, not calm). If the H1 loop task itself dies, summaries stop: that
  reads as missing ``heartbeat.h1:`` rows in grammar_events.
  per-class tick counts, raw flips, mean/min/max of
  std_ratio (ensemble spread), mean_ratio and bulk depth, and per-producer
  counts of grammar atoms heartbeat could not route.

What it does NOT do: nothing here writes a field channel, substrate node or
prior. No reducer reads ``heartbeat.h1:`` traces; they land in the
grammar_events ledger (sql-writer) as an observable record. See the service
README "What heartbeat emits" for why no field consumer was wired.

Pure, no I/O: the service owns the clock and the bus.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1, TimeRangeV1

# Module-level literal so tests/test_grammar_event_producer_catalog.py's static
# scan can resolve this file's GrammarProvenanceV1 identity.
SOURCE_SERVICE = "orion-heartbeat"
TRACE_PREFIX = "heartbeat.h1:"
ROLE_HOURLY_SUMMARY = "h1_hourly_summary"
VERDICTS = ("concentrated", "mixed", "redundant")

# Labels carried on every summary atom (and on /h1), from the 2026-11-08
# replay: the class is noise; mean_ratio is reproducible (r=+0.91 across
# seeds) but remembers about the last 3 atoms (held-out R^2 0.88).
VERDICT_MEANING = "noise_no_better_than_shuffled_input_2026-11-08"
MEAN_RATIO_MEANING = "last_3_events_sequence_meter_2026-11-08"

_MAX_UNROUTED_KEYS = 32
_SAFE_RE = re.compile(r"[^a-z0-9_.-]")


def safe_token(value: Any, default: str = "unknown") -> str:
    raw = _SAFE_RE.sub("", str(value or "").strip().lower())
    return raw[:64] or default


def _stamp(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class H1Tick:
    """The four H1 numbers this module needs, decoupled from EnsembleH1ResultV1."""

    at: datetime
    verdict: str
    mean_ratio: float
    std_ratio: float
    bulk_penetration_depth: float


@dataclass
class HourlyWindow:
    """Accumulates one summary window. ``drain`` returns its kv summary and resets."""

    start: datetime
    verdict_ticks: dict[str, int] = field(default_factory=lambda: {v: 0 for v in VERDICTS})
    raw_flips: int = 0
    h1_failures: int = 0
    unrouted_by_source: dict[str, int] = field(default_factory=dict)
    _last_verdict: Optional[str] = None
    _std: list[float] = field(default_factory=list)
    _mean: list[float] = field(default_factory=list)
    _bulk: list[float] = field(default_factory=list)

    def record_tick(self, tick: H1Tick) -> None:
        self.verdict_ticks[tick.verdict] = self.verdict_ticks.get(tick.verdict, 0) + 1
        if self._last_verdict is not None and tick.verdict != self._last_verdict:
            self.raw_flips += 1
        self._last_verdict = tick.verdict
        self._std.append(tick.std_ratio)
        self._mean.append(tick.mean_ratio)
        self._bulk.append(tick.bulk_penetration_depth)

    def record_unrouted(self, source_service: str) -> None:
        key = safe_token(source_service)
        if key not in self.unrouted_by_source and len(self.unrouted_by_source) >= _MAX_UNROUTED_KEYS:
            key = "other"
        self.unrouted_by_source[key] = self.unrouted_by_source.get(key, 0) + 1

    @property
    def h1_ticks(self) -> int:
        return len(self._std)

    def summary(self, end: datetime) -> str:
        def stat(name: str, xs: list[float]) -> str:
            if not xs:
                return f"{name}_mean=none {name}_min=none {name}_max=none"
            return (
                f"{name}_mean={sum(xs) / len(xs):.4f} "
                f"{name}_min={min(xs):.4f} {name}_max={max(xs):.4f}"
            )

        ticks = "|".join(f"{v}:{self.verdict_ticks.get(v, 0)}" for v in VERDICTS)
        unrouted = "|".join(f"{k}:{n}" for k, n in sorted(self.unrouted_by_source.items())) or "none"
        return (
            f"window_start={self.start.isoformat()} window_end={end.isoformat()} "
            f"h1_ticks={self.h1_ticks} h1_failures={self.h1_failures} "
            f"verdict_meaning={VERDICT_MEANING} mean_ratio_meaning={MEAN_RATIO_MEANING} "
            f"verdict_ticks={ticks} raw_flips={self.raw_flips} "
            f"{stat('std_ratio', self._std)} {stat('mean_ratio', self._mean)} "
            f"{stat('bulk', self._bulk)} "
            f"unrouted_atoms={unrouted}"
        )


def _event(
    *,
    trace_id: str,
    role: str,
    summary: str,
    emitted_at: datetime,
    time_range: TimeRangeV1,
    confidence: float,
) -> GrammarEventV1:
    event_id = f"{trace_id}:{role}"
    dims = ["heartbeat", "h1"]
    return GrammarEventV1(
        event_id=event_id,
        event_kind="atom_emitted",
        trace_id=trace_id,
        emitted_at=emitted_at,
        observed_at=emitted_at,
        layer="substrate",
        dimensions=dims,
        atom=GrammarAtomV1(
            atom_id=event_id,
            trace_id=trace_id,
            atom_type="observation",
            semantic_role=role,
            layer="substrate",
            dimensions=dims,
            summary=summary,
            text_value=summary,
            confidence=confidence,
            salience=0.2,
            time_range=time_range,
        ),
        provenance=GrammarProvenanceV1(
            source_service=SOURCE_SERVICE,
            source_component="h1_verdict",
            source_trace_id=trace_id,
        ),
    )


def build_summary_event(*, node: str, window: HourlyWindow, end: datetime) -> GrammarEventV1:
    trace_id = f"{TRACE_PREFIX}{safe_token(node, 'node')}:summary:{_stamp(window.start)}"
    # confidence 0 when no H1 tick completed: the window is a report of absence.
    return _event(
        trace_id=trace_id,
        role=ROLE_HOURLY_SUMMARY,
        summary=window.summary(end),
        emitted_at=end,
        time_range=TimeRangeV1(start=window.start, end=end),
        confidence=1.0 if window.h1_ticks else 0.0,
    )
