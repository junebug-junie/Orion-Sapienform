"""Heartbeat's H1 verdict as a bounded trace on orion:grammar:event.

Heartbeat computes an H1 verdict (redundant / concentrated / mixed) every 30 s
and, before this module, told no one except whoever polled ``/h1``. This
module turns that stream into two kinds of grammar atom:

* ``h1_verdict_transition`` -- the verdict class changed AND the new class held
  for ``settle_ticks`` consecutive H1 ticks. Not every raw flip: replaying 7
  days of the verdict as AST/HOT recorded it (substrate_attention_self_model,
  16,642 samples, 2026-09-30..10-07) found 31.8 raw class changes per hour --
  ``redundant`` runs have median length 1 and never exceed 3 samples,
  ``concentrated`` median 1. A per-flip atom would be ~760/day of threshold
  noise. Requiring 3 settled ticks cut that to ~1.1/hour on the same history.
  A hard per-hour cap (``max_transitions_per_hour``) bounds it regardless.
* ``h1_hourly_summary`` -- once per summary window, always (also when no H1
  tick completed, so a dead H1 loop reads as ``h1_ticks=0``, not as calm):
  per-class tick counts, raw flips, settled transitions, mean/min/max of
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
ROLE_TRANSITION = "h1_verdict_transition"
ROLE_HOURLY_SUMMARY = "h1_hourly_summary"
VERDICTS = ("concentrated", "mixed", "redundant")

_MAX_UNROUTED_KEYS = 32
_SAFE_RE = re.compile(r"[^a-z0-9_.-]")


def _safe(value: Any, default: str = "unknown") -> str:
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


@dataclass(frozen=True)
class VerdictTransition:
    from_verdict: str  # "none" for the first settled verdict after a (re)start
    to_verdict: str
    held_ticks: int
    held_since: datetime
    confirmed_at: datetime
    mean_ratio: float
    std_ratio: float
    bulk_penetration_depth: float
    mean_std_ratio_while_held: float


class VerdictTransitionTracker:
    """Debounced class-change detector. ``observe`` returns a transition only
    when a class different from the last confirmed one has held for
    ``settle_ticks`` consecutive ticks."""

    def __init__(self, *, settle_ticks: int) -> None:
        if settle_ticks < 1:
            raise ValueError(f"settle_ticks must be >= 1, got {settle_ticks}")
        self.settle_ticks = settle_ticks
        self.confirmed: Optional[str] = None
        self._streak_verdict: Optional[str] = None
        self._streak: list[H1Tick] = []

    def observe(self, tick: H1Tick) -> Optional[VerdictTransition]:
        if tick.verdict != self._streak_verdict:
            self._streak_verdict = tick.verdict
            self._streak = []
        self._streak.append(tick)
        if len(self._streak) < self.settle_ticks or tick.verdict == self.confirmed:
            return None
        previous = self.confirmed
        self.confirmed = tick.verdict
        held = list(self._streak)
        return VerdictTransition(
            from_verdict=previous or "none",
            to_verdict=tick.verdict,
            held_ticks=len(held),
            held_since=held[0].at,
            confirmed_at=tick.at,
            mean_ratio=tick.mean_ratio,
            std_ratio=tick.std_ratio,
            bulk_penetration_depth=tick.bulk_penetration_depth,
            mean_std_ratio_while_held=sum(t.std_ratio for t in held) / len(held),
        )


@dataclass
class HourlyWindow:
    """Accumulates one summary window. ``drain`` returns its kv summary and resets."""

    start: datetime
    verdict_ticks: dict[str, int] = field(default_factory=lambda: {v: 0 for v in VERDICTS})
    raw_flips: int = 0
    transitions_emitted: int = 0
    transitions_suppressed: int = 0
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
        key = _safe(source_service)
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
            f"h1_ticks={self.h1_ticks} h1_failures={self.h1_failures} verdict_ticks={ticks} "
            f"raw_flips={self.raw_flips} transitions={self.transitions_emitted} "
            f"transitions_suppressed={self.transitions_suppressed} "
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


def build_transition_event(*, node: str, transition: VerdictTransition) -> GrammarEventV1:
    t = transition
    trace_id = f"{TRACE_PREFIX}{_safe(node, 'node')}:transition:{_stamp(t.confirmed_at)}"
    summary = (
        f"from={t.from_verdict} to={t.to_verdict} held_ticks={t.held_ticks} "
        f"held_since={t.held_since.isoformat()} confirmed_at={t.confirmed_at.isoformat()} "
        f"mean_ratio={t.mean_ratio:.4f} std_ratio={t.std_ratio:.4f} "
        f"mean_std_ratio_while_held={t.mean_std_ratio_while_held:.4f} "
        f"bulk_penetration_depth={t.bulk_penetration_depth:.4f}"
    )
    return _event(
        trace_id=trace_id,
        role=ROLE_TRANSITION,
        summary=summary,
        emitted_at=t.confirmed_at,
        time_range=TimeRangeV1(start=t.held_since, end=t.confirmed_at),
        confidence=1.0,
    )


def build_summary_event(*, node: str, window: HourlyWindow, end: datetime) -> GrammarEventV1:
    trace_id = f"{TRACE_PREFIX}{_safe(node, 'node')}:summary:{_stamp(window.start)}"
    # confidence 0 when no H1 tick completed: the window is a report of absence.
    return _event(
        trace_id=trace_id,
        role=ROLE_HOURLY_SUMMARY,
        summary=window.summary(end),
        emitted_at=end,
        time_range=TimeRangeV1(start=window.start, end=end),
        confidence=1.0 if window.h1_ticks else 0.0,
    )
