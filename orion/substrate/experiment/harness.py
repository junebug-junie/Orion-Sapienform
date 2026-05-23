"""SubstrateExperimentHarness — records substrate activity for daily rollups.

Designed to be sprinkled into emit/traversal sites with minimal coupling. The
harness is fully in-memory; persistence happens via the daily_rollup module.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from threading import RLock
from typing import Iterable

from orion.substrate.molecules import SubstrateMoleculeV1
from orion.substrate.operators import GradientDelta


def _today() -> date:
    return datetime.now(timezone.utc).date()


@dataclass
class _EmitRecord:
    molecule_id: str
    molecule_kind: str
    organ: str
    when: datetime
    atoms_snapshot: dict[str, str]


@dataclass
class _TraversalRecord:
    when: datetime
    requesting_organ: str
    query_gradients: tuple[str, ...]
    threshold: float
    hit_ids: tuple[str, ...]


@dataclass
class _GradientRecord:
    molecule_id: str
    when: datetime
    cause: str
    before: dict[str, float]
    after: dict[str, float]


@dataclass
class _DayBucket:
    emits: list[_EmitRecord] = field(default_factory=list)
    traversals: list[_TraversalRecord] = field(default_factory=list)
    gradient_changes: list[_GradientRecord] = field(default_factory=list)
    referenced_ids: set[str] = field(default_factory=set)
    references_by_organ: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))


class SubstrateExperimentHarness:
    """In-memory recorder for substrate emit/traversal/gradient events.

    A single harness can span many days; metrics are bucketed by UTC date.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._buckets: dict[date, _DayBucket] = defaultdict(_DayBucket)
        self._emits_by_id: dict[str, _EmitRecord] = {}

    # -- bucket helpers --------------------------------------------------------

    def _bucket(self, when: datetime) -> _DayBucket:
        return self._buckets[when.date()]

    # -- public recording API --------------------------------------------------

    def record_emit(
        self,
        molecule: SubstrateMoleculeV1,
        organ: str,
        *,
        when: datetime | None = None,
    ) -> None:
        when = when or datetime.now(timezone.utc)
        record = _EmitRecord(
            molecule_id=molecule.molecule_id,
            molecule_kind=molecule.molecule_kind,
            organ=organ,
            when=when,
            atoms_snapshot=dict(molecule.atoms),
        )
        with self._lock:
            self._bucket(when).emits.append(record)
            self._emits_by_id[molecule.molecule_id] = record

    def record_traversal(
        self,
        *,
        query_gradients: Iterable[str],
        threshold: float,
        results: Iterable[SubstrateMoleculeV1],
        requesting_organ: str,
        when: datetime | None = None,
    ) -> None:
        when = when or datetime.now(timezone.utc)
        hit_ids = tuple(m.molecule_id for m in results)
        record = _TraversalRecord(
            when=when,
            requesting_organ=requesting_organ,
            query_gradients=tuple(query_gradients),
            threshold=threshold,
            hit_ids=hit_ids,
        )
        with self._lock:
            bucket = self._bucket(when)
            bucket.traversals.append(record)
            for hit_id in hit_ids:
                emit = self._emits_by_id.get(hit_id)
                bucket.referenced_ids.add(hit_id)
                if emit is not None and emit.organ != requesting_organ:
                    bucket.references_by_organ[requesting_organ].add(hit_id)

    def record_gradient_delta(
        self,
        delta: GradientDelta,
        *,
        when: datetime | None = None,
    ) -> None:
        when = when or datetime.now(timezone.utc)
        record = _GradientRecord(
            molecule_id=delta.molecule_id,
            when=when,
            cause=delta.cause,
            before=dict(delta.before),
            after=dict(delta.after),
        )
        with self._lock:
            self._bucket(when).gradient_changes.append(record)
            if delta.cause not in {"decay",}:
                # reinforcement/contradiction/stabilize all count as "touched"
                self._bucket(when).referenced_ids.add(delta.molecule_id)

    # -- convenience wrappers --------------------------------------------------

    def record_reinforcement(self, delta: GradientDelta) -> None:
        self.record_gradient_delta(delta)

    def record_decay(self, delta: GradientDelta) -> None:
        self.record_gradient_delta(delta)

    # -- read API for rollups --------------------------------------------------

    def days_recorded(self) -> tuple[date, ...]:
        with self._lock:
            return tuple(sorted(self._buckets.keys()))

    def bucket_for(self, day: date) -> _DayBucket | None:
        with self._lock:
            return self._buckets.get(day)

    def all_emit_records(self) -> list[_EmitRecord]:
        with self._lock:
            return list(self._emits_by_id.values())
