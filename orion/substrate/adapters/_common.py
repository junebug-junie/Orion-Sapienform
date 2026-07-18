from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from orion.core.schemas.cognitive_substrate import (
    DEFAULT_CONCEPT_ACTIVATION_HALF_LIFE_SECONDS,
    SubstrateActivationV1,
    SubstrateProvenanceV1,
    SubstrateTemporalWindowV1,
)

__all__ = [
    "DEFAULT_CONCEPT_ACTIVATION_HALF_LIFE_SECONDS",
    "make_activation",
    "make_provenance",
    "make_temporal",
]


def make_activation(
    *,
    initial_activation: float,
    decay_half_life_seconds: int = DEFAULT_CONCEPT_ACTIVATION_HALF_LIFE_SECONDS,
) -> SubstrateActivationV1:
    """Explicitly seed a concept node's activation signal.

    Optional for callers: `ConceptNodeV1` itself now auto-seeds `activation`
    from `salience` and applies `DEFAULT_CONCEPT_ACTIVATION_HALF_LIFE_SECONDS`
    whenever a producer leaves `signals.activation` at its pure schema default
    (see `ConceptNodeV1._seed_activation_if_unset` in
    `orion/core/schemas/cognitive_substrate.py`) -- that fix applies to every
    producer, not just the ones that call this helper. Use this directly when
    a producer wants a specific initial value other than raw `salience`, or
    wants to be explicit about it at the call site.

    `recency_score` is deliberately left at its schema default (0.0) here --
    it is meant to be derived from `temporal.observed_at` at read/tick time
    (see `orion/substrate/dynamics.py::_compute_activations`), not fabricated
    at construction time from an unrelated salience/confidence value.
    """
    return SubstrateActivationV1(
        activation=min(1.0, max(0.0, initial_activation)),
        decay_half_life_seconds=decay_half_life_seconds,
    )


def _ensure_tz(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def make_temporal(*, observed_at: datetime | None, valid_from: datetime | None = None, valid_to: datetime | None = None) -> SubstrateTemporalWindowV1:
    return SubstrateTemporalWindowV1(
        observed_at=_ensure_tz(observed_at),
        valid_from=_ensure_tz(valid_from) if valid_from else None,
        valid_to=_ensure_tz(valid_to) if valid_to else None,
    )


def make_provenance(
    *,
    source_kind: str,
    source_channel: str,
    producer: str,
    correlation_id: str | None = None,
    trace_id: str | None = None,
    evidence_refs: Optional[list[str]] = None,
) -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind=source_kind,
        source_channel=source_channel,
        producer=producer,
        correlation_id=correlation_id,
        trace_id=trace_id,
        evidence_refs=list(evidence_refs or []),
    )
