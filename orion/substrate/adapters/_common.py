from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from orion.core.schemas.cognitive_substrate import SubstrateProvenanceV1, SubstrateTemporalWindowV1


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
