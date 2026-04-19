from __future__ import annotations

from typing import Any, Protocol

from orion.schemas.evidence_index import EvidenceUnitV1


class EvidenceAdapter(Protocol):
    source_family: str

    def to_units(self, payload: Any, *, kind: str, correlation_id: str | None = None) -> list[EvidenceUnitV1]:
        ...
