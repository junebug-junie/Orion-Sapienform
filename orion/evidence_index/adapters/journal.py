from __future__ import annotations

from typing import Any

from orion.journaler import JournalEntryWriteV1
from orion.schemas.evidence_index import EvidenceUnitV1


class JournalEvidenceAdapter:
    source_family = "journal"

    def to_units(self, payload: Any, *, kind: str, correlation_id: str | None = None) -> list[EvidenceUnitV1]:
        container = payload if isinstance(payload, dict) else {}
        inner = container.get("payload") if isinstance(container.get("payload"), dict) else container
        write = JournalEntryWriteV1.model_validate(inner)

        unit = EvidenceUnitV1(
            unit_id=write.entry_id,
            unit_kind="journal_entry",
            source_family=self.source_family,
            source_kind=write.source_kind or "journal",
            source_ref=write.source_ref or write.entry_id,
            correlation_id=write.correlation_id or correlation_id,
            title=write.title,
            summary=write.title,
            body=write.body,
            facets=[f"mode:{write.mode}", f"author:{write.author}"],
            metadata={"mode": write.mode, "author": write.author},
            created_at=write.created_at,
        )
        return [unit]
