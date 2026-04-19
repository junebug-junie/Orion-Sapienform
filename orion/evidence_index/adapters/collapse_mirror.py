from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from orion.schemas.evidence_index import EvidenceUnitV1


def _as_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        raw = value.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return datetime.now(timezone.utc)
    return datetime.now(timezone.utc)


class CollapseMirrorEvidenceAdapter:
    source_family = "collapse_mirror"

    def to_units(self, payload: Any, *, kind: str, correlation_id: str | None = None) -> list[EvidenceUnitV1]:
        data = payload if isinstance(payload, dict) else {}
        unit_id = str(data.get("id") or data.get("event_id") or data.get("correlation_id") or "")
        if not unit_id:
            return []

        source_ref = str(data.get("source_service") or data.get("observer") or "collapse")
        summary = data.get("summary")
        body = data.get("what_changed_summary") or data.get("mantra") or data.get("trigger")
        created_at = _as_dt(data.get("timestamp"))
        collapse_type = str(data.get("type") or "collapse")

        unit = EvidenceUnitV1(
            unit_id=unit_id,
            unit_kind="collapse_mirror_entry",
            source_family=self.source_family,
            source_kind=collapse_type,
            source_ref=source_ref,
            correlation_id=str(data.get("correlation_id") or correlation_id or "") or None,
            title=data.get("emergent_entity"),
            summary=summary,
            body=body,
            facets=[f"type:{collapse_type}"],
            metadata={"observer": data.get("observer"), "pattern_candidate": data.get("pattern_candidate")},
            created_at=created_at,
        )
        return [unit]
