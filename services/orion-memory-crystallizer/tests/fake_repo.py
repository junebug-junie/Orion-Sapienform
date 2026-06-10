"""In-memory fake of CrystallizationRepository for HTTP/worker tests."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from orion.memory.crystallization.governor import GovernanceHistoryEntry
from orion.schemas.memory_crystallization import (
    ActiveMemoryPacketV1,
    MemoryCrystallizationV1,
)


class FakeRepository:
    def __init__(self) -> None:
        self.rows: dict[str, MemoryCrystallizationV1] = {}
        self.history: list[GovernanceHistoryEntry] = []
        self.retrieval_events: dict[str, dict[str, Any]] = {}

    # writes

    def upsert(
        self,
        crystallization: MemoryCrystallizationV1,
        *,
        expected_statuses: Iterable[str] | None = None,
    ) -> bool:
        existing = self.rows.get(crystallization.crystallization_id)
        if (
            expected_statuses is not None
            and existing is not None
            and existing.status not in set(expected_statuses)
        ):
            return False
        self.rows[crystallization.crystallization_id] = crystallization
        return True

    def apply_transition(
        self,
        crystallization: MemoryCrystallizationV1,
        entry: GovernanceHistoryEntry,
        *,
        before: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
        expected_statuses: Iterable[str] | None = None,
    ) -> bool:
        if not self.upsert(crystallization, expected_statuses=expected_statuses):
            return False
        self.history.append(entry)
        return True

    def apply_supersession(
        self,
        superseded_old: MemoryCrystallizationV1,
        updated_new: MemoryCrystallizationV1,
        entries: list[GovernanceHistoryEntry],
        *,
        old_expected_statuses: Iterable[str] = ("active", "deprecated"),
    ) -> bool:
        if not self.upsert(superseded_old, expected_statuses=old_expected_statuses):
            return False
        self.upsert(updated_new)
        self.history.extend(entries)
        return True

    def merge_projection_refs(self, crystallization_id: str, ref_updates: dict[str, list[str]]) -> bool:
        crystallization = self.rows.get(crystallization_id)
        if crystallization is None:
            return False
        refs = crystallization.projection_refs
        updates: dict[str, Any] = {"synced_at": datetime.now(timezone.utc)}
        for field, ids in ref_updates.items():
            updates[field] = sorted(set(getattr(refs, field)) | set(ids))
        self.rows[crystallization_id] = crystallization.model_copy(
            update={"projection_refs": refs.model_copy(update=updates)}
        )
        return True

    def record_history(self, entry: GovernanceHistoryEntry, *, before=None, after=None) -> None:
        self.history.append(entry)

    def record_retrieval_event(self, retrieval_event_id: str, packet: ActiveMemoryPacketV1) -> None:
        self.retrieval_events[retrieval_event_id] = {
            "retrieval_event_id": retrieval_event_id,
            "query": packet.query,
            "packet": packet.model_dump(mode="json"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    # reads

    def get(self, crystallization_id: str) -> MemoryCrystallizationV1 | None:
        return self.rows.get(crystallization_id)

    def list(
        self,
        *,
        status: str | None = None,
        kind: str | None = None,
        scope: str | None = None,
        limit: int = 100,
    ) -> list[MemoryCrystallizationV1]:
        out = []
        for row in self.rows.values():
            if status and row.status != status:
                continue
            if kind and row.kind != kind:
                continue
            if scope and scope not in row.scope:
                continue
            out.append(row)
        out.sort(key=lambda c: c.salience, reverse=True)
        return out[:limit]

    def list_links(self, crystallization_id: str) -> list[dict[str, Any]]:
        out = []
        for row in self.rows.values():
            for link in row.links:
                if (
                    row.crystallization_id == crystallization_id
                    or link.target_crystallization_id == crystallization_id
                ):
                    out.append(
                        {
                            "from_crystallization_id": row.crystallization_id,
                            "to_crystallization_id": link.target_crystallization_id,
                            "relation": link.relation,
                            "confidence": link.confidence,
                            "note": link.note,
                        }
                    )
        return out

    def list_history(self, crystallization_id: str, limit: int = 100) -> list[dict[str, Any]]:
        return [
            {"op": e.op, "actor": e.actor, "reason": e.reason}
            for e in self.history
            if e.crystallization_id == crystallization_id
        ][:limit]

    def get_retrieval_event(self, retrieval_event_id: str) -> dict[str, Any] | None:
        return self.retrieval_events.get(retrieval_event_id)
