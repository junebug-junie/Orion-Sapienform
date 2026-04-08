from __future__ import annotations

from datetime import datetime, timezone

from orion.core.schemas.endogenous import EndogenousHistoryEntryV1, EndogenousWorkflowTypeV1


class InMemoryTriggerHistoryStore:
    """Deterministic cooldown/debounce history for endogenous workflow triggers."""

    def __init__(self) -> None:
        self._entries: list[EndogenousHistoryEntryV1] = []

    def record(self, entry: EndogenousHistoryEntryV1) -> None:
        self._entries.append(entry)

    def recent(self, *, subject_ref: str | None, limit: int = 25) -> list[EndogenousHistoryEntryV1]:
        if not subject_ref:
            return list(reversed(self._entries[-max(0, limit) :]))
        filtered = [item for item in self._entries if item.subject_ref == subject_ref]
        return list(reversed(filtered[-max(0, limit) :]))

    def cooldown_remaining_seconds(
        self,
        *,
        workflow_type: EndogenousWorkflowTypeV1,
        subject_ref: str | None,
        now: datetime,
        cooldown_seconds: int,
    ) -> int:
        now = self._ensure_tz(now)
        last = self._latest(workflow_type=workflow_type, subject_ref=subject_ref)
        if last is None:
            return 0
        elapsed = int((now - last.recorded_at).total_seconds())
        remaining = cooldown_seconds - elapsed
        return max(0, remaining)

    def has_recent_signature(
        self,
        *,
        cause_signature: str,
        subject_ref: str | None,
        now: datetime,
        within_seconds: int,
    ) -> bool:
        now = self._ensure_tz(now)
        for item in reversed(self._entries):
            if item.cause_signature != cause_signature:
                continue
            if subject_ref and item.subject_ref != subject_ref:
                continue
            if int((now - item.recorded_at).total_seconds()) <= within_seconds:
                return True
        return False

    def _latest(
        self,
        *,
        workflow_type: EndogenousWorkflowTypeV1,
        subject_ref: str | None,
    ) -> EndogenousHistoryEntryV1 | None:
        for item in reversed(self._entries):
            if item.workflow_type != workflow_type:
                continue
            if subject_ref and item.subject_ref != subject_ref:
                continue
            return item
        return None

    @staticmethod
    def _ensure_tz(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
