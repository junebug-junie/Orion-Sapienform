from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from orion.schemas.evidence_index import EvidenceUnitV1


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        raw = value.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


class NotifyOutputEvidenceAdapter:
    source_family = "notify_output"

    def to_units(self, payload: Any, *, kind: str, correlation_id: str | None = None) -> list[EvidenceUnitV1]:
        data = payload if isinstance(payload, dict) else {}
        container = data.get("payload") if isinstance(data.get("payload"), dict) else data

        if kind == "notify.notification.request.v1":
            notification_id = str(container.get("notification_id") or "")
            if not notification_id:
                return []
            title = container.get("title")
            body = container.get("body_text") or container.get("body_md")
            event_kind = str(container.get("event_kind") or "notify")
            severity = str(container.get("severity") or "info")
            created_at = _parse_dt(container.get("created_at"))
            return [
                EvidenceUnitV1(
                    unit_id=f"notify::{notification_id}",
                    unit_kind="notify_event",
                    source_family=self.source_family,
                    source_kind=event_kind,
                    source_ref=notification_id,
                    correlation_id=str(container.get("correlation_id") or correlation_id or "") or None,
                    title=title,
                    summary=title,
                    body=body,
                    facets=[f"severity:{severity}", "artifact:notify"],
                    metadata={
                        "source_service": container.get("source_service"),
                        "recipient_group": container.get("recipient_group"),
                    },
                    created_at=created_at,
                )
            ]

        if kind == "notify.notification.receipt.v1":
            receipt_id = str(container.get("receipt_id") or "")
            message_id = str(container.get("message_id") or "")
            if not receipt_id:
                return []
            receipt_type = str(container.get("receipt_type") or "receipt")
            created_at = _parse_dt(container.get("created_at") or container.get("received_at"))
            parent_id = f"notify::{message_id}" if message_id else None
            return [
                EvidenceUnitV1(
                    unit_id=f"notify-receipt::{receipt_id}",
                    unit_kind="notify_receipt",
                    source_family=self.source_family,
                    source_kind=receipt_type,
                    source_ref=receipt_id,
                    correlation_id=correlation_id,
                    parent_unit_id=parent_id,
                    title=f"Notify receipt {receipt_type}",
                    summary=f"Receipt recorded for message {message_id}" if message_id else "Receipt recorded",
                    body=None,
                    facets=[f"receipt_type:{receipt_type}", "artifact:notify"],
                    metadata={"session_id": container.get("session_id"), "message_id": message_id or None},
                    created_at=created_at,
                )
            ]

        return []
