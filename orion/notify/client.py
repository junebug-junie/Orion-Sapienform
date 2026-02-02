from __future__ import annotations

import base64
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Iterable, List, Optional
from uuid import UUID

import requests

from orion.schemas.notify import (
    ChatAttentionAck,
    ChatAttentionRequest,
    ChatMessageNotification,
    ChatMessageReceipt,
    NotificationAccepted,
    NotificationAttachment,
    NotificationRequest,
)

logger = logging.getLogger("orion.notify.client")


class NotifyClient:
    def __init__(self, base_url: str, api_token: Optional[str] = None, timeout: int = 10) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.timeout = timeout

    def send(self, request: NotificationRequest) -> NotificationAccepted:
        url = f"{self.base_url}/notify"
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["X-Orion-Notify-Token"] = self.api_token

        try:
            response = requests.post(url, json=request.model_dump(mode="json"), headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return NotificationAccepted(**response.json())
        except Exception as exc:
            logger.error("[NOTIFY] Failed to send notify request to %s: %s", url, exc)
            return NotificationAccepted(ok=False, detail=str(exc))

    def attention_request(
        self,
        message: str,
        severity: str = "info",
        require_ack: bool = True,
        context: Optional[dict] = None,
        expires_in_minutes: Optional[int] = None,
    ) -> NotificationAccepted:
        url = f"{self.base_url}/attention/request"
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["X-Orion-Notify-Token"] = self.api_token
        payload_context = context or {}
        source_service = payload_context.get("source_service", "unknown")
        reason = payload_context.get("reason", "attention_request")
        expires_at = None
        if expires_in_minutes is not None:
            expires_at = datetime.utcnow() + timedelta(minutes=expires_in_minutes)
        request = ChatAttentionRequest(
            source_service=source_service,
            reason=reason,
            severity=severity,
            message=message,
            context=payload_context,
            expires_at=expires_at,
            require_ack=require_ack,
        )
        try:
            response = requests.post(url, json=request.model_dump(mode="json"), headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            notification_id = data.get("notification_id")
            return NotificationAccepted(ok=True, notification_id=UUID(notification_id) if notification_id else None)
        except Exception as exc:
            logger.error("[NOTIFY] Failed to send attention request to %s: %s", url, exc)
            return NotificationAccepted(ok=False, detail=str(exc))

    def attention_ack(
        self,
        attention_id: UUID,
        ack_type: str = "seen",
        note: Optional[str] = None,
    ) -> NotificationAccepted:
        url = f"{self.base_url}/attention/{attention_id}/ack"
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["X-Orion-Notify-Token"] = self.api_token
        payload = ChatAttentionAck(attention_id=attention_id, ack_type=ack_type, note=note)
        try:
            response = requests.post(url, json=payload.model_dump(mode="json"), headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            notification_id = data.get("notification_id")
            return NotificationAccepted(ok=True, notification_id=UUID(notification_id) if notification_id else None)
        except Exception as exc:
            logger.error("[NOTIFY] Failed to send attention ack to %s: %s", url, exc)
            return NotificationAccepted(ok=False, detail=str(exc))

    def chat_message(
        self,
        session_id: str,
        preview_text: str,
        full_text: Optional[str] = None,
        severity: str = "info",
        require_read_receipt: bool = True,
        tags: Optional[List[str]] = None,
        source_service: str = "unknown",
    ) -> NotificationAccepted:
        url = f"{self.base_url}/chat/message"
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["X-Orion-Notify-Token"] = self.api_token
        payload = ChatMessageNotification(
            source_service=source_service,
            session_id=session_id,
            preview_text=preview_text,
            full_text=full_text,
            severity=severity,
            require_read_receipt=require_read_receipt,
            tags=tags or ["chat", "message"],
        )
        try:
            response = requests.post(url, json=payload.model_dump(mode="json"), headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            notification_id = data.get("notification_id")
            return NotificationAccepted(ok=True, notification_id=UUID(notification_id) if notification_id else None)
        except Exception as exc:
            logger.error("[NOTIFY] Failed to send chat message to %s: %s", url, exc)
            return NotificationAccepted(ok=False, detail=str(exc))

    def chat_message_receipt(
        self,
        message_id: UUID,
        session_id: str,
        receipt_type: str,
    ) -> NotificationAccepted:
        url = f"{self.base_url}/chat/message/{message_id}/receipt"
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["X-Orion-Notify-Token"] = self.api_token
        payload = ChatMessageReceipt(message_id=message_id, session_id=session_id, receipt_type=receipt_type)
        try:
            response = requests.post(url, json=payload.model_dump(mode="json"), headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            notification_id = data.get("notification_id")
            return NotificationAccepted(ok=True, notification_id=UUID(notification_id) if notification_id else None)
        except Exception as exc:
            logger.error("[NOTIFY] Failed to send chat message receipt to %s: %s", url, exc)
            return NotificationAccepted(ok=False, detail=str(exc))

    @staticmethod
    def attachments_from_paths(
        paths: Iterable[str | Path],
        mime_type: str = "image/jpeg",
    ) -> List[NotificationAttachment]:
        attachments: List[NotificationAttachment] = []
        for path in paths:
            path_obj = Path(path)
            try:
                data = path_obj.read_bytes()
            except Exception as exc:
                logger.error("[NOTIFY] Failed to read attachment %s: %s", path_obj, exc)
                continue

            attachments.append(
                NotificationAttachment(
                    filename=path_obj.name,
                    content_base64=base64.b64encode(data).decode("utf-8"),
                    mime_type=mime_type,
                )
            )
        return attachments
