from __future__ import annotations

from uuid import uuid4

from orion.notify.client import NotifyClient

from .settings import Settings


class AttentionPublisher:
    def __init__(self, settings: Settings) -> None:
        self._client = NotifyClient(
            base_url=settings.notify_base_url,
            api_token=settings.notify_api_token,
            timeout=10,
        )
        self._source = settings.service_name

    def publish_transition(
        self,
        *,
        service_id: str,
        heartbeat_name: str,
        event: dict,
    ) -> None:
        severity = event.get("severity", "error")
        message = event.get("message", f"mesh health: {service_id}")
        event_ctx = event.get("context") or {}
        mesh_event = event_ctx.get("event", "attention")
        reason = f"[Orion mesh] {service_id} — {mesh_event}"
        body = "\n".join(
            [
                message,
                "",
                f"service: {service_id}",
                f"heartbeat: {heartbeat_name}",
                f"event: {mesh_event}",
            ]
        )
        context = {
            "source_service": self._source,
            "event_kind": "orion.mesh.health.attention.v1",
            "service_id": service_id,
            "heartbeat_name": heartbeat_name,
            "correlation_id": event.get("correlation_id") or str(uuid4()),
            "reason": reason,
            **event_ctx,
        }
        self._client.attention_request(
            message=body,
            severity=severity,
            require_ack=True,
            context=context,
        )
