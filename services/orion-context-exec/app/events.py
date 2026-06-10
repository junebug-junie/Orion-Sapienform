from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.context_exec import ContextExecMode, ContextExecVerbStepV1

from .settings import settings

logger = logging.getLogger("orion-context-exec.events")


def _source() -> ServiceRef:
    return ServiceRef(
        name=settings.service_name,
        node=settings.node_name,
        version=settings.service_version,
    )


class ContextExecEventEmitter:
    """Publishes lifecycle events to orion:context_exec:event."""

    def __init__(self, bus: OrionBusAsync | None) -> None:
        self._bus = bus

    async def publish(self, kind: str, *, run_id: str, mode: ContextExecMode, payload: dict[str, Any]) -> None:
        if self._bus is None or not settings.orion_bus_enabled:
            return
        body = {"run_id": run_id, "mode": mode, **payload}
        try:
            await self._bus.publish(
                settings.channel_context_exec_event,
                BaseEnvelope(
                    kind=kind,
                    source=_source(),
                    correlation_id=str(uuid4()),
                    payload=body,
                ),
            )
        except Exception as exc:
            logger.warning("context_exec event publish failed kind=%s run_id=%s err=%s", kind, run_id, exc)

    async def started(self, *, run_id: str, mode: ContextExecMode, text: str) -> None:
        await self.publish(
            "context.exec.started.v1",
            run_id=run_id,
            mode=mode,
            payload={"text_head": text[:220]},
        )

    async def verb_step(self, *, run_id: str, mode: ContextExecMode, step: ContextExecVerbStepV1) -> None:
        await self.publish(
            "context.exec.verb_step.v1",
            run_id=run_id,
            mode=mode,
            payload=step.model_dump(mode="json"),
        )

    async def schema_invalid(self, *, run_id: str, mode: ContextExecMode, artifact_type: str | None) -> None:
        await self.publish(
            "context.exec.schema_invalid.v1",
            run_id=run_id,
            mode=mode,
            payload={"artifact_type": artifact_type},
        )

    async def finished(
        self,
        *,
        run_id: str,
        mode: ContextExecMode,
        status: str,
        artifact_type: str | None,
        schema_valid: bool,
        failure_modes: list[str],
    ) -> None:
        await self.publish(
            "context.exec.finished.v1",
            run_id=run_id,
            mode=mode,
            payload={
                "status": status,
                "artifact_type": artifact_type,
                "schema_valid": schema_valid,
                "failure_modes": failure_modes,
            },
        )
