"""Cortex is the kickoff for durable cognition runs.

A `CortexClientRequest` whose `context.metadata.durable_run` validates as
`DurableRunRequestV1` is not executed here: cortex-orch records that
cognition is starting and hands the run to `orion-durable-runs` over
`orion:durable:run:request` (single consumer), replying `accepted` at once.
The run itself takes 10-40 minutes and is checkpointed by the runner; cortex
does not hold the RPC open for it.

Mirrors `workflow_runtime.has_explicit_workflow_request` (an explicit
metadata key, checked before any routing), and stays out of the
verb-runtime path on purpose -- nothing about how chat verbs execute
changes. Design: docs/superpowers/specs/2026-09-06-durable-cognition-runs-
from-cortex-design.md.
"""

from __future__ import annotations

import logging
from typing import Any, Dict
from uuid import UUID, uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.resilience import publish_with_reconnect
from orion.schemas.cortex.contracts import CortexClientRequest, CortexClientResult
from orion.schemas.durable_run import (
    DURABLE_RUN_REQUEST_CHANNEL,
    DURABLE_RUN_REQUEST_KIND,
    DurableRunRequestV1,
)

logger = logging.getLogger("orion-cortex-orch.durable_runs")

DURABLE_RUN_METADATA_KEY = "durable_run"


def durable_run_request_from(req: CortexClientRequest) -> DurableRunRequestV1 | None:
    """The validated request, or None when the metadata key is absent.
    A present-but-invalid key raises: a malformed kickoff must fail loudly at
    cortex, not be quietly routed into chat."""
    metadata = req.context.metadata if isinstance(req.context.metadata, dict) else {}
    raw = metadata.get(DURABLE_RUN_METADATA_KEY)
    if not isinstance(raw, dict):
        return None
    return DurableRunRequestV1.model_validate(raw)


def has_durable_run_request(req: CortexClientRequest) -> bool:
    metadata = req.context.metadata if isinstance(req.context.metadata, dict) else {}
    return isinstance(metadata.get(DURABLE_RUN_METADATA_KEY), dict)


def _corr(raw: str) -> UUID:
    try:
        return UUID(str(raw))
    except (ValueError, TypeError):
        return uuid4()


async def dispatch_durable_run(
    *,
    bus: Any,
    source: ServiceRef,
    req: CortexClientRequest,
    correlation_id: str,
    channel: str = DURABLE_RUN_REQUEST_CHANNEL,
) -> CortexClientResult:
    request = durable_run_request_from(req)
    assert request is not None  # caller checked has_durable_run_request
    verb = f"durable:{request.workflow}"
    try:
        await publish_with_reconnect(
            bus,
            channel,
            BaseEnvelope(
                kind=DURABLE_RUN_REQUEST_KIND,
                source=source,
                correlation_id=_corr(request.correlation_id),
                payload=request.model_dump(mode="json"),
            ),
            log_label="cortex_durable_run_dispatch",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("durable_run_dispatch_failed corr=%s run=%s", correlation_id, request.run_id)
        return CortexClientResult(
            ok=False,
            mode=str(req.mode),
            verb=verb,
            status="fail",
            final_text=None,
            memory_used=False,
            recall_debug={},
            steps=[],
            error={"message": str(exc), "type": type(exc).__name__, "run_id": request.run_id},
            correlation_id=correlation_id,
            metadata=_meta(request, "dispatch_failed"),
        )
    logger.info(
        "durable_run_dispatched run=%s workflow=%s corr=%s channel=%s",
        request.run_id,
        request.workflow,
        request.correlation_id,
        channel,
    )
    return CortexClientResult(
        ok=True,
        mode=str(req.mode),
        verb=verb,
        status="accepted",
        final_text=None,
        memory_used=False,
        recall_debug={},
        steps=[],
        correlation_id=correlation_id,
        metadata=_meta(request, "dispatched"),
    )


def _meta(request: DurableRunRequestV1, status: str) -> Dict[str, Any]:
    return {
        "durable_run": {
            "run_id": request.run_id,
            "workflow": request.workflow,
            "status": status,
            "request_channel": DURABLE_RUN_REQUEST_CHANNEL,
        }
    }
