"""Start a `dream.carry` durable run through cortex-orch's durable ingress.

Same kickoff shape as orion-actions' world_pulse_journal.submit_durable_run_via_cortex: one
`cortex.orch.request` RPC carrying `context.metadata.durable_run`; cortex-orch hands it to
orion-durable-runs and replies `accepted` only when the runner's receipt names this run,
workflow and resource. Anything else is a reason string, never an exception.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import NAMESPACE_URL, uuid4, uuid5

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.dream_carry import (
    DREAM_CARRY_LLM_ROUTE,
    DREAM_CARRY_WORKFLOW,
    DreamCarryBriefV1,
    dream_carry_run_id,
)
from orion.schemas.durable_run import DurableRunRequestV1
from orion.schemas.resource_admission import ResourceRequirementV1
from orion.schemas.telemetry.dream import DreamSleepDigestV1

# A reason starting with this is bus trouble (timeout, dead connection), not a cortex answer.
TRANSPORT_PREFIX = "transport_"
CARRY_PROMPT = "Carry tonight's dream through words and pictures."


def build_carry_request(
    trigger_id: str, sleep: Optional[DreamSleepDigestV1], *, deadline_sec: float, now: Optional[datetime] = None,
) -> DurableRunRequestV1:
    now = now or datetime.now(timezone.utc)
    return DurableRunRequestV1(
        run_id=dream_carry_run_id(trigger_id),
        workflow=DREAM_CARRY_WORKFLOW,
        correlation_id=str(uuid5(NAMESPACE_URL, f"orion:dream.carry:{trigger_id}")),
        brief=DreamCarryBriefV1(trigger_id=trigger_id, sleep=sleep),
        admission=ResourceRequirementV1(
            resource=f"llm.route.{DREAM_CARRY_LLM_ROUTE}",
            preferred_lane=DREAM_CARRY_LLM_ROUTE,
            priority="background",
            deadline_at=now + timedelta(seconds=float(deadline_sec)),
        ),
    )


async def submit_via_cortex(
    *, bus: Any, source: ServiceRef, request: DurableRunRequestV1, request_channel: str, timeout_sec: float = 20.0,
) -> Optional[str]:
    """None when cortex accepted and the receipt names THIS run, workflow and resource; else the reason."""
    reply_channel = f"orion:cortex:result:dream-carry:{uuid4()}"
    payload = {
        "mode": "brain",
        "context": {
            "messages": [{"role": "user", "content": CARRY_PROMPT}],
            "user_message": CARRY_PROMPT,
            "session_id": request.brief.session_id,
            "metadata": {"durable_run": request.model_dump(mode="json", exclude_none=True)},
        },
    }
    envelope = BaseEnvelope(kind="cortex.orch.request", source=source, correlation_id=request.correlation_id,
                            reply_to=reply_channel, payload=payload)
    try:
        raw = await bus.rpc_request(request_channel, envelope, reply_channel=reply_channel, timeout_sec=timeout_sec)
        decoded = bus.codec.decode(raw.get("data") if isinstance(raw, dict) else raw)
        result = decoded.envelope.payload if decoded.ok and decoded.envelope is not None else None
    except Exception as exc:  # noqa: BLE001
        return f"{TRANSPORT_PREFIX}{type(exc).__name__}: {exc}"[:300]
    if not isinstance(result, dict):
        return "undecodable_reply"
    if result.get("status") != "accepted":
        error = result.get("error") if isinstance(result.get("error"), dict) else {}
        return f"not_accepted:{result.get('status') or 'no_status'}:{error.get('message') or error.get('type') or ''}"[:300]
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    receipt = metadata.get("durable_run") if isinstance(metadata.get("durable_run"), dict) else {}
    if receipt.get("run_id") != request.run_id:
        return "receipt_run_id_mismatch"
    if receipt.get("workflow_kind", receipt.get("workflow")) != request.workflow:
        return "receipt_workflow_mismatch"
    if request.admission is not None and receipt.get("requested_resource") != request.admission.resource:
        return "receipt_resource_mismatch"
    return None
