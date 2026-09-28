"""Submit a durable run through cortex-orch's generic durable ingress.

cortex-orch treats a `CortexClientRequest` whose `context.metadata.durable_run`
validates as `DurableRunRequestV1` as a kickoff: it hands the run to
orion-durable-runs and replies `accepted` without executing anything
(services/orion-cortex-orch/app/durable_runs.py). Shared by self-study reflect
(which then waits for completion) and render_scene (which returns at once and
lets execution-dispatch settle the outcome later).
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.durable_run import DurableRunRequestV1

logger = logging.getLogger("orion.cortex.exec.durable_kickoff")

DEFAULT_KICKOFF_TIMEOUT_SEC = 20.0


class DurableKickoffReply(NamedTuple):
    # Decoded CortexClientResult payload, or None when the RPC failed / was unreadable.
    result: dict[str, Any] | None
    error: str | None

    @property
    def status(self) -> str:
        return str((self.result or {}).get("status") or "")


async def submit_durable_run(
    *,
    bus: Any,
    source: ServiceRef,
    request: DurableRunRequestV1,
    request_channel: str,
    reply_channel: str,
    envelope_correlation_id: str,
    session_id: str,
    user_message: str,
    timeout_sec: float = DEFAULT_KICKOFF_TIMEOUT_SEC,
) -> DurableKickoffReply:
    """One kickoff RPC. Never raises: transport and decode failures come back
    as `result=None` with the error text."""
    payload = {
        "mode": "brain",
        "context": {
            "messages": [{"role": "user", "content": user_message}],
            "user_message": user_message,
            "session_id": session_id,
            "metadata": {"durable_run": request.model_dump(mode="json", exclude_none=True)},
        },
    }
    envelope = BaseEnvelope(
        kind="cortex.orch.request",
        source=source,
        correlation_id=envelope_correlation_id,
        reply_to=reply_channel,
        payload=payload,
    )
    try:
        raw = await bus.rpc_request(request_channel, envelope, reply_channel=reply_channel, timeout_sec=timeout_sec)
    except Exception as exc:  # noqa: BLE001
        return DurableKickoffReply(None, f"{type(exc).__name__}: {exc}")
    try:
        decoded = bus.codec.decode(raw.get("data") if isinstance(raw, dict) else raw)
        result = decoded.envelope.payload if decoded.ok else None
    except Exception as exc:  # noqa: BLE001
        return DurableKickoffReply(None, f"undecodable_reply: {exc}")
    if not isinstance(result, dict):
        return DurableKickoffReply(None, "undecodable_reply")
    return DurableKickoffReply(result, None)


def receipt_mismatch(reply: DurableKickoffReply, request: DurableRunRequestV1) -> str | None:
    """None when the reply proves durable-runs registered THIS run; else why not.

    `status == "accepted"` alone is not proof for an admitted run: cortex-orch
    echoes the runner's `DurableRunReceiptV1` under `metadata.durable_run`, and
    only a receipt naming our run_id / workflow / resource says the run exists.
    """
    if reply.result is None:
        return reply.error or "no_reply"
    if reply.status != "accepted":
        error = reply.result.get("error") if isinstance(reply.result.get("error"), dict) else {}
        detail = error.get("message") or error.get("type")
        return f"not_accepted:{reply.status or 'no_status'}" + (f":{detail}" if detail else "")
    metadata = reply.result.get("metadata") if isinstance(reply.result.get("metadata"), dict) else {}
    receipt = metadata.get("durable_run") if isinstance(metadata.get("durable_run"), dict) else {}
    if receipt.get("run_id") != request.run_id:
        return "receipt_run_id_mismatch"
    if receipt.get("workflow_kind", receipt.get("workflow")) != request.workflow:
        return "receipt_workflow_mismatch"
    if request.admission is not None and receipt.get("requested_resource") != request.admission.resource:
        return "receipt_resource_mismatch"
    return None
