from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.journaler import (
    JOURNAL_WRITE_KIND,
    build_autonomy_episode_trigger,
    build_compose_request,
    build_write_payload,
    draft_from_cortex_result,
)

logger = logging.getLogger(__name__)


def _new_reply_channel(prefix: str) -> str:
    return f"{prefix}:{uuid4()}"


async def dispatch_autonomy_episode_journal(
    *,
    bus: OrionBusAsync,
    parent: BaseEnvelope,
    source: ServiceRef,
    goal_artifact_id: str,
    spawned_correlation_id: str,
    narrative_seed: str,
    cortex_request_channel: str,
    cortex_result_prefix: str,
    journal_write_channel: str,
    timeout_sec: float,
    session_id: str = "orion",
    user_id: str = "juniper",
    author: str = "orion",
) -> dict[str, Any]:
    """Compose autonomy episode journal via cortex RPC and publish journal write."""
    await bus.connect()
    trigger = build_autonomy_episode_trigger(
        goal_artifact_id=goal_artifact_id,
        spawned_correlation_id=spawned_correlation_id,
        narrative_seed=narrative_seed,
    )
    req = build_compose_request(
        trigger,
        session_id=session_id,
        user_id=user_id,
        trace_id=spawned_correlation_id,
        recall_enabled=False,
        options={"timeout_sec": timeout_sec},
    )
    reply_channel = _new_reply_channel(cortex_result_prefix)
    req_env = parent.derive_child(
        kind="cortex.orch.request",
        source=source,
        payload=req.model_dump(mode="json"),
        reply_to=reply_channel,
    )
    msg = await bus.rpc_request(
        cortex_request_channel,
        req_env,
        reply_channel=reply_channel,
        timeout_sec=timeout_sec,
    )
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok or decoded.envelope is None:
        raise RuntimeError(f"cortex_orch_decode_failed:{decoded.error}")
    orch_payload = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
    if not orch_payload.get("ok", False):
        raise RuntimeError(f"journal_compose_failed:{orch_payload.get('error') or orch_payload.get('status')}")
    draft = draft_from_cortex_result(orch_payload)
    write = build_write_payload(
        draft,
        trigger=trigger,
        correlation_id=spawned_correlation_id,
        author=author,
    )
    write_env = parent.derive_child(
        kind=JOURNAL_WRITE_KIND,
        source=source,
        payload=write.model_dump(mode="json"),
        reply_to=None,
    )
    await bus.publish(journal_write_channel, write_env)
    logger.info(
        "substrate_episode_journal_dispatched goal=%s spawned=%s entry_id=%s",
        goal_artifact_id,
        spawned_correlation_id,
        write.entry_id,
    )
    return {
        "draft": draft.model_dump(mode="json"),
        "write": write.model_dump(mode="json"),
        "orch_payload": orch_payload,
    }
