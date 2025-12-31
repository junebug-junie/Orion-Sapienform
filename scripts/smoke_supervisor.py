"""
Quick smoke scripts for the supervisor path.

Usage:
    python scripts/smoke_supervisor.py react
    python scripts/smoke_supervisor.py escalate
"""

import asyncio
import os
import sys
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef, LLMMessage
from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, RecallDirective

ORCH_CHANNEL = os.getenv("CORTEX_ORCH_REQUEST_CHANNEL", "orion-cortex:request")
ORCH_RESULT_PREFIX = os.getenv("CORTEX_ORCH_RESULT_PREFIX", "orion-cortex:result")


def _source() -> ServiceRef:
    return ServiceRef(name="smoke-supervisor", node="local", version="0.0.1")


async def _rpc(bus: OrionBusAsync, payload: CortexClientRequest, *, timeout_sec: float = 120.0) -> dict:
    corr = str(uuid4())
    reply_channel = f"{ORCH_RESULT_PREFIX}:{corr}"
    env = BaseEnvelope(
        kind="cortex.orch.request",
        source=_source(),
        correlation_id=corr,
        reply_to=reply_channel,
        payload=payload.model_dump(mode="json"),
    )
    msg = await bus.rpc_request(ORCH_CHANNEL, env, reply_channel=reply_channel, timeout_sec=timeout_sec)
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok:
        raise RuntimeError(decoded.error)
    return decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else decoded.envelope.payload.model_dump(mode="json")


async def run_react_smoke() -> None:
    bus = OrionBusAsync(url=os.getenv("ORION_BUS_URL", "redis://localhost:6379/0"))
    await bus.connect()

    ctx = CortexClientContext(
        messages=[LLMMessage(role="user", content="Summarize this phrase: Orion supervisor demo.")],
        session_id="smoke-react",
    )
    payload = CortexClientRequest(
        mode="agent",
        verb_name="analyze_text",
        packs=["executive_pack"],
        options={"max_steps": 1},
        recall=RecallDirective(enabled=False),
        context=ctx,
    )
    res = await _rpc(bus, payload)
    print("React smoke result:", res)
    await bus.close()


async def run_escalation_smoke() -> None:
    bus = OrionBusAsync(url=os.getenv("ORION_BUS_URL", "redis://localhost:6379/0"))
    await bus.connect()

    ctx = CortexClientContext(
        messages=[LLMMessage(role="user", content="Plan a small research outline and have council review it.")],
        session_id="smoke-escalate",
    )
    payload = CortexClientRequest(
        mode="agent",
        verb_name="plan_action",
        packs=["executive_pack"],
        options={"force_agent_chain": True},
        recall=RecallDirective(enabled=False),
        context=ctx,
    )
    res = await _rpc(bus, payload, timeout_sec=180.0)
    print("Escalation smoke result:", res)
    await bus.close()


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "react"
    if action == "escalate":
        asyncio.run(run_escalation_smoke())
    else:
        asyncio.run(run_react_smoke())
