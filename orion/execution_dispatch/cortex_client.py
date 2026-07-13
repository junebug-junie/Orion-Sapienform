from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from orion.cognition.plan_loader import build_plan_for_verb
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.cortex.schemas import PlanExecutionArgs, PlanExecutionRequest

logger = logging.getLogger("orion.execution_dispatch.cortex_client")


class ExecutionDispatchCortexClient:
    """Thin RPC client sending prepared_for_dispatch envelopes to cortex-exec.

    Mirrors orion.harness.cortex_client.HarnessCortexClient's shape (publish
    + subscribe-to-reply + bounded wait via OrionBusAsync.rpc_request) rather
    than reusing that class directly -- orion/harness/ is the FCC/Claude
    harness's cortex access, not a generic cortex-RPC library; forking a
    ~60-line client keeps that service boundary explicit.
    """

    def __init__(
        self,
        bus: OrionBusAsync,
        *,
        request_channel: str,
        result_prefix: str,
        source_name: str = "orion-execution-dispatch-runtime",
        timeout_sec: float = 120.0,
    ) -> None:
        self.bus = bus
        self.request_channel = request_channel
        self.result_prefix = result_prefix
        self.source_name = source_name
        self.timeout_sec = timeout_sec

    async def dispatch(
        self,
        *,
        verb: str,
        mode: str,
        context: dict[str, Any],
        dispatch_id: str,
        timeout_sec: float | None = None,
    ) -> dict[str, Any]:
        """Send one prepared candidate's envelope and await a bounded reply.

        Returns the decoded result payload dict. Raises on RPC failure,
        timeout, or a non-dict/not-ok reply -- callers must catch and record
        dispatch_error, never let this silently produce a fake success.
        """
        plan = build_plan_for_verb(verb, mode=mode)
        req = PlanExecutionRequest(
            plan=plan,
            args=PlanExecutionArgs(
                request_id=dispatch_id,
                trigger_source=self.source_name,
                extra={
                    "mode": mode,
                    "origin": "endogenous.dispatch",
                    "dispatch_id": dispatch_id,
                },
            ),
            context=context,
        )
        # BaseEnvelope.correlation_id must be a real UUID; dispatch_id (e.g.
        # "dispatch:proposal:inspect:execution_dispatch_policy.v1") is not
        # one -- it travels instead in args.extra["dispatch_id"] above and
        # in this envelope's own reply_channel, both sufficient to trace a
        # result back to the originating candidate.
        correlation_id = str(uuid4())
        reply_channel = f"{self.result_prefix}:{correlation_id}"
        source = ServiceRef(name=self.source_name)
        env = BaseEnvelope(
            kind=req.kind,
            source=source,
            correlation_id=correlation_id,
            reply_to=reply_channel,
            payload=req.model_dump(mode="json"),
        )
        logger.info(
            "execution_dispatch cortex RPC -> %s verb=%s dispatch_id=%s corr=%s",
            self.request_channel,
            verb,
            dispatch_id,
            correlation_id,
        )
        msg = await self.bus.rpc_request(
            self.request_channel,
            env,
            reply_channel=reply_channel,
            timeout_sec=timeout_sec if timeout_sec is not None else self.timeout_sec,
        )
        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(f"execution_dispatch cortex RPC failed: {decoded.error}")
        payload = decoded.envelope.payload
        if not isinstance(payload, dict):
            raise RuntimeError("execution_dispatch cortex RPC returned non-dict payload")
        return payload
