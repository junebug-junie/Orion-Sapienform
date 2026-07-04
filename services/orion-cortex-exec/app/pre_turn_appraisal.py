from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef
from orion.schemas.pre_turn_appraisal import PreTurnAppraisalRequestV1, TurnAppraisalBundleV1
from orion.substrate.appraisal import REPAIR_PRESSURE_CONTRACT_METADATA_KEY
from orion.substrate.appraisal.paradigms.registry import PARADIGM_REGISTRY, ParadigmBuildContext

from .settings import settings

logger = logging.getLogger("orion.cortex.pre_turn_appraisal")


def _source() -> ServiceRef:
    return ServiceRef(name=settings.service_name, version=settings.service_version, node=settings.node_name)


async def _llm_probe_call(bus: OrionBusAsync, *, prompt: str, route: str, timeout_sec: float) -> dict[str, Any]:
    rpc_corr = str(uuid4())
    reply_channel = f"orion:exec:result:LLMGatewayService:{rpc_corr}"
    payload = ChatRequestPayload(
        messages=[LLMMessage(role="user", content=prompt)],
        route=route,
        options={
            "return_logprobs": True,
            "logprobs_top_k": 8,
            "logprob_summary_only": False,
            "logprob_probe_mode": "native_completion",
            "max_tokens": 128,
            "purpose": "repair_pressure_probe",
            "skip_spark_candidate_publish": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    env = BaseEnvelope(
        kind="llm.chat.request",
        source=_source(),
        correlation_id=rpc_corr,
        reply_to=reply_channel,
        payload=payload.model_dump(mode="json"),
    )
    msg = await bus.rpc_request(settings.channel_llm_intake, env, reply_channel=reply_channel, timeout_sec=timeout_sec)
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok or not isinstance(decoded.envelope.payload, dict):
        return {"text": "", "llm_uncertainty": {"available": False}}
    return decoded.envelope.payload


async def handle_pre_turn_appraisal_request(env: BaseEnvelope) -> BaseEnvelope:
    payload_obj = env.payload.model_dump(mode="json") if hasattr(env.payload, "model_dump") else env.payload
    req = PreTurnAppraisalRequestV1.model_validate(payload_obj or {})
    failed: list[str] = []
    paradigms: dict[str, Any] = {}
    metadata_attachments: dict[str, Any] = {}
    grammar_scalars: dict[str, dict[str, float]] = {}

    bus = _get_bus()
    timeout_sec = max(0.1, req.options.timeout_ms / 1000.0)

    async def llm_caller(prompt: str) -> dict[str, Any]:
        return await _llm_probe_call(
            bus,
            prompt=prompt,
            route=settings.repair_pressure_probe_route,
            timeout_sec=timeout_sec,
        )

    build_ctx = ParadigmBuildContext(
        llm_caller=llm_caller,
        weights_path=settings.repair_pressure_weights_v2_path,
    )

    for paradigm_name in req.paradigms_requested:
        factory = PARADIGM_REGISTRY.get(paradigm_name)
        if factory is None:
            logger.warning(
                "pre_turn_appraisal_unknown_paradigm corr=%s paradigm=%s",
                req.correlation_id,
                paradigm_name,
            )
            failed.append(paradigm_name)
            continue
        try:
            paradigm = factory(build_ctx)
            slice_ = await asyncio.wait_for(paradigm.run(req), timeout=timeout_sec)
            paradigms[paradigm_name] = slice_.model_dump(mode="json")
            grammar_scalars[paradigm_name] = {
                "level": slice_.level,
                "confidence": slice_.confidence,
            }
            if paradigm_name == "repair_pressure":
                before_mode = str((req.contract_before or {}).get("mode") or "")
                after_mode = str((slice_.contract_delta or {}).get("mode") or before_mode)
                if before_mode != after_mode:
                    metadata_attachments[REPAIR_PRESSURE_CONTRACT_METADATA_KEY] = dict(slice_.contract_delta)
        except Exception:
            logger.warning(
                "pre_turn_appraisal_paradigm_failed corr=%s paradigm=%s",
                req.correlation_id,
                paradigm_name,
                exc_info=True,
            )
            failed.append(paradigm_name)

    bundle = TurnAppraisalBundleV1(
        correlation_id=req.correlation_id,
        paradigms=paradigms,
        metadata_attachments=metadata_attachments,
        grammar_scalars=grammar_scalars,
        failed_paradigms=failed,
    )
    return BaseEnvelope(
        kind="pre_turn_appraisal.result.v1",
        source=_source(),
        correlation_id=req.correlation_id,
        causality_chain=env.causality_chain,
        payload=bundle.model_dump(mode="json"),
    )


_BUS: OrionBusAsync | None = None


def _get_bus() -> OrionBusAsync:
    assert _BUS is not None, "pre_turn_appraisal bus not initialized"
    return _BUS


def bind_pre_turn_appraisal_bus(bus: OrionBusAsync) -> None:
    global _BUS
    _BUS = bus
