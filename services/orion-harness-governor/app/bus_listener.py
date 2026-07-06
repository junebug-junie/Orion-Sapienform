from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any
from uuid import UUID, uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.harness.cortex_client import HarnessCortexClient
from orion.harness.finalize import emit_post_turn_closure, run_harness_finalize_chain
from orion.harness.repair import map_repair_pressure_contract
from orion.harness.runner import HarnessRunner
from orion.harness.substrate_client import HarnessSubstrateClient
from orion.schemas.harness_finalize import HarnessRunRequestV1, HarnessRunV1

from .settings import settings

logger = logging.getLogger("orion-harness-governor.bus")


def _source() -> ServiceRef:
    return ServiceRef(
        name=settings.service_name,
        node=settings.node_name,
        version=settings.service_version,
    )


def _envelope_correlation_id(raw: str | None) -> UUID:
    if raw:
        try:
            return UUID(str(raw))
        except ValueError:
            pass
    return uuid4()


def validate_harness_run_request(request: HarnessRunRequestV1) -> str | None:
    """Return a refusal reason when the motor must not run; None when eligible."""
    thought = request.thought_event
    if thought.disposition == "refuse":
        return "thought_refused"
    if thought.disposition == "defer":
        return "thought_deferred"
    return None


def _grammar_event_ids(receipts: list[Any]) -> list[str]:
    return [r.grammar_event_id for r in receipts if getattr(r, "grammar_event_id", None)]


async def handle_harness_run_request(
    bus: OrionBusAsync,
    request: HarnessRunRequestV1,
    *,
    reply_to: str,
    correlation_id: str | None = None,
    causality_chain: list[str] | None = None,
    runner: HarnessRunner | None = None,
    cortex_client: HarnessCortexClient | None = None,
    substrate_client: HarnessSubstrateClient | None = None,
) -> HarnessRunV1:
    corr = correlation_id or request.correlation_id or str(uuid4())
    causality = list(causality_chain or [])

    refusal = validate_harness_run_request(request)
    if refusal is not None:
        run = HarnessRunV1(
            correlation_id=corr,
            final_text=None,
            finalize_ran=False,
            step_count=0,
            compliance_verdict="refused",
            grounding_status=refusal,
        )
        await _reply_and_artifact(bus, run, reply_to=reply_to, corr=corr, causality=causality)
        return run

    repair_overlay = map_repair_pressure_contract(request.repair_pressure_contract)
    motor_runner = runner or HarnessRunner(
        bus,
        grammar_channel=settings.channel_grammar_event,
        step_channel=settings.channel_harness_run_step,
        fcc_timeout_sec=settings.fcc_timeout_sec,
    )
    cortex = cortex_client or HarnessCortexClient(
        bus,
        request_channel=settings.channel_cortex_exec_request,
        result_prefix=settings.channel_cortex_exec_result_prefix,
        source_name=settings.service_name,
        timeout_sec=settings.finalize_reflect_timeout_sec,
    )
    substrate = substrate_client or HarnessSubstrateClient(
        bus,
        request_channel=settings.channel_finalize_appraisal_request,
        result_prefix=settings.channel_finalize_appraisal_result_prefix,
        source_name=settings.service_name,
        timeout_sec=settings.substrate_finalize_timeout_sec,
    )

    motor = await motor_runner.run(request, repair_overlay=repair_overlay)
    if not motor.draft_text or motor.draft_molecule is None:
        run = HarnessRunV1(
            correlation_id=corr,
            final_text=None,
            draft_text=None,
            finalize_ran=False,
            step_count=motor.step_count,
            exit_code=motor.exit_code,
            compliance_verdict=motor.compliance_verdict if motor.compliance_verdict != "completed" else "failed",
            grounding_status=motor.grounding_status,
            grammar_event_ids=_grammar_event_ids(motor.grammar_receipts),
        )
        await _reply_and_artifact(bus, run, reply_to=reply_to, corr=corr, causality=causality)
        return run

    async def _substrate_client(molecule: Any) -> Any:
        return await substrate.finalize_appraisal(molecule, correlation_id=corr)

    chain = await run_harness_finalize_chain(
        correlation_id=corr,
        draft_text=motor.draft_text,
        draft_molecule=motor.draft_molecule,
        thought=request.thought_event,
        grammar_receipts=motor.grammar_receipts,
        repair_overlay=repair_overlay,
        user_message=request.user_message,
        voice_contract=request.answer_contract,
        cortex_client=cortex,
        substrate_client=_substrate_client,
        bus=bus,
    )

    run = HarnessRunV1(
        correlation_id=corr,
        final_text=chain.final_text,
        draft_text=motor.draft_text,
        substrate_appraisal=chain.substrate_appraisal,
        reflection=chain.reflection,
        verdict_molecule_id=chain.verdict_molecule_id,
        finalize_ran=True,
        finalize_changed=chain.finalize_changed,
        quick_lane_skipped_5b=chain.quick_lane_skipped_5b,
        step_count=motor.step_count,
        exit_code=motor.exit_code,
        compliance_verdict=motor.compliance_verdict,
        grounding_status=motor.grounding_status,
        grammar_event_ids=_grammar_event_ids(motor.grammar_receipts),
    )
    await _reply_and_artifact(bus, run, reply_to=reply_to, corr=corr, causality=causality)

    await emit_post_turn_closure(
        correlation_id=corr,
        outcome_molecule=chain.outcome_molecule,
        verdict_molecule_id=chain.verdict_molecule_id,
        grammar_event_ids=run.grammar_event_ids,
        channel=settings.channel_post_turn_closure,
        bus=bus,
    )
    return run


async def _reply_and_artifact(
    bus: OrionBusAsync,
    run: HarnessRunV1,
    *,
    reply_to: str,
    corr: str,
    causality: list[str],
) -> None:
    payload = run.model_dump(mode="json")
    envelope = BaseEnvelope(
        kind="harness.run.v1",
        source=_source(),
        correlation_id=_envelope_correlation_id(corr),
        causality_chain=causality,
        payload=payload,
    )
    await bus.publish(reply_to, envelope)
    await bus.publish(settings.channel_harness_run_artifact, envelope)
    logger.info(
        "harness run complete corr=%s reply=%s artifact=%s finalize_ran=%s",
        corr,
        reply_to,
        settings.channel_harness_run_artifact,
        run.finalize_ran,
    )


async def run_bus_worker(stop_event: asyncio.Event | None = None) -> None:
    if not settings.orion_bus_enabled:
        logger.info("Bus disabled; worker not started")
        return
    if not settings.orion_harness_governor_enabled:
        logger.info("Harness governor disabled; worker not started")
        return

    bus = OrionBusAsync(url=settings.orion_bus_url)
    channel = settings.channel_harness_run_request
    await bus.connect()
    logger.info("subscribed channel=%s", channel)

    try:
        async with bus.subscribe(channel) as pubsub:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                try:
                    msg = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0),
                        timeout=1.2,
                    )
                except asyncio.TimeoutError:
                    continue
                if not msg or msg.get("type") not in ("message", "pmessage"):
                    continue
                try:
                    await _handle_bus_message(bus, msg)
                except Exception:
                    logger.exception("unhandled bus worker error")
    except asyncio.CancelledError:
        raise
    finally:
        with suppress(Exception):
            await bus.close()


async def _handle_bus_message(bus: OrionBusAsync, raw_msg: dict[str, Any]) -> None:
    decoded = bus.codec.decode(raw_msg.get("data"))
    if not decoded.ok:
        logger.warning("decode failed: %s", decoded.error)
        return

    env = decoded.envelope
    reply_channel = env.reply_to or (env.payload or {}).get("reply_channel")
    if not reply_channel:
        logger.warning("missing reply_to corr=%s", env.correlation_id)
        return

    kind = env.kind or ""
    if kind not in ("harness.run.request.v1", "legacy.message"):
        logger.warning("unsupported kind=%s", kind)
        return

    corr = str(env.correlation_id or uuid4())
    payload = env.payload or {}
    causality = list(env.causality_chain or [])

    try:
        request = HarnessRunRequestV1.model_validate(payload)
        if not request.correlation_id:
            request = request.model_copy(update={"correlation_id": corr})
        await handle_harness_run_request(
            bus,
            request,
            reply_to=reply_channel,
            correlation_id=corr,
            causality_chain=causality,
        )
    except Exception as exc:
        logger.error("harness run error corr=%s err=%s", corr, exc)
        err_run = HarnessRunV1(
            correlation_id=corr,
            final_text=None,
            finalize_ran=False,
            step_count=0,
            compliance_verdict="failed",
            grounding_status=str(exc),
        )
        await _reply_and_artifact(
            bus,
            err_run,
            reply_to=reply_channel,
            corr=corr,
            causality=causality,
        )
