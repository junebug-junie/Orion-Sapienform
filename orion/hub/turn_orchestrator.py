from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Awaitable, Callable, Protocol

from orion.cognition.answer_contract_normalize import heuristic_answer_contract
from orion.hub.association import build_hub_association_bundle
from orion.hub.harness_step_stream import relay_harness_run_steps
from orion.hub.turn_request import build_orion_turn_request
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.schemas.harness_finalize import HarnessRunRequestV1, HarnessRunV1
from orion.schemas.pre_turn_appraisal import (
    PreTurnAppraisalOptionsV1,
    PreTurnAppraisalRequestV1,
    TurnAppraisalBundleV1,
)
from orion.schemas.thought import StanceReactRequestV1, ThoughtEventV1
from orion.substrate.appraisal.turn_window import build_turn_window

logger = logging.getLogger("orion.hub.turn_orchestrator")

DEFAULT_UNIFIED_TURN_FCC_MODEL_LABEL = "MODEL_SONNET"

EmitObservationFn = Callable[..., Any]
FrameSender = Callable[[dict[str, Any]], Awaitable[None]]


class _WebSocketLike(Protocol):
    async def send_json(self, data: dict[str, Any]) -> None: ...


def _repair_pressure_contract(repair_bundle: TurnAppraisalBundleV1 | None) -> dict[str, Any] | None:
    if repair_bundle is None:
        return None
    contract = (repair_bundle.metadata_attachments or {}).get("repair_pressure_contract")
    if isinstance(contract, dict) and contract:
        return dict(contract)
    rp = repair_bundle.paradigms.get("repair_pressure")
    if rp is not None and rp.contract_delta:
        return dict(rp.contract_delta)
    return None


def _thought_deferred_frame(thought: ThoughtEventV1, *, correlation_id: str) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "type": "turn_deferred",
        "correlation_id": correlation_id,
        "reason": (
            thought.disposition_reasons[0]
            if thought.disposition_reasons
            else thought.disposition
        ),
    }
    if thought.boundary_register:
        frame["boundary_register"] = True
    return frame


def _harness_error_frame(run: HarnessRunV1, *, correlation_id: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "type": "turn_error",
        "correlation_id": correlation_id,
        "finalize_ran": bool(run.finalize_ran),
    }
    if run.draft_text and run.substrate_appraisal is None:
        base["phase"] = "substrate_appraisal"
        return base
    if run.substrate_appraisal is not None and (run.reflection is None or not run.final_text):
        base["phase"] = "finalize"
        return base
    base["phase"] = "harness"
    if run.step_count:
        base["partial"] = run.step_count
    if run.grounding_status:
        base["error"] = run.grounding_status
    return base


def _success_frames(run: HarnessRunV1, *, correlation_id: str) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    if run.substrate_appraisal is not None:
        frames.append(
            {
                "type": "substrate_appraisal",
                "correlation_id": correlation_id,
                "appraisal": run.substrate_appraisal.model_dump(mode="json"),
            }
        )
    if run.reflection is not None:
        frames.append(
            {
                "type": "reflection",
                "correlation_id": correlation_id,
                "reflection": run.reflection.model_dump(mode="json"),
            }
        )
    frames.append(
        {
            "type": "final",
            "correlation_id": correlation_id,
            "mode": "orion",
            "llm_response": run.final_text,
            "finalize_ran": run.finalize_ran,
            "finalize_changed": run.finalize_changed,
            "harness_step_count": run.step_count,
            "harness_grounding_status": run.grounding_status,
        }
    )
    return frames


async def _run_pre_turn_appraisal(
    *,
    bus: Any,
    correlation_id: str,
    session_id: str | None,
    user_message: str,
    continuity_messages: list[dict[str, Any]] | None,
    settings: Any,
) -> TurnAppraisalBundleV1 | None:
    if bus is None or not getattr(settings, "ENABLE_PRE_TURN_APPRAISAL", False):
        return None
    from scripts.pre_turn_appraisal_client import PreTurnAppraisalClient

    turn_window = build_turn_window(
        continuity_messages or [{"role": "user", "content": user_message}]
    )
    paradigms = str(getattr(settings, "PRE_TURN_APPRAISAL_PARADIGMS", "repair_pressure"))
    timeout_ms = int(getattr(settings, "PRE_TURN_APPRAISAL_TIMEOUT_MS", 60000))
    return await PreTurnAppraisalClient(bus).appraise(
        PreTurnAppraisalRequestV1(
            correlation_id=correlation_id,
            session_id=str(session_id or "anonymous"),
            turn_window=turn_window,
            paradigms_requested=[p.strip() for p in paradigms.split(",") if p.strip()],
            contract_before={"mode": "default"},
            options=PreTurnAppraisalOptionsV1(timeout_ms=timeout_ms),
        ),
        correlation_id=correlation_id,
    )


async def execute_unified_turn(
    *,
    bus: Any,
    correlation_id: str,
    session_id: str | None,
    user_message: str,
    payload: dict[str, Any] | None = None,
    continuity_messages: list[dict[str, Any]] | None = None,
    emit_observation_fn: EmitObservationFn | None = None,
    settings: Any | None = None,
    send_frame: FrameSender | None = None,
) -> list[dict[str, Any]]:
    """Run the unified Orion turn saga and return WS frames (never includes draft_text)."""
    from scripts.settings import settings as hub_settings

    cfg = settings or hub_settings
    payload = payload or {}

    if emit_observation_fn is not None:
        try:
            emit_observation_fn(surface_text=user_message, source_id=session_id or "anonymous")
        except Exception:
            logger.debug("emit_observation hook failed corr=%s", correlation_id, exc_info=True)
    else:
        try:
            from orion.mind.substrate_emit import emit_observation

            emit_observation(surface_text=user_message, source_id=session_id or "anonymous")
        except Exception:
            logger.debug("emit_observation failed corr=%s", correlation_id, exc_info=True)

    repair_bundle = await _run_pre_turn_appraisal(
        bus=bus,
        correlation_id=correlation_id,
        session_id=session_id,
        user_message=user_message,
        continuity_messages=continuity_messages,
        settings=cfg,
    )
    build_orion_turn_request(
        correlation_id=correlation_id,
        session_id=session_id,
        user_message=user_message,
        repair_bundle=repair_bundle,
    )
    association = build_hub_association_bundle(
        correlation_id=correlation_id,
        repair_bundle=repair_bundle,
    )

    if bus is None:
        return [
            {
                "type": "turn_error",
                "phase": "config",
                "correlation_id": correlation_id,
                "error": "bus_unavailable",
            }
        ]

    from scripts.harness_governor_client import HarnessGovernorClient
    from scripts.thought_client import ThoughtClient

    stance_req = StanceReactRequestV1(
        correlation_id=correlation_id,
        session_id=session_id,
        user_message=user_message,
        association=association,
        repair_bundle=repair_bundle,
        stance_inputs={"user_message": user_message},
    )
    thought = await ThoughtClient(bus).react(stance_req, correlation_id=correlation_id)
    if thought is None:
        return [
            {
                "type": "turn_deferred",
                "correlation_id": correlation_id,
                "reason": "stance_react_timeout",
            }
        ]
    if thought.disposition in ("defer", "refuse"):
        return [_thought_deferred_frame(thought, correlation_id=correlation_id)]

    harness_req = HarnessRunRequestV1(
        correlation_id=correlation_id,
        thought_event=thought,
        user_message=user_message,
        permissions=ContextExecPermissionV1(
            read_memory=True,
            read_graph=True,
            read_recall=True,
            read_repo=True,
            read_runtime_logs=True,
            read_redis_traces=True,
        ),
        answer_contract=heuristic_answer_contract(user_message),
        repair_pressure_contract=_repair_pressure_contract(repair_bundle),
        fcc_model_label=payload.get("fcc_model_label") or DEFAULT_UNIFIED_TURN_FCC_MODEL_LABEL,
    )
    step_stop = asyncio.Event()
    step_task = None
    if send_frame is not None and bus is not None:
        step_channel = getattr(cfg, "CHANNEL_HARNESS_RUN_STEP", "orion:harness:run:step")
        step_task = asyncio.create_task(
            relay_harness_run_steps(
                bus,
                correlation_id=correlation_id,
                channel=step_channel,
                send_frame=send_frame,
                stop_event=step_stop,
            )
        )
    try:
        run = await HarnessGovernorClient(bus).run(harness_req, correlation_id=correlation_id)
    finally:
        step_stop.set()
        if step_task is not None:
            step_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await step_task
    if run is None:
        return [
            {
                "type": "turn_error",
                "phase": "harness",
                "correlation_id": correlation_id,
                "finalize_ran": False,
                "error": "harness_rpc_timeout",
            }
        ]
    if not run.finalize_ran or not run.final_text:
        return [_harness_error_frame(run, correlation_id=correlation_id)]
    return _success_frames(run, correlation_id=correlation_id)


async def run_unified_turn(
    websocket: _WebSocketLike,
    *,
    bus: Any,
    correlation_id: str,
    session_id: str | None,
    user_message: str,
    payload: dict[str, Any] | None = None,
    continuity_messages: list[dict[str, Any]] | None = None,
    with_biometrics: Callable[[dict[str, Any], Any], Awaitable[dict[str, Any]]] | None = None,
    biometrics_cache: Any = None,
) -> list[dict[str, Any]]:
    """Execute unified turn and emit WS frames."""
    async def _send_live_frame(frame: dict[str, Any]) -> None:
        outbound = frame
        if with_biometrics is not None:
            outbound = await with_biometrics(frame, cache=biometrics_cache)
        await websocket.send_json(outbound)

    frames = await execute_unified_turn(
        bus=bus,
        correlation_id=correlation_id,
        session_id=session_id,
        user_message=user_message,
        payload=payload,
        continuity_messages=continuity_messages,
        send_frame=_send_live_frame,
    )
    for frame in frames:
        outbound = frame
        if with_biometrics is not None:
            outbound = await with_biometrics(frame, cache=biometrics_cache)
        await websocket.send_json(outbound)
    return frames
