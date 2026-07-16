from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Awaitable, Callable, Protocol

from orion.schemas.cognition.answer_contract import AnswerContract
from orion.hub.association import build_hub_association_bundle
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
from orion.fcc.context_budget import (
    apply_context_overflow_hint,
    is_context_overflow_text,
    max_context_tokens,
)

logger = logging.getLogger("orion.hub.turn_orchestrator")

DEFAULT_UNIFIED_TURN_FCC_MODEL_LABEL = "MODEL_SONNET"

EmitObservationFn = Callable[..., Any]


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


async def _publish_unified_turn_chat_grammar(
    *,
    bus: Any,
    correlation_id: str,
    session_id: str | None,
    user_message: str,
    repair_bundle: TurnAppraisalBundleV1 | None,
    stance_disposition: str,
    stance_disposition_reasons: list[str],
    stance_boundary_register: bool,
    settings: Any,
) -> None:
    """Orion capability: unified-turn conversational-envelope grammar trace.

    Publishes the same hub.chat: trace the classic websocket_handler chat path
    already produces (session, utterance word count, repair signal), extended
    with the Thought stance decision (proceed/defer/refuse + reasons +
    boundary register) -- a fact with no representation anywhere else in the
    substrate ladder. Fires once per turn, right after the stance decision is
    known, regardless of whether the turn goes on to the harness or stops
    here on defer/refuse. Awaited directly (matches this file's other publish
    calls, e.g. publish_chat_history/publish_chat_turn in
    _publish_unified_turn_chat_history) rather than scheduled as a background
    task -- this is a single lightweight bus publish, not a network round
    trip to an LLM, so the added latency ahead of the harness dispatch is
    negligible next to the harness call itself. Fail-open: chat must work
    whether grammar publishing is on or off, or this call fails outright.
    """
    if bus is None or not getattr(settings, "PUBLISH_HUB_CHAT_GRAMMAR", False):
        return
    try:
        from scripts.grammar_emit import build_chat_turn_grammar_events
        from scripts.grammar_publish import publish_hub_chat_grammar_trace
        from scripts.pre_turn_appraisal_wiring import repair_pressure_grammar_scalars

        repair_pressure_level, repair_pressure_confidence = repair_pressure_grammar_scalars(
            pre_turn_bundle=repair_bundle,
            substrate_summary=None,
        )
        events = build_chat_turn_grammar_events(
            turn_id=correlation_id,
            session_id=str(session_id or "anonymous"),
            node_id=settings.NODE_NAME,
            word_count=len((user_message or "").split()),
            repair_pressure_level=repair_pressure_level,
            repair_pressure_confidence=repair_pressure_confidence,
            has_repair_signal=repair_bundle is not None,
            stance_disposition=stance_disposition,
            stance_disposition_reasons=stance_disposition_reasons,
            stance_boundary_register=stance_boundary_register,
        )
        await publish_hub_chat_grammar_trace(
            bus,
            events,
            correlation_id=correlation_id,
            channel=settings.GRAMMAR_EVENT_CHANNEL,
            enabled=True,
        )
    except Exception:
        logger.warning("unified_turn_chat_grammar_publish_failed corr=%s", correlation_id, exc_info=True)


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


_PARTIAL_DRAFT_MAX_LEN = 2000


def _with_overflow_hint(text: str | None) -> str | None:
    if not text:
        return text
    if not is_context_overflow_text(text):
        return text
    return apply_context_overflow_hint(text, n_ctx=max_context_tokens())


def _partial_draft_from_run(run: HarnessRunV1) -> str | None:
    draft = run.draft_text
    if not draft:
        return None
    if len(draft) <= _PARTIAL_DRAFT_MAX_LEN:
        return draft
    return draft[:_PARTIAL_DRAFT_MAX_LEN]


def _finalize_phase_error(run: HarnessRunV1) -> bool:
    if not run.draft_text:
        return False
    if run.finalize_ran and run.final_text:
        return False
    if run.substrate_appraisal is not None or run.reflection is not None:
        return True
    return "orion_voice_finalize" in (run.grounding_status or "")


def _harness_error_frame(run: HarnessRunV1, *, correlation_id: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "type": "turn_error",
        "correlation_id": correlation_id,
        "finalize_ran": bool(run.finalize_ran),
    }
    if run.grounding_status:
        base["error"] = _with_overflow_hint(run.grounding_status) or run.grounding_status
        if is_context_overflow_text(run.grounding_status or ""):
            base["error_code"] = "context_overflow"
            base["context_overflow"] = True
    if run.draft_text and run.substrate_appraisal is None and not _finalize_phase_error(run):
        base["phase"] = "harness" if (run.compliance_verdict or "").strip().lower() in {
            "partial",
            "failed",
        } else "substrate_appraisal"
        partial = _partial_draft_from_run(run)
        if partial:
            base["partial_draft"] = _with_overflow_hint(partial) or partial
        return base
    if _finalize_phase_error(run) or (
        run.substrate_appraisal is not None and (run.reflection is None or not run.final_text)
    ):
        base["phase"] = "finalize"
        partial = _partial_draft_from_run(run)
        if partial:
            base["partial_draft"] = _with_overflow_hint(partial) or partial
        if run.grounding_status:
            base["error"] = _with_overflow_hint(run.grounding_status) or run.grounding_status
            if is_context_overflow_text(run.grounding_status or ""):
                base["error_code"] = "context_overflow"
                base["context_overflow"] = True
        return base
    base["phase"] = "harness"
    if run.step_count:
        base["partial"] = run.step_count
    partial = _partial_draft_from_run(run)
    if partial:
        base["partial_draft"] = _with_overflow_hint(partial) or partial
    if run.grounding_status:
        base["error"] = _with_overflow_hint(run.grounding_status) or run.grounding_status
        if is_context_overflow_text(run.grounding_status or ""):
            base["error_code"] = "context_overflow"
            base["context_overflow"] = True
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
    final_text = _with_overflow_hint(run.final_text) or run.final_text
    final_frame: dict[str, Any] = {
        "type": "final",
        "correlation_id": correlation_id,
        "mode": "orion",
        "llm_response": final_text,
        "finalize_ran": run.finalize_ran,
        "finalize_changed": run.finalize_changed,
        "harness_step_count": run.step_count,
        "harness_grounding_status": run.grounding_status,
    }
    if is_context_overflow_text(run.final_text or ""):
        final_frame["context_overflow"] = True
    if run.recall_debug is not None:
        final_frame["recall_debug"] = run.recall_debug
    if run.memory_digest:
        final_frame["memory_digest"] = run.memory_digest
    frames.append(final_frame)
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
    harness_rpc_bus: Any | None = None,
    harness_step_relay: Any | None = None,
    harness_step_queue: asyncio.Queue | None = None,
) -> list[dict[str, Any]]:
    """Orion capability: unified Hub chat turn.

    Owns the Hub-side saga: surface observation, optional pre-turn appraisal,
    association-bundle construction, the Thought stance RPC, defer/refuse
    admission, the HarnessRunRequestV1 handoff to the harness governor over
    bus RPC (with step-relay liveness), and the final WebSocket frames. It
    delegates the FCC motor and finalize chain to the governor; the returned
    frames never include draft_text.

    Runtime evidence: correlation_id-linked harness steps, turn_deferred /
    turn_error / success frames, HarnessRunV1 from the governor, and
    unified-turn chat envelopes. Start here when an Orion-mode turn never
    reached the governor (harness_rpc_timeout) or a finalized result was not
    handed back or persisted.
    """
    from scripts.settings import settings as hub_settings

    cfg = settings or hub_settings
    payload = dict(payload or {})

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
    react_result = await ThoughtClient(bus).react(stance_req, correlation_id=correlation_id)
    thought = react_result.thought
    if thought is None:
        await _publish_unified_turn_chat_grammar(
            bus=bus,
            correlation_id=correlation_id,
            session_id=session_id,
            user_message=user_message,
            repair_bundle=repair_bundle,
            stance_disposition="stance_timeout",
            stance_disposition_reasons=[react_result.failure_reason or "stance_react_timeout"],
            stance_boundary_register=False,
            settings=cfg,
        )
        return [
            {
                "type": "turn_deferred",
                "correlation_id": correlation_id,
                "reason": react_result.failure_reason or "stance_react_timeout",
            }
        ]
    if thought.disposition in ("defer", "refuse"):
        await _publish_unified_turn_chat_grammar(
            bus=bus,
            correlation_id=correlation_id,
            session_id=session_id,
            user_message=user_message,
            repair_bundle=repair_bundle,
            stance_disposition=thought.disposition,
            stance_disposition_reasons=thought.disposition_reasons,
            stance_boundary_register=bool(thought.boundary_register),
            settings=cfg,
        )
        return [_thought_deferred_frame(thought, correlation_id=correlation_id)]

    await _publish_unified_turn_chat_grammar(
        bus=bus,
        correlation_id=correlation_id,
        session_id=session_id,
        user_message=user_message,
        repair_bundle=repair_bundle,
        stance_disposition=thought.disposition,
        stance_disposition_reasons=thought.disposition_reasons,
        stance_boundary_register=bool(thought.boundary_register),
        settings=cfg,
    )

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
        answer_contract=AnswerContract(),
        repair_pressure_contract=_repair_pressure_contract(repair_bundle),
        fcc_model_label=payload.get("fcc_model_label") or DEFAULT_UNIFIED_TURN_FCC_MODEL_LABEL,
    )
    harness_bus = harness_rpc_bus or bus
    if harness_step_relay is not None and harness_step_queue is not None:
        harness_step_relay.register_queue(correlation_id, harness_step_queue)
    liveness_check = (
        (lambda within_sec: harness_step_relay.seen_recently(correlation_id, within_sec=within_sec))
        if harness_step_relay is not None
        else None
    )
    try:
        run = await HarnessGovernorClient(harness_bus).run(
            harness_req,
            correlation_id=correlation_id,
            liveness_check=liveness_check,
        )
    finally:
        if harness_step_relay is not None and harness_step_queue is not None:
            harness_step_relay.unregister_queue(correlation_id, harness_step_queue)
        if harness_step_relay is not None:
            harness_step_relay.forget(correlation_id)
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
    if run.finalize_degraded_reason and run.final_text:
        await _publish_unified_turn_chat_history(
            bus=bus,
            correlation_id=correlation_id,
            session_id=session_id,
            user_message=user_message,
            response_text=str(run.final_text),
            payload=payload,
            run=run,
            source_label=str(payload.get("chat_history_source") or "hub_orion"),
        )
        degraded_frame = {
            "type": "turn_degraded",
            "correlation_id": correlation_id,
            "reason": run.finalize_degraded_reason,
        }
        return [degraded_frame, *_success_frames(run, correlation_id=correlation_id)]
    if not run.finalize_ran or not run.final_text:
        return [_harness_error_frame(run, correlation_id=correlation_id)]
    await _publish_unified_turn_chat_history(
        bus=bus,
        correlation_id=correlation_id,
        session_id=session_id,
        user_message=user_message,
        response_text=str(run.final_text),
        payload=payload,
        run=run,
        source_label=str(payload.get("chat_history_source") or "hub_orion"),
    )
    return _success_frames(run, correlation_id=correlation_id)


async def _publish_unified_turn_chat_history(
    *,
    bus: Any,
    correlation_id: str,
    session_id: str | None,
    user_message: str,
    response_text: str,
    payload: dict[str, Any],
    run: HarnessRunV1,
    source_label: str = "hub_orion",
) -> None:
    """Orion capability: unified-turn persistence after successful handoff.

    Persists the finalized turn only after the governor returned final text:
    chat-history envelopes (so sql-writer lands chat_history_log rows), a
    chat-turn envelope, and the Spark introspection candidate, honoring
    no_write. When earlier phases fail none of this exists — the governor's
    run artifact is the evidence trail instead.

    Runtime evidence: chat_history_log rows, chat-turn envelopes, and the
    spark candidate carrying unified_turn metadata. Start here when a
    finalized answer reached the client but is missing from history or Spark.
    """
    if bus is None:
        return
    if payload.get("no_write") or payload.get("x_no_write"):
        return
    if not str(response_text or "").strip():
        return

    from scripts.chat_history import (
        build_chat_history_envelope,
        build_chat_turn_envelope,
        publish_chat_history,
        publish_chat_turn,
    )
    from scripts.settings import settings as hub_settings
    from scripts.spark_candidate import publish_spark_introspect_candidate

    session = str(session_id or "anonymous")
    user_id = payload.get("user_id")
    mode_tag = "orion"
    spark_meta = {
        "mode": mode_tag,
        "unified_turn": True,
        "harness_step_count": run.step_count,
        "harness_grounding_status": run.grounding_status,
    }

    reasoning_trace: dict[str, Any] | None = None
    if run.reflection is not None:
        reflection_bits = [
            str(run.reflection.imperative or "").strip(),
            *[str(note).strip() for note in (run.reflection.alignment_notes or []) if str(note).strip()],
        ]
        reflection_text = "\n".join(bit for bit in reflection_bits if bit)
        if reflection_text:
            reasoning_trace = {
                "trace_role": "reflection",
                "trace_stage": "post_answer",
                "content": reflection_text,
                "correlation_id": correlation_id,
                "session_id": session,
            }

    envelopes = [
        build_chat_history_envelope(
            content=user_message,
            role="user",
            session_id=session,
            correlation_id=correlation_id,
            speaker=str(user_id or "user"),
            tags=[mode_tag],
            message_id=f"{correlation_id}:user",
            memory_status="accepted",
            memory_tier="ephemeral",
        ),
        build_chat_history_envelope(
            content=response_text,
            role="assistant",
            session_id=session,
            correlation_id=correlation_id,
            speaker=hub_settings.SERVICE_NAME,
            tags=[mode_tag],
            message_id=f"{correlation_id}:assistant",
            reasoning_trace=reasoning_trace,
        ),
    ]
    await publish_chat_history(bus, envelopes)

    env_turn = build_chat_turn_envelope(
        prompt=user_message,
        response=response_text,
        session_id=session,
        correlation_id=correlation_id,
        user_id=str(user_id) if user_id else None,
        source_label=source_label,
        spark_meta=spark_meta,
        turn_id=correlation_id,
        memory_status="accepted",
        memory_tier="ephemeral",
        reasoning_trace=reasoning_trace,
        thinking_source="orion_unified_turn",
    )
    await publish_chat_turn(bus, env_turn)

    if hub_settings.PUBLISH_CHAT_HISTORY_LOG:
        try:
            await bus.publish(
                hub_settings.chat_history_channel,
                {
                    "correlation_id": correlation_id,
                    "source": source_label,
                    "prompt": user_message,
                    "response": response_text,
                    "session_id": session,
                    "mode": mode_tag,
                    "spark_meta": spark_meta,
                    "reasoning_trace": reasoning_trace,
                },
            )
        except Exception:
            logger.warning(
                "unified_turn legacy chat_history publish failed corr=%s",
                correlation_id,
                exc_info=True,
            )

    try:
        await publish_spark_introspect_candidate(
            bus,
            trace_id=correlation_id,
            prompt=user_message,
            response=response_text,
            spark_meta=spark_meta,
            source=source_label,
            correlation_id=correlation_id,
        )
    except Exception:
        logger.warning(
            "unified_turn spark candidate publish failed corr=%s",
            correlation_id,
            exc_info=True,
        )

    logger.info(
        "unified_turn chat_history published corr=%s session=%s source=%s",
        correlation_id,
        session,
        source_label,
    )


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
    harness_rpc_bus: Any | None = None,
    harness_step_relay: Any | None = None,
) -> list[dict[str, Any]]:
    """Execute unified turn and emit WS frames."""
    step_queue: asyncio.Queue | None = None
    drain_task: asyncio.Task | None = None
    if harness_step_relay is not None:
        step_queue = asyncio.Queue(maxsize=256)

        async def _drain_harness_steps() -> None:
            assert step_queue is not None
            try:
                while True:
                    frame = await step_queue.get()
                    outbound = frame
                    if with_biometrics is not None:
                        outbound = await with_biometrics(frame, cache=biometrics_cache)
                    await websocket.send_json(outbound)
            except asyncio.CancelledError:
                pass

        drain_task = asyncio.create_task(
            _drain_harness_steps(),
            name=f"harness-steps-{correlation_id}",
        )

    try:
        frames = await execute_unified_turn(
            bus=bus,
            correlation_id=correlation_id,
            session_id=session_id,
            user_message=user_message,
            payload=payload,
            continuity_messages=continuity_messages,
            harness_rpc_bus=harness_rpc_bus,
            harness_step_relay=harness_step_relay,
            harness_step_queue=step_queue,
        )
    finally:
        if harness_step_relay is not None and step_queue is not None:
            while not step_queue.empty():
                try:
                    frame = step_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                outbound = frame
                if with_biometrics is not None:
                    outbound = await with_biometrics(frame, cache=biometrics_cache)
                await websocket.send_json(outbound)
        if drain_task is not None:
            drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await drain_task
    for frame in frames:
        outbound = frame
        if with_biometrics is not None:
            outbound = await with_biometrics(frame, cache=biometrics_cache)
        await websocket.send_json(outbound)
    # Mirror the classic lane contract (websocket_handler emits {"state": "idle"} at end of
    # turn): the Hub status line is set to "Sent..." on send and only resets to "Ready." when
    # a frame carries state 'idle'. The unified terminal frames omit state, so emit a trailing
    # idle-state frame to unstick the status after the turn completes.
    idle_frame: dict[str, Any] = {"state": "idle"}
    if with_biometrics is not None:
        idle_frame = await with_biometrics(idle_frame, cache=biometrics_cache)
    await websocket.send_json(idle_frame)
    return frames
