from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Awaitable, Callable, Protocol

from orion.schemas.cognition.answer_contract import AnswerContract
from orion.hub.association import build_hub_association_bundle
from orion.hub.chat_route import CHAT_ROUTE_UNIFIED_TURN_HARNESS
from orion.hub.turn_request import build_orion_turn_request
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.situational.context import build_situation_for_ctx, hub_settings_to_runtime_namespace
from orion.harness.attachment_staging import prune_staging, stage_attachments
from orion.schemas.harness_finalize import (
    HARNESS_RECENT_TURNS_MAX,
    HarnessAttachmentV1,
    HarnessRunRequestV1,
    HarnessRunV1,
)
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


async def _publish_turn_timeout_grammar(
    *, bus: Any, correlation_id: str, settings: Any
) -> None:
    """Orion capability: liveness marker for a turn whose HarnessGovernorClient.run()
    call did not yield a usable result (the `run is None` case below).

    Most commonly a true harness-governor RPC timeout -- the ONE case in the
    unified-turn saga where no governor-side grammar event is ever published at all
    (HarnessGrammarCollector only flushes at the end of a run, a point a true timeout
    never reaches). `run is None` also covers a rarer reply-failed-to-decode sub-case
    where the governor's reply did arrive -- see build_turn_timeout_grammar_events'
    own docstring for the full caveat. Reuses PUBLISH_HUB_CHAT_GRAMMAR (no new env
    key, see services/orion-hub/README.md's note on this coupling) since this is still
    Hub-side grammar publishing onto the same channel. Fail-open, same shape as
    _publish_unified_turn_chat_grammar above.
    """
    if bus is None or not getattr(settings, "PUBLISH_HUB_CHAT_GRAMMAR", False):
        return
    try:
        from scripts.grammar_emit import build_turn_timeout_grammar_events
        from scripts.grammar_publish import publish_hub_chat_grammar_trace

        events = build_turn_timeout_grammar_events(
            correlation_id=correlation_id,
            node_id=settings.NODE_NAME,
        )
        await publish_hub_chat_grammar_trace(
            bus,
            events,
            correlation_id=correlation_id,
            channel=settings.GRAMMAR_EVENT_CHANNEL,
            enabled=True,
        )
    except Exception:
        logger.warning("turn_timeout_grammar_publish_failed corr=%s", correlation_id, exc_info=True)


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


def _success_frames(
    run: HarnessRunV1, *, correlation_id: str, fcc_model_label: str | None = None
) -> list[dict[str, Any]]:
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
    if fcc_model_label:
        # The identity that actually produced this response -- previously not
        # exposed on the frame at all, so callers that consume frames
        # directly instead of relying on _publish_unified_turn_chat_history's
        # own persistence (e.g. endogenous_outreach.py, which sets
        # no_write=True and does its own publish from the frame) had no
        # identity to report at all.
        #
        # NOTE this key is dual-purpose by caller intent, not by accident:
        # callers of _success_frames pass execute_unified_turn's
        # `resolved_model_label` (run.fcc_served_model when discovery fired,
        # else harness_req.fcc_model_label as fallback -- see that
        # computation above), so this value is USUALLY the real backend
        # model, not the small fixed set of ~/.fcc/.env route aliases
        # (e.g. "MODEL_SONNET") that HarnessRunRequestV1.fcc_model_label
        # itself always holds. Any new consumer reading this key off a frame
        # must not assume it is one of those aliases.
        final_frame["fcc_model_label"] = fcc_model_label
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
    from scripts.pre_turn_appraisal_wiring import (
        _publish_repair_pressure_appraisal,
        build_repair_pressure_summary,
    )

    turn_window = build_turn_window(
        continuity_messages or [{"role": "user", "content": user_message}]
    )
    paradigms = str(getattr(settings, "PRE_TURN_APPRAISAL_PARADIGMS", "repair_pressure"))
    timeout_ms = int(getattr(settings, "PRE_TURN_APPRAISAL_TIMEOUT_MS", 60000))
    bundle = await PreTurnAppraisalClient(bus).appraise(
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
    # Unified-turn (mode="orion"/harness-governor) is a second, independent
    # caller of PreTurnAppraisalClient alongside pre_turn_appraisal_wiring's
    # websocket/HTTP "brain" path. Both must publish, or
    # orion-equilibrium-service's relational metacog gate silently never
    # sees appraisals from whichever caller skips it -- confirmed live
    # 2026-07-21: this path was the one skipping it, so every real
    # orion-mode/journal turn since deploy had a computed repair_pressure
    # result that never reached the bus.
    summary = build_repair_pressure_summary(bundle)
    if summary is not None:
        await _publish_repair_pressure_appraisal(
            bus,
            correlation_id=correlation_id,
            summary=summary,
        )
    return bundle


async def _build_situation_prompt_fragment(
    *,
    session_id: str | None,
    user_message: str,
    payload: dict[str, Any],
    settings: Any,
    correlation_id: str,
) -> str | None:
    """Orion capability: local time-of-day/day-phase/conversation-phase/presence
    context for the unified-turn prompt.

    Root cause of the "Orion asked how my evening was going at 12:45pm"
    report (2026-08-22): this data was already fully built by
    `orion.situational.context.build_situation_for_ctx` -- correctly, hour
    12 buckets to `midday` not `evening` -- but that builder was only ever
    called from services/orion-cortex-exec's own legacy chat-verb dispatch
    lane, a DIFFERENT chat path from the one `execute_unified_turn`
    actually serves. Fail-open by design, same contract as every other
    situation provider: any failure here degrades to no situation context
    for this turn, never an exception into turn assembly.

    Runtime evidence: `HarnessRunRequestV1.situation_prompt_fragment` on the
    request this builds, and the "Situation:" block it produces in the
    compiled harness prefix (orion/harness/prefix.py::compile_harness_prefix).
    """
    try:
        situation_runtime_ns = hub_settings_to_runtime_namespace(settings)
        situation_ctx: dict[str, Any] = {
            "session_id": session_id or "anonymous",
            "raw_user_text": user_message,
        }
        presence_context = payload.get("presence_context")
        if not isinstance(presence_context, dict):
            # Fall back to the same stored per-session presence
            # (`/api/presence`'s PresenceContextStore) the classic
            # websocket_handler.py chat lane already merges in via
            # `inject_session_presence` -- confirmed live 2026-08-22 that the
            # unified-turn branch returns before that call ever runs, so a
            # presence set via `/api/presence` was silently invisible to
            # every Orion-mode turn. Best-effort: a stale/unimportable store
            # degrades to "no stored presence", not a failed turn.
            try:
                from scripts.presence_session import inject_session_presence
                import scripts.main as hub_main

                enriched = inject_session_presence(
                    payload, str(session_id or "anonymous"), getattr(hub_main, "presence_context_store", None)
                )
                presence_context = enriched.get("presence_context")
            except Exception:
                presence_context = None
        if isinstance(presence_context, dict):
            situation_ctx["presence_context"] = presence_context
        _, situation_fragment = await build_situation_for_ctx(situation_ctx, situation_runtime_ns)
        compact_text = situation_fragment.get("compact_text") if situation_fragment else None
        return str(compact_text) if compact_text else None
    except Exception:
        logger.warning("unified_turn_situation_context_failed corr=%s", correlation_id, exc_info=True)
        return None


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

    situation_prompt_fragment = await _build_situation_prompt_fragment(
        session_id=session_id,
        user_message=user_message,
        payload=payload,
        settings=cfg,
        correlation_id=correlation_id,
    )

    # Stage any attached images into the FCC sandbox BEFORE dispatch. The Hub is
    # the only container that mounts both the attachment store and the sandbox --
    # the harness-governor, which actually runs `claude`, cannot see the store at
    # all. So if this does not happen here, it cannot happen at all.
    staged_attachments: list[HarnessAttachmentV1] = []
    # Tracked separately from the list above: if model construction throws after
    # the copies already landed, `staged_attachments` is empty but there ARE
    # files on disk, and pruning on the empty list would leak them forever.
    staging_attempted = False
    raw_attachments = payload.get("attachments") or []
    if raw_attachments:
        staging_attempted = True
        try:
            # to_thread: shutil.copyfile of up to HUB_CHAT_ATTACHMENT_MAX_PER_TURN
            # x HUB_CHAT_ATTACHMENT_MAX_BYTES is real blocking I/O, and this Hub
            # serves every WebSocket client from one event loop.
            staged = await asyncio.to_thread(
                stage_attachments, raw_attachments, correlation_id=correlation_id
            )
            staged_attachments = [
                HarnessAttachmentV1(
                    path=item.path, mime=item.mime, sha256=item.sha256, filename=item.filename
                )
                for item in staged
            ]
        except Exception:  # noqa: BLE001 -- staging must never take the turn down
            logger.exception("attachment staging failed corr=%s", correlation_id)
            staged_attachments = []
        if not staged_attachments:
            # Say it out loud. A turn that quietly loses the image Juniper
            # attached is the exact failure this feature exists to prevent.
            logger.error(
                "attachments present but none staged corr=%s count=%s",
                correlation_id,
                len(raw_attachments),
            )
        else:
            logger.info(
                "unified turn carrying %s attachment(s) corr=%s",
                len(staged_attachments),
                correlation_id,
            )

    # Reuses build_turn_window (already the appraisal call above's normalizer)
    # rather than passing continuity_messages through raw -- caps to the last
    # HARNESS_RECENT_TURNS_MAX messages regardless of what the caller already
    # capped, so HarnessRunRequestV1.recent_turns's own bound holds even if a
    # future caller forgets to cap. No synthetic single-message fallback here
    # (unlike _run_pre_turn_appraisal's turn_window above): an empty
    # continuity_messages should render as a genuinely empty recent_turns, not
    # a fabricated one-line "history".
    #
    # Real callers (services/orion-hub/scripts/api_routes.py's build_continuity_
    # messages(history=user_messages, ...) and websocket_handler.py's history
    # list) both include THIS turn's own message as the trailing entry --
    # confirmed live review 2026-08-20: without stripping it here, recent_turns'
    # last item duplicates user_message, so compile_harness_prefix would render
    # the current question twice (once under RECENT CONVERSATION, once as
    # "User message: ..."). _run_pre_turn_appraisal's turn_window above is left
    # alone -- that RPC may legitimately want the current message included.
    history_only = list(continuity_messages or [])
    if history_only:
        last = history_only[-1]
        last_content = str(last.get("content") or "").strip() if isinstance(last, dict) else None
        if last_content is not None and last_content == user_message.strip():
            history_only = history_only[:-1]
    recent_turns = build_turn_window(history_only, max_turns=HARNESS_RECENT_TURNS_MAX)

    harness_req = HarnessRunRequestV1(
        correlation_id=correlation_id,
        thought_event=thought,
        user_message=user_message,
        attachments=staged_attachments,
        recent_turns=recent_turns,
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
        situation_prompt_fragment=situation_prompt_fragment,
    )
    harness_bus = harness_rpc_bus or bus
    if harness_step_relay is not None and harness_step_queue is not None:
        harness_step_relay.register_queue(correlation_id, harness_step_queue)
    liveness_check = (
        (lambda within_sec: harness_step_relay.seen_recently(correlation_id, within_sec=within_sec))
        if harness_step_relay is not None
        else None
    )
    _harness_run_completed = False
    try:
        run = await HarnessGovernorClient(harness_bus).run(
            harness_req,
            correlation_id=correlation_id,
            liveness_check=liveness_check,
        )
        # `run is None` means an RPC timeout with no cancel published -- the
        # motor may still be mid-turn, so this stays False and we leave the
        # staged files alone rather than yanking them from a live reader.
        _harness_run_completed = run is not None
    finally:
        if harness_step_relay is not None and harness_step_queue is not None:
            harness_step_relay.unregister_queue(correlation_id, harness_step_queue)
        if harness_step_relay is not None:
            harness_step_relay.forget(correlation_id)
        # Staged images are per-turn scratch; without cleanup they accumulate in
        # the sandbox forever. The bytes still live in the content-addressed
        # store, so a follow-up turn about the same image just re-stages it.
        #
        # Only prune when the run actually COMPLETED. This finally also fires on
        # an RPC timeout (HarnessGovernorClient.run returns None without
        # publishing a cancel) and on CancelledError from
        # run_awaitable_cancel_on_ws_disconnect -- in both cases the governor's
        # claude process is still going, and deleting the directory out from
        # under it turns a not-yet-issued Read into file-not-found.
        if staging_attempted and _harness_run_completed:
            await asyncio.to_thread(prune_staging, correlation_id)
    if run is None:
        await _publish_turn_timeout_grammar(bus=bus, correlation_id=correlation_id, settings=cfg)
        return [
            {
                "type": "turn_error",
                "phase": "harness",
                "correlation_id": correlation_id,
                "finalize_ran": False,
                "error": "harness_rpc_timeout",
            }
        ]
    # Prefer the real backend model HarnessRunner discovered from the CLI's own
    # stream-json events (run.fcc_served_model -- see orion/harness/fcc_motor.py's
    # _served_model_from_assistant) over the requested ~/.fcc/.env route alias
    # (harness_req.fcc_model_label, e.g. "MODEL_SONNET"). The alias is the only
    # thing known before dispatch, but confirmed live 2026-08-19 that MODEL_SONNET
    # and MODEL_OPUS both point at the identical llamacpp/chat route, so it can't
    # tell backends apart. Falls back to the alias when discovery never fired
    # (e.g. the motor failed before any assistant turn).
    resolved_model_label = run.fcc_served_model or harness_req.fcc_model_label
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
            fcc_model_label=resolved_model_label,
        )
        degraded_frame = {
            "type": "turn_degraded",
            "correlation_id": correlation_id,
            "reason": run.finalize_degraded_reason,
        }
        return [
            degraded_frame,
            *_success_frames(run, correlation_id=correlation_id, fcc_model_label=resolved_model_label),
        ]
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
        fcc_model_label=resolved_model_label,
    )
    return _success_frames(run, correlation_id=correlation_id, fcc_model_label=resolved_model_label)


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
    fcc_model_label: str | None = None,
) -> None:
    """Orion capability: unified-turn persistence after successful handoff.

    Persists the finalized turn only after the governor returned final text:
    chat-history envelopes (so sql-writer lands chat_history_log rows) and a
    chat-turn envelope, honoring no_write. When earlier phases fail none of
    this exists — the governor's run artifact is the evidence trail instead.

    Runtime evidence: chat_history_log rows and chat-turn envelopes. Start
    here when a finalized answer reached the client but is missing from
    history.

    2026-07-28 (spark-introspector retirement): the Spark introspection
    candidate publish this function used to also send was removed. Its sole
    purpose was feeding orion-spark-introspector's Cognitive EKG; the
    chat-turn envelope published just above already feeds
    orion-vector-host's real OrionTissue physics feed
    (services/orion-vector-host/app/tissue_feed.py) in-process, so no
    separate candidate event is needed anymore.
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

    session = str(session_id or "anonymous")
    user_id = payload.get("user_id")
    mode_tag = "orion"
    spark_meta = {
        "mode": mode_tag,
        "unified_turn": True,
        "harness_step_count": run.step_count,
        "harness_grounding_status": run.grounding_status,
        "chat_route": CHAT_ROUTE_UNIFIED_TURN_HARNESS,
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
            model=fcc_model_label,
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
        response_identity=fcc_model_label,
        source_label=source_label,
        spark_meta=spark_meta,
        turn_id=correlation_id,
        memory_status="accepted",
        memory_tier="ephemeral",
        reasoning_trace=reasoning_trace,
        thinking_source="orion_unified_turn",
    )
    await publish_chat_turn(bus, env_turn)

    # 2026-08-14: the third, raw-dict legacy publish to `chat_history_channel`
    # was deleted here (the WS twin of the one in
    # services/orion-hub/scripts/api_routes.py). Published with no `kind`, so
    # `orion/core/bus/codec.py:72` stamped it `legacy.message` -- no sql-writer
    # route matched, so every one landed in `bus_fallback_log` and was written
    # nowhere else. The calls above already carry this turn as registered
    # envelopes: `publish_chat_history` sends two `chat.history.message.v1` on
    # this same log channel, and `publish_chat_turn` sends one `chat.history` on
    # the separate *turn* channel (`orion:chat:history:turn`). Note the log
    # channel therefore no longer carries a prompt/response pairing at all --
    # only the two independent message envelopes. The turn channel is where
    # paired turn data lives.
    # Note: do NOT "fix" this by flipping PUBLISH_CHAT_HISTORY_LOG -- that flag
    # also gates those two real publishes (services/orion-hub/scripts/
    # chat_history.py:288 and :392), so turning it off kills real persistence.

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
