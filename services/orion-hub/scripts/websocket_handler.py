# services/orion-hub/scripts/websocket_handler.py
from __future__ import annotations

import asyncio
import time
import base64
import json
import logging
import os
import sys
import uuid
from typing import Any, Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from scripts.settings import settings
from scripts.cortex_request_builder import (
    HubRequestValidationError,
    build_chat_request,
    build_continuity_messages,
    validate_single_verb_override,
)
from scripts.biometrics_cache import BiometricsCache
from scripts.chat_history import (
    build_chat_history_envelope,
    publish_chat_history,
    build_chat_turn_envelope,
    publish_social_room_turn,
    publish_chat_turn,
    select_reasoning_trace_for_history,
)
from scripts.social_room import (
    apply_social_memory_summary_to_payload,
    hub_direct_room_identity,
    is_social_room_payload,
    social_room_client_meta,
)
from scripts import social_room_inspection_cache
from scripts.cortex_chat_display import hub_effective_chat_text
from scripts.context_exec_agent_bridge import run_hub_agent_via_context_exec, should_use_context_exec_agent_lane
from scripts.agent_claude_input import prepare_agent_claude_input
from scripts.fcc_claude_bridge import (
    active_turns,
    build_harness_reasoning_trace,
    context_overflow_operator_hint,
    is_context_overflow_text,
    run_turn_from_settings,
)
from scripts.turn_cancel import cancel_in_flight_turn, run_awaitable_cancel_on_ws_disconnect
from scripts.settings import settings
from orion.fcc.sandbox_sync import record_sync_skip, sync_fcc_sandbox
from scripts.fcc_model_mapping import DEFAULT_FCC_MODEL_LABEL
from scripts.trace_payloads import extract_agent_trace_payload
from scripts.voice_stt_errors import (
    build_audio_debug,
    empty_transcript_error_message,
    sanitize_client_audio_meta,
)
from scripts.autonomy_payloads import extract_autonomy_payload, log_autonomy_payload_extraction
from scripts.workflow_payloads import extract_workflow_payload
from scripts.mutation_cognition_context import build_mutation_cognition_context
from scripts.presence_session import inject_session_presence
from scripts import chat_turn_affect
from scripts.substrate_effect_pipeline import run_substrate_effect_pipeline
from scripts.repair_pressure_wiring import attach_repair_pressure_contract
from scripts.warm_start import mini_personality_summary
from orion.schemas.cortex.contracts import CortexChatRequest, CortexChatResult
from orion.schemas.metacognitive_trace import MetacognitiveTraceV1
from orion.schemas.tts import TTSRequestPayload, TTSResultPayload, STTRequestPayload, STTResultPayload
from orion.cognition.verb_activation import is_active

logger = logging.getLogger("orion-hub.ws")

# Registry of the turn each live WS connection currently owns, keyed by a
# per-connection id (NOT session_id — session_id is persisted client-side in
# localStorage and shared across every browser tab on the same origin, so keying
# on it would let a "stop" click in one tab cancel another tab's turn). The main
# receive loop below blocks awaiting the in-flight turn between
# websocket.receive_text() calls, so a same-connection "stop" message would sit
# unread until the turn already finished — a stop command needs a side channel.
# The HTTP cancel endpoint (api_routes.api_chat_turn_cancel) looks a connection up
# here and reuses the same cancel_in_flight_turn() path WS-disconnect already uses.
#
# Each entry is the connection's own `active_turn` dict, stored by reference and
# registered once at connection setup — mutating `active_turn` in the main loop
# (as it already does) is automatically visible here with no separate
# register/clear calls to keep in sync at every turn site.
_ACTIVE_TURNS_BY_CONNECTION: Dict[str, Dict[str, Optional[str]]] = {}


async def cancel_active_turn_for_connection(
    connection_id: str, *, bus: Any, reason: str = "user_stop"
) -> Optional[str]:
    """Cancel whichever turn `connection_id` currently owns. Returns the cancelled
    correlation_id, or None if that connection has no turn in flight.
    """
    entry = _ACTIVE_TURNS_BY_CONNECTION.get(str(connection_id or "").strip())
    if not entry or not entry.get("correlation_id"):
        return None
    corr = str(entry["correlation_id"])
    kind = str(entry.get("kind") or "orion")
    await cancel_in_flight_turn(bus=bus, correlation_id=corr, kind=kind, reason=reason)
    return corr


async def _safe_ws_send_json(websocket: WebSocket, payload: Any) -> bool:
    """Send JSON only if the socket is still open; avoids RuntimeError after client disconnect."""
    if websocket.client_state != WebSocketState.CONNECTED:
        logger.warning(
            "ws_send_json_skipped reason=not_connected client_state=%s application_state=%s",
            websocket.client_state,
            getattr(websocket, "application_state", None),
        )
        return False
    try:
        await websocket.send_json(payload)
        return True
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "send" in msg and ("close" in msg or "disconnect" in msg or "not connected" in msg):
            logger.warning("ws_send_json_skipped runtime_error=%s", exc)
            return False
        raise


def _thought_debug_enabled() -> bool:
    return str(os.getenv("DEBUG_THOUGHT_PROCESS", "false")).strip().lower() in {"1", "true", "yes", "on"}


def _debug_len(value: Any) -> int:
    return len(str(value or ""))


def _debug_snippet(value: Any, max_len: int = 200) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}…"


def _preview_text(value: str | None, limit: int = 220) -> str:
    if not value:
        return ""
    return repr(value[:limit])


def _coerce_metacog_trace(trace: Any) -> Optional[MetacognitiveTraceV1]:
    if isinstance(trace, MetacognitiveTraceV1):
        return trace
    if isinstance(trace, dict):
        try:
            return MetacognitiveTraceV1.model_validate(trace)
        except Exception:
            return None
    return None


#________________________
# store chat turns
#________________________

def _normalize_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


async def _apply_hub_direct_social_room_mode(data: Dict[str, Any]) -> Dict[str, Any]:
    """Populate chat_profile/continuity fields for the Hub UI's own social_room toggle.

    Hub-local only: does not invoke external room delivery or third-party chat bridges.
    Mirrors the social-memory prefetch pattern (identity + GET /summary) but stops at
    build_chat_request() — reply delivery stays in the Hub websocket UI.

    The write-back half (publish_social_room_turn -> orion-sql-writer ->
    orion-social-memory) fires automatically off chat_profile=="social_room".
    """
    if str(data.get("social_room_mode") or "").strip().lower() != "hub_direct":
        return data
    identity = hub_direct_room_identity(data.get("user_id"))
    enriched = dict(data)
    enriched["chat_profile"] = "social_room"
    enriched["social_room_mode"] = "hub_direct"
    enriched.setdefault("external_room", identity["external_room"])
    enriched.setdefault("external_participant", identity["external_participant"])
    posture = str(data.get("social_redaction_posture") or "").strip().lower()
    if posture in ("strict", "relaxed"):
        enriched["social_redaction_posture"] = posture
    try:
        from scripts.api_routes import _fetch_social_memory

        summary = await _fetch_social_memory(
            "/summary",
            {
                "platform": identity["external_room"]["platform"],
                "room_id": identity["external_room"]["room_id"],
                "participant_id": identity["external_participant"]["participant_id"],
            },
        )
    except Exception as exc:
        logger.warning("hub_direct_social_memory_fetch_failed error=%s", exc)
        return enriched
    return apply_social_memory_summary_to_payload(enriched, summary)


def _log_hub_route_decision(
    *,
    corr_id: str,
    session_id: str,
    route_debug: Dict[str, Any],
    user_prompt: str,
) -> None:
    emitted_mode = route_debug.get("mode")
    emitted_verb = route_debug.get("verb")
    effective_verb = emitted_verb
    if not effective_verb and emitted_mode not in {"agent", "council"}:
        effective_verb = "chat_general"
    summary = {
        "corr_id": corr_id,
        "session_id": session_id,
        "selected_ui_route": route_debug.get("selected_ui_route"),
        "emitted_mode": emitted_mode,
        "emitted_verb": emitted_verb,
        "effective_verb": effective_verb,
        "emitted_options": route_debug.get("options") or {},
        "packs": route_debug.get("packs") or [],
        "force_agent_chain": bool(route_debug.get("force_agent_chain")),
        "supervised": bool(route_debug.get("supervised")),
        "diagnostic": bool(route_debug.get("diagnostic")),
        "last_user_head": (user_prompt or "")[:120],
    }
    logger.info("hub_route_egress %s", json.dumps(summary, sort_keys=True, default=str))




def _truncate_text(value: Any, limit: int = 800) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _compact_council_debug(payload: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    opinions_out = []
    opinions = payload.get("opinions")
    if isinstance(opinions, list):
        for item in opinions[:12]:
            if not isinstance(item, dict):
                continue
            opinions_out.append({
                "agent_name": _truncate_text(item.get("agent_name") or item.get("name") or "unknown", 80),
                "confidence": item.get("confidence"),
                "text": _truncate_text(item.get("text") or "", 800),
            })

    verdict_out = {}
    verdict = payload.get("verdict")
    if isinstance(verdict, dict):
        verdict_out = {
            "action": verdict.get("action"),
            "reason": _truncate_text(verdict.get("reason") or "", 500),
            "constraints": verdict.get("constraints") if isinstance(verdict.get("constraints"), dict) else {},
        }

    blink_out = {}
    blink = payload.get("blink")
    if isinstance(blink, dict):
        blink_out = {
            "proposed_answer": _truncate_text(blink.get("proposed_answer") or "", 500),
            "scores": blink.get("scores") if isinstance(blink.get("scores"), dict) else {},
        }

    if not opinions_out and not verdict_out and not blink_out:
        return None
    return {"opinions": opinions_out, "verdict": verdict_out, "blink": blink_out}


def _extract_council_debug_from_result(resp: CortexChatResult) -> Dict[str, Any] | None:
    if not resp or not getattr(resp, "cortex_result", None):
        return None

    cr = resp.cortex_result
    recall_debug = cr.recall_debug if isinstance(cr.recall_debug, dict) else {}
    metadata = cr.metadata if isinstance(cr.metadata, dict) else {}

    for candidate in (
        recall_debug.get("council_debug"),
        metadata.get("council"),
        metadata.get("council_debug"),
    ):
        compact = _compact_council_debug(candidate if isinstance(candidate, dict) else None)
        if compact:
            return compact

    steps = cr.steps if isinstance(cr.steps, list) else []
    for step in reversed(steps):
        step_result = getattr(step, "result", None)
        if not isinstance(step_result, dict):
            continue
        council_payload = step_result.get("CouncilService")
        if not isinstance(council_payload, dict):
            continue
        compact = _compact_council_debug(council_payload.get("debug_compact") if isinstance(council_payload.get("debug_compact"), dict) else council_payload)
        if compact:
            return compact

    return None

def _schedule_publish(coro: asyncio.Future, label: str) -> None:
    task = asyncio.create_task(coro)

    def _log_result(t: asyncio.Task) -> None:
        try:
            t.result()
        except Exception as exc:
            logger.warning("Failed to publish %s: %s", label, exc, exc_info=True)

    task.add_done_callback(_log_result)


def _rec_tape_req(
    *,
    corr_id: str,
    session_id: Optional[str],
    mode: str,
    use_recall: bool,
    recall_profile: Optional[str],
    user_head: str,
    no_write: bool,
) -> None:
    if not settings.HUB_DEBUG_RECALL:
        return
    logger.info(
        "REC_TAPE REQ corr_id=%s sid=%s mode=%s recall=%s profile=%s user_head=%r no_write=%s",
        corr_id,
        session_id,
        mode,
        use_recall,
        recall_profile,
        user_head,
        no_write,
    )


def _rec_tape_rsp(
    *,
    corr_id: str,
    memory_used: bool,
    recall_count: int,
    backend_counts: Dict[str, Any] | None,
    memory_digest: Optional[str],
) -> None:
    if not settings.HUB_DEBUG_RECALL:
        return
    digest_chars = len(memory_digest or "")
    logger.info(
        "REC_TAPE RSP corr_id=%s memory_used=%s digest_chars=%s recall_count=%s backend_counts=%s",
        corr_id,
        memory_used,
        digest_chars,
        recall_count,
        backend_counts or {},
    )


def _build_prompt_with_history(
    history: List[Dict[str, Any]],
    user_text: str,
    turns: int,
    max_chars: int,
) -> str:
    """Build a single prompt string that includes the last N turns as plain text.

    This is intentionally *Hub-side only* and does not require any schema changes.

    Notes:
      - "turns" means userassistant pairs; we keep up to 2*turns messages.
      - We exclude system messages (the backend already has its own system prompt).
    """
    msgs = [m for m in history if m.get("role") in ("user", "assistant")]

    # "turns" = userassistant pairs -> 2*turns messages
    tail = msgs[-2 * max(0, int(turns)) :] if turns else []

    lines: List[str] = []
    for m in tail:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        speaker = "You" if role == "user" else "Orion"
        # Avoid accidental mega-prompts if something goes sideways
        if len(content) > 6000:
            content = content[:6000].rstrip() + " …"

        lines.append(f"{speaker}: {content}")

    base_user = (user_text or "").strip()
    if not lines:
        return base_user

    header = "Conversation context (most recent last):"

    def _compose(ls: List[str]) -> str:
        ctx = "\n".join(ls).strip()
        return f"{header}\n{ctx}\n\nYou: {base_user}\nOrion:"

    prompt = _compose(lines)

    # Trim oldest context lines until we fit under max_chars
    if max_chars and max_chars > 0:
        while len(prompt) > max_chars and lines:
            lines.pop(0)
            prompt = _compose(lines)

    return prompt



async def _with_biometrics(
    payload: Dict[str, Any],
    *,
    cache: Optional[BiometricsCache],
) -> Dict[str, Any]:
    """No longer attaches a `biometrics` key.

    This used to enrich every outgoing websocket message with a cache snapshot
    (lock acquisition + dict copies + per-node snapshot construction, every
    call) for the client's `updateBiometricsPanel()`/`d.biometrics` reader.
    That reader -- and the `#biometricsPanel` widget it drove -- was removed
    in the same change that stopped attaching this key: `rg -n '\\.biometrics\\b'
    services/orion-hub/static/js/` has zero matches. `biometrics-view.js`'s
    replacement (the EKG-card preview + modal) polls `/api/biometrics/preview/*`
    over plain HTTP instead, so nothing on the client reads this field anymore.

    Kept as a passthrough rather than removed outright, and `cache` kept
    unused rather than threaded out: this function is still called at 20+
    sites across this module, several through multi-level parameter passing
    (e.g. `drain_queue`) -- unwinding that threading is a separate, larger,
    real-time-path-risk cleanup better done as its own change, not folded
    into a widget swap. `biometrics_heartbeat`, the one caller whose entire
    job was this enrichment (a periodic `{"biometrics_tick": True}` push with
    no other purpose), was fully removed in this same change instead, since
    keeping an infinite per-connection send loop running for a payload key
    nobody reads is not a defensible passthrough.
    """
    return dict(payload)


async def _rehydrate_connection_history(
    history: List[Dict[str, Any]], *, session_id: Any
) -> int:
    """Restore this socket's conversation context from persisted turns.

    Best-effort: a failure here leaves the pre-existing behaviour (start cold),
    so it can never break a connection.
    """
    try:
        from scripts.chat_history_rehydrate import rehydrate_history

        return await rehydrate_history(
            history,
            session_id=session_id,
            max_turns=int(getattr(settings, "HUB_CONTEXT_TURNS", 10)),
            max_age_hours=float(getattr(settings, "HUB_HISTORY_REHYDRATE_MAX_AGE_HOURS", 48.0)),
            enabled=bool(getattr(settings, "HUB_HISTORY_REHYDRATE_ENABLED", True)),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("history_rehydrate_failed session=%s err=%s", session_id, exc)
        return 0


async def drain_queue(websocket: WebSocket, queue: asyncio.Queue, cache: Optional[BiometricsCache]):
    try:
        while websocket.client_state.name == "CONNECTED":
            msg = await queue.get()
            try:
                await websocket.send_json(await _with_biometrics(msg, cache=cache))
            except WebSocketDisconnect:
                break
            queue.task_done()
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"drain_queue error: {e}", exc_info=True)

def extract_unified_turn_final_text(frames: list[dict]) -> Optional[str]:
    """Pulls the real assistant reply text out of a run_unified_turn()
    frame list, for the orion-lane TTS trigger below.

    Deliberately the "final" frame's `llm_response` ONLY -- not a
    turn_error frame's `partial_draft`. A partial draft is real
    assistant-authored text (the browser does render it as an Orion
    bubble), but speaking an error-path partial aloud is a different,
    untested product decision, not folded in here.

    Frame order is not assumed: `_success_frames` in turn_orchestrator.py
    can prepend substrate_appraisal/reflection frames before the final one,
    so this scans for `type == "final"` rather than trusting frames[-1].
    A frame list with no "final" frame (turn_deferred/turn_error/
    turn_degraded-only, or empty) yields None, same as "nothing to speak".
    """
    for frame in frames:
        if frame.get("type") == "final":
            text = frame.get("llm_response")
            return text if isinstance(text, str) and text.strip() else None
    return None


# Strong references to fire-and-forget TTS synthesis tasks. Review finding,
# 2026-08-27: asyncio holds only a WEAK reference to a running task -- both
# TTS dispatch call sites below discard create_task's return value, which is
# exactly the shape that lets a task be garbage-collected mid-synthesis with
# nothing surfaced (documented asyncio behavior; same class of bug already
# fixed the same way in services/orion-whisper-tts/app/cuda_watchdog.py's
# own _INFLIGHT set, PR #1901, same day). Self-trimming via the done
# callback, so this never grows unbounded.
_TTS_DISPATCH_INFLIGHT: set[asyncio.Task] = set()


def dispatch_tts_reply(
    *,
    text: Optional[str],
    disable_tts: bool,
    tts_client,
    tts_q: asyncio.Queue,
    correlation_id: str,
    session_id: Optional[str],
    lane: str,
    extra_gate: bool = True,
    log_extra: Optional[dict] = None,
) -> bool:
    """Shared TTS gate + fire-and-forget dispatch, used by BOTH the classic
    lane and the orion (unified-turn) lane.

    Review finding, 2026-08-27: before this, each lane hand-rolled its own
    copy of this ~15-line gate. That duplication is exactly the shape that
    produced the bug this whole file's orion-lane TTS wiring fixes in the
    first place -- one lane had the logic, the other didn't, and nothing
    enforced they stay in sync. One shared function means a future change
    to the gate (a rate limit, a content filter, a new condition) cannot be
    applied to one lane and silently forgotten in the other.

    `extra_gate` folds in a lane-specific condition (the classic lane's own
    `not workflow_metadata_only`) without this function needing to know
    what that concept is. `log_extra` lets a caller add lane-specific
    fields to the decision log line (again: classic lane's
    `workflow_metadata_only` value) without hardcoding classic-lane
    vocabulary here.

    Returns whether a synthesis task was actually dispatched -- the classic
    lane's own tts_debug payload branch needs this to decide whether to
    tell the browser "no voice reply is coming."
    """
    # .strip() truthiness, not bare truthiness: a whitespace-only string is
    # non-empty (Python truthy) but is not real text to speak. Caught by
    # this module's own tests while unifying the two lanes' gates -- the
    # classic lane's ORIGINAL, pre-existing gate had this same gap (`bool
    # (orion_response_text and ...)` with no strip), just never exercised
    # by a whitespace-only real response. Fixed for both lanes at once here
    # since they now share this one gate.
    has_real_text = bool(text and text.strip())
    will_tts = bool(has_real_text and extra_gate and not disable_tts and tts_client)
    log_fields = {
        "corr": correlation_id,
        "sid": session_id,
        "response_len": len(text or ""),
        "disable_tts": disable_tts,
        "has_tts_client": bool(tts_client),
        "will_tts": will_tts,
        "lane": lane,
    }
    if log_extra:
        log_fields.update(log_extra)
    logger.info(
        "voice.tts.decision " + " ".join(f"{k}=%s" for k in log_fields),
        *log_fields.values(),
    )
    if not will_tts:
        return False
    task = asyncio.create_task(
        run_tts_remote(
            text, tts_client, tts_q,
            correlation_id=correlation_id, session_id=session_id,
        )
    )
    _TTS_DISPATCH_INFLIGHT.add(task)
    task.add_done_callback(_TTS_DISPATCH_INFLIGHT.discard)
    return True


async def synthesize_tts_reply(
    text: str,
    tts_client,
    *,
    timeout_sec: float,
    lane: str,
    correlation_id: str = "-",
    session_id: str = "-",
) -> dict:
    """The shared TTS synthesis core: build the request, await `speak()`
    under a timeout, and return a plain dict shaped for merging into
    EITHER a queue message (the WS lane, via `run_tts_remote` below) or an
    HTTP JSON response body (`api_routes.py`'s HTTP-fallback chat route) --
    `{"audio_response":..., "tts_source_text":..., "tts_meta":...}` on
    success, `{"tts_error": "..."}` on failure, `{}` when there is nothing
    to synthesize (no text, or no configured client).

    Review finding, 2026-08-27: the HTTP-fallback fix originally hand-rolled
    a THIRD independent copy of this exact logic -- the same "one lane
    wired, one lane not, nothing keeping them in sync" shape that
    `dispatch_tts_reply` (same file) was built earlier the same day to stop
    happening for the will_tts *gate*. This is that same convergence
    applied to the *synthesis* itself: one place emits
    `voice.tts.start`/`voice.tts.done`/`voice.tts.error`, in one format,
    regardless of which lane calls it.

    Deliberately does NOT own the will_tts gate (disable_tts, tts_client
    truthiness, any lane-specific extra condition) -- that decision differs
    enough in shape between a fire-and-forget WS dispatch and a synchronous
    HTTP await that forcing it into one function added more indirection
    than it removed. Callers gate; this only synthesizes.
    """
    if not text or not text.strip() or not tts_client:
        return {}
    logger.info(
        "voice.tts.start lane=%s corr=%s sid=%s text_len=%d",
        lane,
        correlation_id,
        session_id,
        len(text),
    )
    try:
        req = TTSRequestPayload(text=text)
        result: TTSResultPayload = await asyncio.wait_for(
            tts_client.speak(req),
            timeout=timeout_sec,
        )
        if not result.audio_b64:
            # Review finding, 2026-08-27: a "successful" call with an empty
            # clip is a silent voice-drop with zero trace otherwise --
            # nothing raised, nothing queued/returned, the UI does nothing.
            # Treated as a real failure, not a quiet no-op.
            logger.warning(
                "voice.tts.empty_result lane=%s corr=%s sid=%s text_len=%d",
                lane,
                correlation_id,
                session_id,
                len(text),
            )
            return {"tts_error": "TTS returned no audio"}
        logger.info(
            "voice.tts.done lane=%s corr=%s sid=%s text_len=%d audio_b64_len=%d "
            "content_type=%s duration_sec=%s metadata=%s",
            lane,
            correlation_id,
            session_id,
            len(text),
            len(result.audio_b64),
            result.content_type,
            result.duration_sec,
            result.metadata,
        )
        return {
            "audio_response": result.audio_b64,
            # Not `text` — the UI treats d.text like llm_response and would duplicate the bubble.
            "tts_source_text": text,
            "tts_meta": {
                "content_type": result.content_type,
                "duration_sec": result.duration_sec,
                "metadata": result.metadata,
            },
        }
    except asyncio.TimeoutError:
        err = f"TTS timed out after {timeout_sec}s"
        logger.error("voice.tts.error lane=%s corr=%s sid=%s %s", lane, correlation_id, session_id, err)
        return {"tts_error": err}
    except Exception as e:
        err = str(e) or "TTS synthesis failed"
        logger.error(
            "voice.tts.error lane=%s corr=%s sid=%s %s", lane, correlation_id, session_id, err, exc_info=True
        )
        return {"tts_error": err}


async def run_tts_remote(
    text: str,
    tts_client,
    queue: asyncio.Queue,
    *,
    correlation_id: str = "-",
    session_id: str = "-",
):
    result = await synthesize_tts_reply(
        text,
        tts_client,
        timeout_sec=float(settings.HUB_TTS_TIMEOUT_SEC),
        lane="ws",
        correlation_id=correlation_id,
        session_id=session_id,
    )
    if not result:
        return
    msg = {**result, "state": "speaking" if "audio_response" in result else "idle"}
    if "tts_error" in result:
        msg["text"] = text
    await queue.put(msg)


def _agent_claude_enabled() -> bool:
    return bool(getattr(settings, "HUB_AGENT_CLAUDE_ENABLED", False))


async def _run_agent_claude_turn_ws(
    *,
    websocket: WebSocket,
    data: Dict[str, Any],
    transcript: str,
    trace_id: str,
    biometrics_cache: Any,
) -> Optional[Dict[str, Any]]:
    """Run FCC harness turn; stream claude_step frames. Returns final payload or None if error sent."""
    if not _agent_claude_enabled():
        await websocket.send_json(
            await _with_biometrics(
                {
                    "error": "Agent Claude mode is disabled",
                    "error_code": "agent_claude_disabled",
                    "mode": "agent-claude",
                    "correlation_id": trace_id,
                },
                cache=biometrics_cache,
            )
        )
        return None

    turn = prepare_agent_claude_input(transcript)
    fcc_label = str(data.get("fcc_model_label") or DEFAULT_FCC_MODEL_LABEL).strip() or DEFAULT_FCC_MODEL_LABEL

    final_text = ""
    final_meta: Dict[str, Any] = {}
    harness_steps: List[Dict[str, Any]] = []

    async def _consume() -> Optional[Dict[str, Any]]:
        nonlocal final_text, final_meta
        async for event in run_turn_from_settings(
            prompt=turn.prompt,
            fcc_model_label=fcc_label,
            correlation_id=trace_id,
        ):
            etype = str(event.get("type") or "")
            if etype == "step":
                step = event.get("step") if isinstance(event.get("step"), dict) else {}
                harness_steps.append(step)
                await websocket.send_json(
                    await _with_biometrics(
                        {
                            "kind": "claude_step",
                            "correlation_id": trace_id,
                            "mode": "agent-claude",
                            "step": step,
                        },
                        cache=biometrics_cache,
                    )
                )
            elif etype == "error":
                partial = str(event.get("llm_response") or "")
                await websocket.send_json(
                    await _with_biometrics(
                        {
                            "error": str(event.get("error") or "agent-claude failed"),
                            "error_code": str(event.get("error_code") or "fcc_claude_nonzero_exit"),
                            "mode": "agent-claude",
                            "correlation_id": trace_id,
                            "llm_response": partial or None,
                            "metadata": event.get("metadata"),
                        },
                        cache=biometrics_cache,
                    )
                )
                return None
            elif etype == "final":
                final_text = str(event.get("llm_response") or "")
                final_meta = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        return {
            "llm_response": final_text,
            "fcc_model_label": fcc_label,
            "metadata": final_meta,
            "harness_steps": harness_steps,
        }

    return await run_awaitable_cancel_on_ws_disconnect(
        websocket,
        _consume(),
        bus=None,
        correlation_id=trace_id,
        kind="agent-claude",
    )


_fcc_sandbox_sync_lock = asyncio.Lock()


async def _sync_fcc_sandbox_background(connection_id: str) -> None:
    """Best-effort, serialized, turn-aware wrapper around ``sync_fcc_sandbox``.

    Runs as a detached task from the WS connect handler (see its call site) so a
    slow/unreachable origin can't delay connection readiness. Serialized via
    ``_fcc_sandbox_sync_lock`` so two connects arriving close together can't race
    each other's git invocations on the same shared sandbox path, and skipped
    entirely while any FCC turn is in flight (``active_turns()``) so a reset/clean
    never runs against a workspace a claude subprocess is still reading/writing.
    """
    # A test process must never mutate the live sandbox. Not hypothetical: the
    # hub suite drives the real websocket_endpoint() (see
    # test_workflow_schedule_runtime_paths.py), which reaches this hook with the
    # real HUB_AGENT_CLAUDE_WORKSPACE from .env. That was harmless while the sync
    # only *read* git status and bailed on dirt; the moment it gained a rescue +
    # reset path it started stashing and moving Orion's actual checkout -- caught
    # live on 2026-08-14, stash@{1} authored by the test runner. Mocking it in
    # each test would work until the next test forgets; this cannot be forgotten.
    # PYTEST_CURRENT_TEST alone is not enough: it is set per test *item*, so a task
    # or thread that outlives the item -- which is exactly what this fire-and-forget
    # task is -- can observe it as unset. `sys.modules` is stable for the whole
    # process, and hub never imports pytest in production.
    if os.environ.get("PYTEST_CURRENT_TEST") or "pytest" in sys.modules:
        logger.info("fcc_sandbox_sync_skipped_test_process connection_id=%s", connection_id)
        record_sync_skip(
            getattr(settings, "HUB_AGENT_CLAUDE_WORKSPACE", None),
            "skipped_test_process",
        )
        return

    if active_turns():
        logger.info(
            "fcc_sandbox_sync_skipped_turn_in_flight connection_id=%s", connection_id
        )
        # Recorded, not just logged: an unrecorded skip would leave the status
        # surface reporting whatever the previous connect saw, which is exactly
        # the "looks fine, is stale" failure this whole patch exists to kill.
        record_sync_skip(
            getattr(settings, "HUB_AGENT_CLAUDE_WORKSPACE", None),
            "skipped_turn_in_flight",
        )
        return
    async with _fcc_sandbox_sync_lock:
        # Re-check after acquiring the lock: a turn may have started while queued.
        if active_turns():
            logger.info(
                "fcc_sandbox_sync_skipped_turn_in_flight connection_id=%s", connection_id
            )
            record_sync_skip(
                getattr(settings, "HUB_AGENT_CLAUDE_WORKSPACE", None),
                "skipped_turn_in_flight",
            )
            return
        sync_result = await asyncio.to_thread(
            sync_fcc_sandbox, getattr(settings, "HUB_AGENT_CLAUDE_WORKSPACE", None)
        )
        logger.info(
            "fcc_sandbox_sync connection_id=%s result=%s", connection_id, sync_result
        )


async def websocket_endpoint(websocket: WebSocket):
    """Orion capability: Hub chat entry.

    Accepts every Hub chat WebSocket and routes each message by client mode;
    an Orion-mode message with ORION_UNIFIED_TURN_ENABLED routes into
    run_unified_turn rather than the older direct Cortex request path. This
    function owns connection lifecycle and frame relay only — turn cognition
    lives in the unified turn orchestrator.

    Runtime evidence: connection_ready frame, relayed harness step frames, and
    the turn frames returned by the orchestrator. Start here when an
    Orion-mode message produced no turn frames at all.
    """
    import scripts.main
    bus = scripts.main.bus
    cortex_client = scripts.main.cortex_client
    tts_client = scripts.main.tts_client
    biometrics_cache = scripts.main.biometrics_cache
    notification_cache = scripts.main.notification_cache
    agent_step_relay = scripts.main.agent_step_relay
    harness_step_relay = scripts.main.harness_step_relay
    rpc_bus = scripts.main.rpc_bus
    presence_state = scripts.main.presence_state
    presence_context_store = getattr(scripts.main, "presence_context_store", None)

    await websocket.accept()
    logger.info("WebSocket accepted.")
    if presence_state:
        presence_state.connected()

    connection_id = str(uuid.uuid4())
    await websocket.send_json({"type": "connection_ready", "connection_id": connection_id})

    # Refresh Orion's disposable FCC sandbox checkout (HUB_AGENT_CLAUDE_WORKSPACE) to
    # current origin/main once per browser session, not per turn -- see
    # orion/fcc/sandbox_sync.py for why this lives here and what it guards against.
    # Fire-and-forget: runs in the background so a slow/unreachable origin can't delay
    # this connection's readiness handshake, and _fcc_sandbox_sync_lock plus the
    # active-turn check keep it from racing a concurrent connection's sync or
    # resetting the workspace out from under a claude subprocess that's still running
    # with cwd=workspace. Best-effort: a sync failure just degrades to "this turn sees
    # a stale checkout", never breaks the connection.
    asyncio.create_task(_sync_fcc_sandbox_background(connection_id))

    client_meta = {
        "user_agent": websocket.headers.get("user-agent"),
        "origin": websocket.headers.get("origin"),
        "x_forwarded_for": websocket.headers.get("x-forwarded-for"),
        "client_host": getattr(websocket.client, "host", None),
        "client_port": getattr(websocket.client, "port", None),
    }

    # Soft warning if services missing, but keep connection alive
    if not bus or not cortex_client:
        logger.warning("OrionBus/CortexClient not ready. Chat will be limited.")
        await websocket.send_json(await _with_biometrics({
            "llm_response": "[SYSTEM WARNING] Bus disconnected. Brain is offline, but UI is active.", 
            "state": "idle"
        }, cache=biometrics_cache))

    history: List[Dict[str, Any]] = [
        {"role": "system", "content": mini_personality_summary()}
    ]

    tts_q: asyncio.Queue = asyncio.Queue()
    if notification_cache is not None:
        notification_cache.register_queue(tts_q)
    drain_task = asyncio.create_task(drain_queue(websocket, tts_q, biometrics_cache))
    # Active FCC / harness turn for this socket — cancelled on WS disconnect.
    active_turn: Dict[str, Optional[str]] = {"correlation_id": None, "kind": None}
    # Registered by reference: mutating active_turn above is automatically visible
    # to the /api/chat/turn/cancel endpoint's lookup, no separate sync needed.
    _ACTIVE_TURNS_BY_CONNECTION[connection_id] = active_turn

    # Endogenous outreach needs the same two things by reference: this socket's
    # outbound queue (to push an unsolicited Orion bubble) and active_turn (so it
    # never speaks over a turn in flight). Best-effort: outreach is optional.
    endogenous_outreach = getattr(scripts.main, "endogenous_outreach", None)
    if endogenous_outreach is not None:
        endogenous_outreach.register_connection(connection_id, tts_q, active_turn)
    # Same queue, so a Claude reply reaches the browser by the identical path
    # an Orion outreach bubble does.
    room_relay = getattr(scripts.main, "room_claude_relay", None)
    if room_relay is not None:
        room_relay.register_connection(connection_id, tts_q)

    try:
        while True:
            # Idle marker for endogenous outreach. Control reaches this line
            # only when the previous message has been fully handled -- every
            # `continue` in this loop body passes through here -- so it is the
            # one reliable "this socket is done" point without restructuring the
            # loop. Paired with note_busy() just below.
            if endogenous_outreach is not None:
                endogenous_outreach.note_idle(connection_id)
            raw = await websocket.receive_text()
            if presence_state:
                presence_state.heartbeat()
            try:
                data: Dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if endogenous_outreach is not None:
                # Set for EVERY mode. active_turn["correlation_id"] is only
                # populated by the unified-orion and agent-claude lanes, so the
                # UI's Quick / Story / Agent modes would otherwise look idle to
                # outreach for the whole duration of a real turn.
                endogenous_outreach.note_busy(connection_id)

            # Connect-time handshake carrying this tab's session_id. Handled
            # before any turn logic and never treated as a chat message: without
            # it, an open-but-idle tab is session-less to outreach (session_id
            # otherwise only arrives on an outbound message), so outreach posts
            # to its fallback session rather than the thread on screen.
            if str(data.get("type") or "").strip() == "session_hello":
                if endogenous_outreach is not None:
                    endogenous_outreach.note_session(connection_id, data.get("session_id"))
                _rr = getattr(scripts.main, "room_claude_relay", None)
                if _rr is not None:
                    _rr.note_session(connection_id, data.get("session_id"))
                # Rebuild conversation context from persisted turns. `history`
                # is in-memory and scoped to this socket, so before this every
                # Hub restart silently discarded the running conversation -- and
                # build_continuity_messages has no fallback, so the next turn
                # went out with nothing but the current prompt. This handshake
                # is the first moment we know which thread the tab is in, which
                # makes it the right place to restore it.
                await _rehydrate_connection_history(
                    history, session_id=data.get("session_id")
                )
                logger.debug(
                    "ws_session_hello connection=%s session=%s",
                    connection_id,
                    data.get("session_id"),
                )
                continue

            mode = data.get("mode") or ("auto" if settings.HUB_AUTO_DEFAULT_ENABLED else "brain")
            client_mode = str(mode or "").strip().lower()
            disable_tts = data.get("disable_tts", False)
            diagnostic = bool(
                data.get("diagnostic")
                or (isinstance(data.get("options"), dict) and data.get("options", {}).get("diagnostic"))
            )
            session_id = data.get("session_id")
            publish_session_id = session_id or "unknown"
            if endogenous_outreach is not None:
                # session_id is client-side (localStorage) and only reaches Hub
                # here, so this is the one place outreach can learn which thread
                # to post into.
                endogenous_outreach.note_session(connection_id, session_id)
            _rr = getattr(scripts.main, "room_claude_relay", None)
            if _rr is not None:
                _rr.note_session(connection_id, session_id)
            if not session_id and diagnostic:
                logger.warning("Missing session_id; publishing chat history with session_id=unknown")
            no_write = bool(data.get("no_write", settings.HUB_DEFAULT_NO_WRITE))

            # Trace Verb & Test Stub Logic ---
            # 1. Default to general chat
            trace_verb = "chat_general"

            # 2. Map modes to verbs for the Visualizer
            if mode == "agent":
                trace_verb = "task_execution"
            elif mode == "agent-claude":
                trace_verb = "task_execution"
            elif mode == "council":
                trace_verb = "council_deliberation"
            elif isinstance(data.get("verbs"), list) and len(data.get("verbs")) == 1:
                candidate_verb = str(data.get("verbs")[0] or "").strip()
                if candidate_verb:
                    trace_verb = candidate_verb

            # 3. Force verb for Test/Stub Submissions
            if data.get("test_mode") or data.get("submission_id"):
                trace_verb = "test_submission"
            # -----------------------------------------

            transcript: Optional[str] = None
            is_text_input = False

            # 1. Input Processing
            possible_text = data.get("text_input") or data.get("text") or data.get("content")
            if possible_text:
                transcript = possible_text
                is_text_input = True
            elif data.get("audio"):
                audio_b64 = data.get("audio") or ""
                try:
                    audio_byte_len = len(base64.b64decode(audio_b64, validate=False))
                except Exception:
                    audio_byte_len = 0
                client_audio_meta = sanitize_client_audio_meta(data.get("client_audio_meta"))
                if client_audio_meta:
                    logger.info(
                        "voice.ws.audio_received session_id=%s audio_bytes=%d client_meta=%s",
                        session_id,
                        audio_byte_len,
                        client_audio_meta,
                    )
                else:
                    logger.info(
                        "voice.ws.audio_received session_id=%s audio_bytes=%d",
                        session_id,
                        audio_byte_len,
                    )
                if tts_client:
                    try:
                        await websocket.send_json(
                            await _with_biometrics({"state": "processing"}, cache=biometrics_cache)
                        )
                        audio_format = (
                            data.get("audio_format")
                            or data.get("format")
                            or "wav"
                        )
                        stt_options: Dict[str, Any] = {}
                        if client_audio_meta:
                            stt_options["client_audio_meta"] = client_audio_meta
                        stt_req = STTRequestPayload(
                            audio_b64=data.get("audio"),
                            language=data.get("language") or "en",
                            format=audio_format,
                            options=stt_options or None,
                        )
                        logger.info(
                            "voice.stt.start session_id=%s format=%s",
                            session_id,
                            audio_format,
                        )
                        stt_result = await asyncio.wait_for(
                            tts_client.transcribe(stt_req),
                            timeout=float(settings.HUB_STT_TIMEOUT_SEC),
                        )
                        transcript = (stt_result.text or "").strip()
                        logger.info(
                            "voice.stt.done session_id=%s transcript_len=%d",
                            session_id,
                            len(transcript),
                        )
                        if not transcript:
                            meta = stt_result.metadata or {}
                            logger.info(
                                "voice.stt.empty session_id=%s meta=%s client_meta=%s",
                                session_id,
                                meta,
                                client_audio_meta,
                            )
                            err_msg = empty_transcript_error_message(
                                client_audio_meta=client_audio_meta,
                                stt_meta=meta,
                                audio_byte_len=audio_byte_len,
                            )
                            audio_debug = build_audio_debug(
                                stt_meta=meta,
                                client_audio_meta=client_audio_meta,
                            )
                            err_payload: Dict[str, Any] = {
                                "error": err_msg,
                                "state": "idle",
                            }
                            if audio_debug:
                                err_payload["audio_debug"] = audio_debug
                            await websocket.send_json(
                                await _with_biometrics(
                                    err_payload,
                                    cache=biometrics_cache,
                                )
                            )
                            continue
                    except asyncio.TimeoutError:
                        logger.error(
                            "voice.stt.error session_id=%s err=STT timed out after %ss",
                            session_id,
                            settings.HUB_STT_TIMEOUT_SEC,
                        )
                        timeout_payload: Dict[str, Any] = {
                            "error": "Transcription timed out",
                            "state": "idle",
                        }
                        audio_debug = build_audio_debug(client_audio_meta=client_audio_meta)
                        if audio_debug:
                            timeout_payload["audio_debug"] = audio_debug
                        await websocket.send_json(
                            await _with_biometrics(
                                timeout_payload,
                                cache=biometrics_cache,
                            )
                        )
                        continue
                    except Exception as e:
                        logger.error("voice.stt.error session_id=%s err=%s", session_id, e)
                        err_text = str(e).strip()
                        if err_text.startswith("STT error:"):
                            err_text = err_text[len("STT error:") :].strip()
                        fail_payload: Dict[str, Any] = {
                            "error": err_text or "Transcription failed",
                            "state": "idle",
                        }
                        audio_debug = build_audio_debug(client_audio_meta=client_audio_meta)
                        if audio_debug:
                            fail_payload["audio_debug"] = audio_debug
                        await websocket.send_json(
                            await _with_biometrics(
                                fail_payload,
                                cache=biometrics_cache,
                            )
                        )
                        continue
                else:
                    await websocket.send_json(
                        await _with_biometrics(
                            {"error": "STT service unavailable", "state": "idle"},
                            cache=biometrics_cache,
                        )
                    )
                    continue

            if not transcript:
                continue

            if not is_text_input:
                await websocket.send_json(
                    await _with_biometrics(
                        {"transcript": transcript, "is_text_input": False},
                        cache=biometrics_cache,
                    )
                )

            # 2. Chat Execution
            if not cortex_client:
                await websocket.send_json(
                    await _with_biometrics(
                        {"error": "Cortex disconnected (Bus offline)", "state": "idle"},
                        cache=biometrics_cache,
                    )
                )
                continue

            if no_write:
                logger.info("NO_WRITE active (WS) sid=%s", session_id)

            # ----------------------------
            # Hub-side short-term memory
            # ----------------------------
            # N = userassistant pairs; helper keeps up to 2*N messages.
            turns = int(data.get("context_turns") or getattr(settings, "HUB_CONTEXT_TURNS", 10))
            prompt_with_ctx = transcript
            # IMPORTANT: store the raw user message for next turn
            history.append({"role": "user", "content": transcript})


            trace_id = str(uuid.uuid4())

            if client_mode == "orion" and settings.ORION_UNIFIED_TURN_ENABLED:
                if not settings.ORION_HARNESS_GOVERNOR_ENABLED:
                    # Pop the user turn just appended above -- no assistant
                    # turn will ever answer it on this early-exit path, and
                    # leaving it in `history` means a client that retries on
                    # the same socket gets two consecutive {role: user}
                    # entries with nothing in between (review finding,
                    # 2026-08-22: caught on the sibling import-guard path
                    # below, applies here too -- same shape, same fix).
                    if history and history[-1].get("role") == "user":
                        history.pop()
                    await _safe_ws_send_json(
                        websocket,
                        await _with_biometrics(
                            {
                                "type": "turn_error",
                                "phase": "config",
                                "error": "harness_governor_disabled",
                            },
                            cache=biometrics_cache,
                        ),
                    )
                    continue
                # Guarded, not a bare module-level import -- confirmed live,
                # 2026-08-23: an ImportError here (orion/situational/
                # context.py's `datetime.UTC`, valid on the dev venv's
                # Python 3.12 but not this container's actual 3.10 runtime)
                # propagated straight out of this whole handler with no
                # `turn_error` frame ever sent -- unlike the
                # harness_governor_disabled case just above, which does.
                # The browser just saw the socket die mid-turn with zero
                # explanation, reading as a silent hang rather than a
                # visible, debuggable error.
                #
                # ImportError specifically, not Exception -- narrower on
                # purpose (review finding, 2026-08-23): this guard exists to
                # turn ONE confirmed failure mode (a py3.10-incompatible
                # stdlib import) into a client-visible error instead of a
                # dead socket. Swallowing every exception class here would
                # also mask an unrelated, unexpected failure somewhere in
                # turn_orchestrator's transitive import graph behind the
                # same generic "phase: import" message, indistinguishable
                # from this bug and re-attempted on every single message on
                # a long-lived connection instead of failing loud once.
                try:
                    from orion.hub.turn_orchestrator import run_unified_turn
                except ImportError as exc:
                    logger.error(
                        "turn_import_failed correlation_id=%s error=%s",
                        trace_id,
                        exc,
                    )
                    if history and history[-1].get("role") == "user":
                        history.pop()
                    await _safe_ws_send_json(
                        websocket,
                        await _with_biometrics(
                            {
                                "type": "turn_error",
                                "phase": "import",
                                "error": str(exc),
                            },
                            cache=biometrics_cache,
                        ),
                    )
                    continue

                active_turn["correlation_id"] = trace_id
                active_turn["kind"] = "orion"
                # Affect bracket, leg 1 of 2. Fired here rather than earlier
                # (right after voice.stt.done) on purpose: everything between
                # those two points can still bail out with `continue`
                # (harness_governor_disabled, the turn_orchestrator import
                # guard), and a "pre" capture with no turn behind it and no
                # "post" to pair with is worse than no capture at all -- it
                # is a webcam recording of Juniper attributed to a
                # conversation that never happened. By this line the turn is
                # actually about to run. Never awaited: see
                # scripts/chat_turn_affect.py for why (up to ~195s) and for
                # the honest consequence (this capture colours the NEXT
                # turn, not this one).
                # subtitle: the transcript Hub ALREADY has from the browser
                # microphone, passed on the PRE leg only. This reverses an
                # earlier deliberate choice to send "" (see the
                # /capture_and_assess route docstring in
                # orion-juniper-affective-state), and the reversal is the
                # point: that choice was correct only while the affect
                # capture recorded its own audio a few seconds later, which
                # it no longer does. Juniper's report, 2026-08-26 -- two
                # divorced audio recordings meant the affect read could only
                # be grounded by her repeating herself into a worse mic.
                #
                # POST gets no subtitle deliberately: she is not speaking
                # then, so there is no transcript belonging to that window,
                # and reusing this one would present her opening words as if
                # they were her reaction to Orion's reply.
                chat_turn_affect.fire(
                    settings=settings,
                    trigger=chat_turn_affect.TRIGGER_PRE,
                    correlation_id=trace_id,
                    is_voice_turn=not is_text_input,
                    subtitle=transcript,
                )
                orion_turn_frames: list[dict] = []
                try:
                    orion_turn_frames = await run_awaitable_cancel_on_ws_disconnect(
                        websocket,
                        run_unified_turn(
                            websocket,
                            bus=bus,
                            correlation_id=trace_id,
                            session_id=session_id,
                            user_message=transcript,
                            payload=data,
                            continuity_messages=build_continuity_messages(
                                history=history,
                                latest_user_prompt=transcript,
                                turns=turns,
                            ),
                            with_biometrics=_with_biometrics,
                            biometrics_cache=biometrics_cache,
                            harness_rpc_bus=rpc_bus or bus,
                            harness_step_relay=harness_step_relay,
                        ),
                        bus=rpc_bus or bus,
                        correlation_id=trace_id,
                        kind="orion",
                    )
                finally:
                    active_turn["correlation_id"] = None
                    active_turn["kind"] = None
                    # Affect bracket, leg 2 of 2 -- in `finally`, not after
                    # it, so a turn that ERRORS still gets its closing read.
                    # An exchange that went wrong is exactly the one whose
                    # after-state is worth having, and a pre with no post is
                    # an unusable half of a matched pair.
                    #
                    # But NOT when the client is gone. run_awaitable_cancel_
                    # on_ws_disconnect cancels the turn the moment
                    # client_state leaves CONNECTED, and that cancellation
                    # lands right here -- so without this check, closing the
                    # tab mid-turn would start a live webcam+mic recording of
                    # Juniper AFTER she left, attributed to a conversation
                    # she had already walked away from. That is the same
                    # objection the pre leg's own comment above makes, and it
                    # applies just as hard on the cancel path (review
                    # finding, 2026-08-26). A disconnect is the one case
                    # where the missing half of the pair is the correct
                    # outcome.
                    if websocket.client_state == WebSocketState.CONNECTED:
                        chat_turn_affect.fire(
                            settings=settings,
                            trigger=chat_turn_affect.TRIGGER_POST,
                            correlation_id=trace_id,
                            is_voice_turn=not is_text_input,
                        )
                    else:
                        logger.info(
                            "chat_turn_affect_post_skipped corr=%s reason=client_disconnected",
                            trace_id,
                        )

                # Voice reply, orion lane. Confirmed live 2026-08-26 (real
                # incident, corr=7dc1bab2-97a4-4390-89a2-cdd1fa4f0092): this
                # lane's own `continue` above exits before EVER reaching the
                # classic lane's own "4. TTS" block further down in this
                # function -- so an Orion-mode turn had no path to speech at
                # all, structurally, regardless of disable_tts. Voice INPUT
                # (STT, above) always worked; voice OUTPUT never did for
                # this lane specifically. Mirrors the classic lane's own
                # will_tts gate (disable_tts, tts_client) so the two lanes
                # behave identically from the browser's point of view --
                # same `tts_q` the classic lane's own drain_task (defined
                # once per connection, above the while loop) already relays
                # audio through, so no new plumbing is needed on the
                # playback side, only the trigger.
                #
                # Deliberately scoped to the "final" frame's real
                # llm_response only -- NOT a turn_error frame's
                # partial_draft. A partial draft is real assistant-authored
                # text and the browser does render it as an Orion bubble,
                # but speaking it aloud on an error path is a materially
                # different, untested product decision left for a
                # deliberate follow-up rather than folded in silently here.
                #
                # Skipped on disconnect for the same reason the post-affect
                # leg is: synthesizing speech for a socket that is already
                # gone is pure waste, and the client_state check is already
                # right here.
                if websocket.client_state == WebSocketState.CONNECTED:
                    orion_final_text = extract_unified_turn_final_text(orion_turn_frames)
                    dispatch_tts_reply(
                        text=orion_final_text,
                        disable_tts=disable_tts,
                        tts_client=tts_client,
                        tts_q=tts_q,
                        correlation_id=trace_id,
                        session_id=session_id,
                        lane="orion",
                    )
                continue

            # Build outbound chat request through shared builder to keep WS/HTTP identical
            inactive = validate_single_verb_override(data, node_name=settings.NODE_NAME, prompt=transcript)
            if inactive:
                await websocket.send_json(await _with_biometrics({"error": inactive.get("message") or inactive.get("error")}, cache=biometrics_cache))
                continue

            continuity_messages = build_continuity_messages(
                history=history,
                latest_user_prompt=transcript,
                turns=turns,
            )
            data = dict(data)
            data = await _apply_hub_direct_social_room_mode(data)
            data = inject_session_presence(data, str(session_id or "anonymous"), presence_context_store)
            data["mutation_cognition_context"] = build_mutation_cognition_context()
            try:
                chat_req, route_debug, use_recall = build_chat_request(
                    payload=data,
                    session_id=session_id,
                    user_id=data.get("user_id"),
                    trace_id=trace_id,
                    default_mode="brain",
                    auto_default_enabled=bool(settings.HUB_AUTO_DEFAULT_ENABLED),
                    source_label="hub_ws",
                    prompt=prompt_with_ctx,
                    messages=continuity_messages,
                )
            except HubRequestValidationError as exc:
                await websocket.send_json(
                    await _with_biometrics(
                        {"error": str(exc), "error_code": exc.code},
                        cache=biometrics_cache,
                    )
                )
                continue
            workflow_request = chat_req.metadata.get("workflow_request") if isinstance(chat_req.metadata, dict) else None
            execution_policy = workflow_request.get("execution_policy") if isinstance(workflow_request, dict) else None
            logger.info(
                "workflow_resolution_result %s",
                json.dumps(
                    {
                        "correlation_id": trace_id,
                        "matched_workflow_id": (workflow_request or {}).get("workflow_id") if isinstance(workflow_request, dict) else None,
                        "fallback_route": route_debug.get("fallback_route"),
                        "reason": route_debug.get("workflow_resolution_reason"),
                    },
                    sort_keys=True,
                    default=str,
                ),
            )
            logger.info(
                "hub_workflow_request corr=%s sid=%s workflow_id=%s invocation_mode=%s schedule_kind=%s source=ws",
                trace_id,
                session_id,
                (workflow_request or {}).get("workflow_id") if isinstance(workflow_request, dict) else None,
                (execution_policy or {}).get("invocation_mode") if isinstance(execution_policy, dict) else None,
                ((execution_policy or {}).get("schedule") or {}).get("kind") if isinstance(execution_policy, dict) else None,
            )
            if route_debug.get("verb"):
                trace_verb = str(route_debug["verb"])
            chat_req.metadata = dict(chat_req.metadata or {})
            chat_req.metadata["trace_verb"] = trace_verb
            mode = chat_req.mode
            recall_payload = chat_req.recall or {"enabled": use_recall}
            turn_client_meta = dict(client_meta)
            if is_social_room_payload(data):
                turn_client_meta.update(
                    social_room_client_meta(
                        payload=data,
                        route_debug=route_debug,
                        trace_verb=trace_verb,
                        memory_digest=None,
                    )
                )

            logger.info(f"WS Chat Request recall config: {recall_payload} session_id={session_id}")
            logger.info(
                "Routing resolved to mode: %s (verb: %s)",
                mode,
                trace_verb,
            )
            logger.info(
                "WS routing resolved mode=%s route_intent=%s verb=%s allowed_verbs=%s",
                chat_req.mode,
                chat_req.route_intent,
                chat_req.verb,
                len(((chat_req.options or {}).get("allowed_verbs") or [])),
            )
            logger.info(
                "WS Chat Request payload session_id=%s history_len=%s last_user_len=%s last_user_head=%r",
                session_id,
                len(history),
                len(transcript or ""),
                (transcript or "")[:120],
            )
            logger.info(
                "hub_egress corr=%s sid=%s mode=%s verb=%s route_intent=%s allowed_verbs=%s packs=%s",
                trace_id,
                session_id,
                chat_req.mode,
                chat_req.verb,
                (chat_req.options or {}).get("route_intent") or "none",
                len(((chat_req.options or {}).get("allowed_verbs") or [])),
                chat_req.packs or [],
            )
            logger.info(
                "hub_context_messages corr=%s sid=%s mode=%s count=%s roles=%s",
                trace_id,
                session_id,
                chat_req.mode,
                len(chat_req.messages or []),
                [m.role if hasattr(m, "role") else m.get("role") for m in (chat_req.messages or [])][:12],
            )
            _log_hub_route_decision(
                corr_id=trace_id,
                session_id=session_id,
                route_debug=route_debug,
                user_prompt=transcript,
            )

            # ─── Hub presence (best-effort, never blocks chat) ──────────────────
            # One timestamp per turn; mirrors a liveness snapshot for self-state.
            try:
                from scripts.hub_presence import record_turn

                record_turn()
            except Exception:
                pass

            substrate_summary = None
            substrate_snapshot = None
            pre_turn_bundle = None

            if settings.ENABLE_PRE_TURN_APPRAISAL:
                from scripts.pre_turn_appraisal_wiring import run_pre_turn_appraisal_wiring

                continuity_messages = [
                    m.model_dump(mode="json") if hasattr(m, "model_dump") else m
                    for m in (chat_req.messages or [])
                ]
                substrate_summary, pre_turn_bundle = await run_pre_turn_appraisal_wiring(
                    chat_req,
                    bus=bus,
                    correlation_id=trace_id,
                    session_id=str(session_id or "anonymous"),
                    continuity_messages=continuity_messages or [{"role": "user", "content": transcript}],
                    user_prompt=transcript,
                    paradigms=settings.PRE_TURN_APPRAISAL_PARADIGMS,
                    timeout_ms=settings.PRE_TURN_APPRAISAL_TIMEOUT_MS,
                )
            else:
                substrate_summary, substrate_snapshot = run_substrate_effect_pipeline(
                    turn_id=trace_id,
                    message_id=None,
                    user_text=transcript,
                    source_id=str(session_id or "anonymous"),
                    contract_before={"mode": "default"},
                )
                attach_repair_pressure_contract(
                    chat_req,
                    substrate_snapshot,
                    enabled=settings.ENABLE_REPAIR_PRESSURE_SPEECH_WIRING,
                )

            if substrate_summary is not None:
                logger.info(
                    "substrate_effect_attached ws corr=%s level=%s changed=%s",
                    trace_id,
                    substrate_summary.get("level_label"),
                    substrate_summary.get("changed_behavior"),
                )

            # Chat grammar trace (fail-open, behind PUBLISH_HUB_CHAT_GRAMMAR env flag)
            if bus and settings.PUBLISH_HUB_CHAT_GRAMMAR:
                try:
                    from scripts.grammar_emit import build_chat_turn_grammar_events
                    from scripts.grammar_publish import publish_hub_chat_grammar_trace
                    from scripts.pre_turn_appraisal_wiring import repair_pressure_grammar_scalars

                    repair_pressure_level, repair_pressure_confidence = repair_pressure_grammar_scalars(
                        pre_turn_bundle=pre_turn_bundle,
                        substrate_summary=substrate_summary,
                    )
                    _chat_grammar_events = build_chat_turn_grammar_events(
                        turn_id=trace_id,
                        session_id=str(session_id or "anonymous"),
                        node_id=settings.NODE_NAME,
                        word_count=len((transcript or "").split()),
                        repair_pressure_level=repair_pressure_level,
                        repair_pressure_confidence=repair_pressure_confidence,
                        has_repair_signal=substrate_summary is not None,
                    )
                    _schedule_publish(
                        publish_hub_chat_grammar_trace(
                            bus,
                            _chat_grammar_events,
                            correlation_id=trace_id,
                            channel=settings.GRAMMAR_EVENT_CHANNEL,
                            enabled=True,
                        ),
                        "chat.grammar",
                    )
                except Exception:
                    logger.warning("hub_chat_grammar_wire_failed corr=%s", trace_id, exc_info=True)

            if diagnostic:
                logger.info("WS outbound CortexChatRequest corr=%s payload=%s", trace_id, chat_req.model_dump(mode="json"))

            _rec_tape_req(
                corr_id=trace_id,
                session_id=session_id,
                mode=mode,
                use_recall=use_recall,
                recall_profile=recall_payload.get("profile"),
                user_head=(transcript or "")[:80],
                no_write=no_write,
            )
            # Publish the inbound user message into chat history
            if bus and not no_write:
                user_env = build_chat_history_envelope(
                    content=transcript,
                    role="user",
                    session_id=publish_session_id,
                    correlation_id=trace_id,
                    speaker=data.get("user_id") or "user",
                    tags=[mode],
                    message_id=f"{trace_id}:user",
                    memory_status="accepted",
                    memory_tier="ephemeral",
                    client_meta=turn_client_meta,
                )
                _schedule_publish(publish_chat_history(bus, [user_env]), "chat.history user")

            orion_response_text = ""
            memory_digest = None
            recall_debug = None
            agent_trace = None
            workflow = None
            autonomy_payload: Dict[str, Any] = {}
            metacog_traces: List[Dict[str, Any]] = []
            reasoning_content: Optional[str] = None
            inline_think_content: Optional[str] = None
            thinking_source: str = "none"
            explicit_reasoning_trace: Optional[Dict[str, Any]] = None
            used_context_exec_lane = False
            used_agent_claude_lane = False
            workflow_metadata_only = False
            resp = None
            cortex_result_dump: Dict[str, Any] = {}
            try:
                logger.info("voice.chat.start corr=%s session_id=%s", trace_id, session_id)
                if client_mode == "agent-claude":
                    used_agent_claude_lane = True
                    active_turn["correlation_id"] = trace_id
                    active_turn["kind"] = "agent-claude"
                    try:
                        agent_claude_out = await _run_agent_claude_turn_ws(
                            websocket=websocket,
                            data=data,
                            transcript=transcript or "",
                            trace_id=trace_id,
                            biometrics_cache=biometrics_cache,
                        )
                    finally:
                        active_turn["correlation_id"] = None
                        active_turn["kind"] = None
                    if agent_claude_out is None:
                        continue
                    orion_response_text = str(agent_claude_out.get("llm_response") or "")
                    agent_trace = None
                    cortex_result_dump = {}
                    route_debug = route_debug if isinstance(route_debug, dict) else {}
                    agent_meta: Dict[str, Any] = {
                        "fcc_model_label": agent_claude_out.get("fcc_model_label"),
                        **(agent_claude_out.get("metadata") or {}),
                    }
                    if is_context_overflow_text(orion_response_text):
                        n_ctx = int(getattr(settings, "HUB_AGENT_CLAUDE_MAX_CONTEXT_TOKENS", 65536))
                        hint = context_overflow_operator_hint(n_ctx=n_ctx)
                        if hint.strip() not in orion_response_text:
                            orion_response_text = f"{orion_response_text.rstrip()}{hint}"
                        agent_meta["context_overflow"] = True
                    route_debug["agent_claude"] = agent_meta
                    harness_steps = agent_claude_out.get("harness_steps") or []
                    harness_trace = build_harness_reasoning_trace(
                        steps=harness_steps if isinstance(harness_steps, list) else [],
                        correlation_id=trace_id,
                        session_id=publish_session_id,
                        model_label=str(agent_claude_out.get("fcc_model_label") or ""),
                    )
                    if harness_trace:
                        explicit_reasoning_trace = harness_trace
                        thinking_source = "agent_claude_harness"
                else:
                    used_context_exec_lane = should_use_context_exec_agent_lane(chat_req)
                    if used_context_exec_lane:
                        step_queue: asyncio.Queue = asyncio.Queue(maxsize=256)
                        relay = agent_step_relay
                        drain_task = None
                        if relay is None:
                            logger.warning(
                                "agent_step_relay unavailable; live step streaming disabled corr=%s",
                                trace_id,
                            )
                        if relay is not None:
                            relay.register_queue(trace_id, step_queue)

                            async def _drain_steps() -> None:
                                try:
                                    while True:
                                        item = await step_queue.get()
                                        await _safe_ws_send_json(websocket, item)
                                except asyncio.CancelledError:
                                    pass

                            drain_task = asyncio.create_task(_drain_steps(), name=f"agent-steps-{trace_id}")
                        try:
                            ctx_out = await run_hub_agent_via_context_exec(
                                req=chat_req,
                                prompt=transcript or prompt_with_ctx,
                                correlation_id=trace_id,
                                route_debug=route_debug if isinstance(route_debug, dict) else {},
                            )
                        finally:
                            if relay is not None:
                                while not step_queue.empty():
                                    try:
                                        item = step_queue.get_nowait()
                                    except asyncio.QueueEmpty:
                                        break
                                    await _safe_ws_send_json(websocket, item)
                            if drain_task is not None:
                                drain_task.cancel()
                                try:
                                    await drain_task
                                except asyncio.CancelledError:
                                    pass
                            if relay is not None:
                                relay.unregister_queue(trace_id, step_queue)
                        if ctx_out.get("error"):
                            await websocket.send_json(
                                await _with_biometrics(
                                    {
                                        "error": ctx_out.get("error"),
                                        "error_code": ctx_out.get("error_code"),
                                        "mode": "agent",
                                        "correlation_id": trace_id,
                                        "routing_debug": ctx_out.get("routing_debug") or route_debug,
                                    },
                                    cache=biometrics_cache,
                                )
                            )
                            continue
                        orion_response_text = str(ctx_out.get("llm_response") or "")
                        agent_trace = ctx_out.get("agent_trace")
                        cortex_result_dump = ctx_out.get("raw") if isinstance(ctx_out.get("raw"), dict) else {}
                        route_debug = ctx_out.get("routing_debug") or route_debug
                        resp = None
                    else:
                        resp: CortexChatResult = await cortex_client.chat(chat_req, correlation_id=trace_id)
                        orion_response_text = hub_effective_chat_text(resp)
                        if resp.cortex_result and isinstance(resp.cortex_result.recall_debug, dict):
                            recall_debug = resp.cortex_result.recall_debug
                            memory_digest = recall_debug.get("memory_digest")
                        agent_trace = extract_agent_trace_payload(resp.cortex_result)
                        cortex_result_dump = (
                            resp.cortex_result.model_dump(mode="json")
                            if getattr(resp, "cortex_result", None) is not None and hasattr(resp.cortex_result, "model_dump")
                            else {}
                        )
                if not used_context_exec_lane and not used_agent_claude_lane:
                    raw_traces = getattr(resp.cortex_result, "metacog_traces", None)
                    if isinstance(raw_traces, list):
                        metacog_traces = [t for t in raw_traces if isinstance(t, dict)]
                    gateway_meta = resp.cortex_result.metadata if resp.cortex_result and isinstance(resp.cortex_result.metadata, dict) else {}
                    explicit_reasoning_trace = (
                        cortex_result_dump.get("reasoning_trace")
                        if isinstance(cortex_result_dump, dict) and isinstance(cortex_result_dump.get("reasoning_trace"), dict)
                        else None
                    )
                    reasoning_content = (
                        (gateway_meta or {}).get("reasoning_content")
                        or (cortex_result_dump.get("reasoning_content") if isinstance(cortex_result_dump, dict) else None)
                        or (((cortex_result_dump.get("raw") or {}).get("reasoning_content")) if isinstance(cortex_result_dump.get("raw"), dict) else None)
                    )
                    inline_think_content = (
                        (gateway_meta or {}).get("inline_think_content")
                        or (cortex_result_dump.get("inline_think_content") if isinstance(cortex_result_dump, dict) else None)
                    )
                    raw_thinking_source = (
                        (gateway_meta or {}).get("thinking_source")
                        or (cortex_result_dump.get("thinking_source") if isinstance(cortex_result_dump, dict) else None)
                    )
                    if isinstance(raw_thinking_source, str) and raw_thinking_source.strip():
                        thinking_source = raw_thinking_source.strip()
                    elif isinstance(reasoning_content, str) and reasoning_content.strip():
                        thinking_source = "provider_reasoning"
                    elif isinstance(inline_think_content, str) and inline_think_content.strip():
                        thinking_source = "inline_think_full_block"
                    if reasoning_content and not (isinstance(explicit_reasoning_trace, dict) and str(explicit_reasoning_trace.get("content") or "").strip()):
                        explicit_reasoning_trace = {
                            "trace_role": "reasoning",
                            "trace_stage": "post_answer",
                            "content": str(reasoning_content).strip(),
                            "metadata": {"source": "hub_reasoning_content_fallback"},
                        }
                    trace_content = explicit_reasoning_trace.get("content") if isinstance(explicit_reasoning_trace, dict) else None
                    print(
                        "===THINK_HOP=== hop=hub_in "
                        f"corr={trace_id} "
                        f"keys={sorted(cortex_result_dump.keys()) if isinstance(cortex_result_dump, dict) else []} "
                        f"reasoning_len={len(reasoning_content) if isinstance(reasoning_content, str) else 0} "
                        f"inline_think_len={len(inline_think_content) if isinstance(inline_think_content, str) else 0} "
                        f"thinking_source={thinking_source} "
                        f"trace_len={len(trace_content) if isinstance(trace_content, str) else 0} "
                        f"provider_reasoning_available={(gateway_meta or {}).get('provider_reasoning_available') if isinstance(gateway_meta, dict) else None} "
                        f"inline_think_extracted={(gateway_meta or {}).get('inline_think_extracted') if isinstance(gateway_meta, dict) else None} "
                        f"metacog_count={len(metacog_traces) if metacog_traces else 0} "
                        f"preview={_preview_text(reasoning_content or trace_content)}",
                        flush=True,
                    )
                    selected_reasoning_trace, selected_reasoning_source = select_reasoning_trace_for_history(
                        correlation_id=trace_id,
                        reasoning_trace=explicit_reasoning_trace,
                        metacog_traces=metacog_traces,
                        reasoning_content=reasoning_content,
                        session_id=publish_session_id,
                        message_id=f"{trace_id}:assistant",
                        model=((gateway_meta or {}).get("model") if isinstance(gateway_meta, dict) else None),
                    )
                    explicit_reasoning_trace = selected_reasoning_trace
                    if _thought_debug_enabled():
                        first_trace = metacog_traces[0] if metacog_traces else {}
                        logger.info(
                            "THOUGHT_DEBUG_HUB stage=ws_ingress_shape corr=%s keys=%s shape=%s",
                            trace_id,
                            sorted(list(cortex_result_dump.keys())) if isinstance(cortex_result_dump, dict) else [],
                            {
                                "reasoning_content_exists": bool(str(reasoning_content or "").strip()),
                                "reasoning_content_len": _debug_len(reasoning_content),
                                "inline_think_content_exists": bool(str(inline_think_content or "").strip()),
                                "inline_think_content_len": _debug_len(inline_think_content),
                                "thinking_source": thinking_source,
                                "reasoning_trace_exists": bool(explicit_reasoning_trace),
                                "reasoning_trace_content_len": _debug_len((explicit_reasoning_trace or {}).get("content") if isinstance(explicit_reasoning_trace, dict) else None),
                                "metacog_traces_exists": bool(metacog_traces),
                                "metacog_first_trace_role": first_trace.get("trace_role") if isinstance(first_trace, dict) else None,
                                "metacog_first_trace_stage": first_trace.get("trace_stage") if isinstance(first_trace, dict) else None,
                                "metacog_first_trace_content_len": _debug_len(first_trace.get("content") if isinstance(first_trace, dict) else None),
                                "selected_reasoning_source": selected_reasoning_source,
                            },
                        )
                    logger.info(
                        "hub_metacog_received corr=%s source=ws traces=%s",
                        trace_id,
                        len(metacog_traces),
                    )
                    workflow = extract_workflow_payload(resp.cortex_result)
                    autonomy_payload = extract_autonomy_payload(resp.cortex_result)
                    log_autonomy_payload_extraction(
                        correlation_id=trace_id,
                        cortex_result=resp.cortex_result,
                        payload=autonomy_payload if isinstance(autonomy_payload, dict) else {},
                        source="ws",
                    )
                    workflow_metadata_only = bool(
                        isinstance(workflow, dict)
                        and str(
                            workflow.get("id")
                            or workflow.get("workflow_id")
                            or workflow.get("raw_metadata", {}).get("workflow_id")
                            or ""
                        ).strip().lower() == "dream_cycle"
                    )
                    if workflow_metadata_only:
                        # Dream workflow is rendered as card-only metadata in Hub.
                        orion_response_text = ""
                    if isinstance(workflow, dict):
                        logger.info(
                            "hub_workflow_response corr=%s workflow_id=%s status=%s scheduled_count=%s persisted_count=%s rendered_path=%s source=ws",
                            trace_id,
                            workflow.get("workflow_id"),
                            workflow.get("status"),
                            len(workflow.get("scheduled") or []),
                            len(workflow.get("persisted") or []),
                            "scheduled_confirmation" if len(workflow.get("scheduled") or []) else "immediate_or_unscheduled",
                        )
                else:
                    workflow_metadata_only = False
                # If the model echoes "Orion:" due to our prompt format, strip it.
                s = (orion_response_text or "").lstrip()
                if s.startswith("Orion:"):
                    orion_response_text = s[len("Orion:"):].lstrip()
                logger.info(
                    "voice.chat.done corr=%s session_id=%s response_len=%d",
                    trace_id,
                    session_id,
                    len(orion_response_text or ""),
                )
                if resp is not None and hasattr(resp, "cortex_result") and resp.cortex_result:
                    trace_verb = str(
                        ((resp.cortex_result.metadata or {}).get("trace_verb") if isinstance(resp.cortex_result.metadata, dict) else None)
                        or resp.cortex_result.verb
                        or trace_verb
                    )
            except Exception as e:
                logger.error("voice.chat.error corr=%s session_id=%s err=%s", trace_id, session_id, e)
                err_payload = await _with_biometrics(
                    {"error": f"Chat failed: {str(e)}", "state": "idle"},
                    cache=biometrics_cache,
                )
                if not await _safe_ws_send_json(websocket, err_payload):
                    logger.info("chat_rpc_error_not_delivered_ws_closed corr=%s", trace_id)
                continue

            # 3. Response & Logging
            recall_count = 0
            backend_counts = None
            if isinstance(recall_debug, dict):
                recall_count = int(recall_debug.get("count") or 0)
                backend_counts = recall_debug.get("backend_counts")
                if backend_counts is None and isinstance(recall_debug.get("debug"), dict):
                    backend_counts = recall_debug["debug"].get("backend_counts")
            memory_used = False
            cortex_res = resp.cortex_result if resp is not None and getattr(resp, "cortex_result", None) else None
            if cortex_res is not None:
                memory_used = bool(getattr(cortex_res, "memory_used", False))
            if not memory_used:
                memory_used = bool(recall_count)
            ingress_status = (
                getattr(cortex_res, "status", None)
                if cortex_res is not None
                else ("ok" if used_context_exec_lane else None)
            )
            logger.info(
                "hub_ingress_result corr=%s sid=%s mode=%s status=%s final_len=%s memory_used=%s recall_count=%s context_exec_lane=%s",
                trace_id,
                session_id,
                mode,
                ingress_status,
                len(orion_response_text or ""),
                memory_used,
                recall_count,
                used_context_exec_lane,
            )
            _rec_tape_rsp(
                corr_id=trace_id,
                memory_used=memory_used,
                recall_count=recall_count,
                backend_counts=backend_counts,
                memory_digest=memory_digest,
            )
            try:
                from scripts.api_routes import record_chat_turn_pressure_telemetry

                record_chat_turn_pressure_telemetry(
                    correlation_id=trace_id,
                    route_debug=route_debug,
                    autonomy_payload=autonomy_payload if isinstance(autonomy_payload, dict) else {},
                    recall_debug=recall_debug if isinstance(recall_debug, dict) else {},
                    source_event_id=f"chat_result_ws:{trace_id}",
                )
            except Exception as exc:
                logger.warning("ws_pressure_telemetry_record_failed corr=%s error=%s", trace_id, exc)
            cortex_corr_id = (
                getattr(resp.cortex_result, "correlation_id", None)
                if resp is not None and resp.cortex_result
                else None
            )
            raw_meta = cortex_result_dump.get("metadata") if isinstance(cortex_result_dump, dict) else {}
            root_corr = raw_meta.get("root_correlation_id") if isinstance(raw_meta, dict) else None
            from scripts.api_routes import _chat_turn_trace_linkage

            trace_linkage = _chat_turn_trace_linkage(
                hub_corr_id=str(trace_id),
                cortex_corr_id=cortex_corr_id,
                root_correlation_id=str(root_corr).strip() if root_corr else None,
            )
            if is_social_room_payload(data):
                _social_room_id = str((data.get("external_room") or {}).get("room_id") or "").strip()
                if _social_room_id and route_debug:
                    social_room_inspection_cache.store(_social_room_id, route_debug)
            ws_payload = {
                "llm_response": orion_response_text,
                # Parity with HTTP /api/chat: lets the UI coalesce if primary string ever lags `raw.final_text`
                "raw": cortex_result_dump if isinstance(cortex_result_dump, dict) else {},
                "mode": mode,
                "correlation_id": trace_id,
                "trace_linkage": trace_linkage,
                "memory_digest": memory_digest,
                "memory_used": memory_used,
                "recall_debug": recall_debug,
                "agent_trace": agent_trace,
                "workflow": workflow,
                "workflow_metadata_only": workflow_metadata_only,
                "no_write": no_write,
                "routing_debug": route_debug,
                "context_exec_lane": used_context_exec_lane,
                "metacog_traces": metacog_traces,
                "reasoning_content": reasoning_content,
                "inline_think_content": inline_think_content,
                "thinking_source": thinking_source,
                "reasoning_trace": explicit_reasoning_trace,
                **autonomy_payload,
            }
            if substrate_summary is not None:
                ws_payload["substrate_effect_summary"] = substrate_summary
            if used_agent_claude_lane:
                agent_meta = route_debug.get("agent_claude") if isinstance(route_debug, dict) else {}
                ws_payload["metadata"] = agent_meta if isinstance(agent_meta, dict) else {}
                ws_payload["context_exec_lane"] = False
            if mode == "council" or settings.HUB_DEBUG_COUNCIL:
                council_debug = _extract_council_debug_from_result(resp)
                if council_debug:
                    ws_payload["council_debug"] = council_debug
            await websocket.send_json(await _with_biometrics(ws_payload, cache=biometrics_cache))

            # Auto-invite Claude to react to the turn that just landed. Fired
            # after the payload is sent so Orion's reply always renders first
            # -- Claude is reacting to the room, not racing it.
            #
            # Fire-and-forget: a room companion that is slow, down, or
            # rate-gated must never delay or fail Orion's own turn.
            try:
                _room_relay = getattr(scripts.main, "room_claude_relay", None)
                _orion_said = str(ws_payload.get("llm_response") or "").strip()
                if (
                    _room_relay is not None
                    and _room_relay.enabled
                    # should_fire_auto_invite checks _orion_said before touching
                    # the rate gate -- see its docstring for why the ordering
                    # matters (a workflow-only turn must not burn the window).
                    and _room_relay.should_fire_auto_invite(_orion_said, session_id, time.time())
                ):
                    # Send the EXCHANGE, not just Orion's half. Sending only
                    # Orion's reply left Claude watching a one-sided
                    # monologue with no idea what had been asked -- it could
                    # only react to Orion's tone, which is what "generic
                    # Claude responses" actually was. Confirmed by reading
                    # Claude's own session transcript.
                    #
                    # `prompt` is the raw reply: build_turn_prompt adds the
                    # speaker prefix, and prefixing here too produced
                    # "Or\u00edon: Or\u00edon: ..." in the live transcript.
                    _juniper_said = str(transcript or "").strip()
                    _exchange = (
                        [{
                            "speaker_id": "juniper",
                            "speaker_name": "Juniper",
                            "speaker_kind": "human",
                            "text": _juniper_said,
                        }]
                        if _juniper_said
                        else []
                    )
                    asyncio.create_task(
                        _room_relay.invite(
                            prompt=_orion_said,
                            invited_by="Or\u00edon",
                            session_id=session_id,
                            room_id=settings.HUB_ROOM_CLAUDE_ROOM_ID,
                            trigger="auto",
                            transcript=_exchange,
                            connection_id=connection_id,
                        )
                    )
            except Exception:
                logger.debug("room_claude_auto_invite_failed", exc_info=True)

            # Log to SQL (Best Effort) & Trigger Introspection
            if bus and not no_write and not workflow_metadata_only:
                enriched_client_meta = dict(turn_client_meta)
                selected_reasoning_trace = explicit_reasoning_trace
                try:
                    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

                    if is_social_room_payload(data):
                        enriched_client_meta.update(
                            social_room_client_meta(
                                payload=data,
                                route_debug=route_debug,
                                trace_verb=trace_verb,
                                memory_digest=memory_digest,
                            )
                        )

                    # Extract rich metadata
                    gateway_meta = {}
                    if used_agent_claude_lane:
                        agent_meta = route_debug.get("agent_claude") if isinstance(route_debug, dict) else {}
                        gateway_meta = agent_meta if isinstance(agent_meta, dict) else {}
                    elif resp is not None and hasattr(resp, "cortex_result") and resp.cortex_result:
                        gateway_meta = resp.cortex_result.metadata or {}

                    # Include trace_verb in spark_meta for the Visualizer
                    spark_meta = {
                        "mode": mode,
                        "trace_verb": trace_verb,
                        "use_recall": use_recall,
                        "reasoning_content": reasoning_content,
                        "inline_think_content": inline_think_content,
                        "thinking_source": thinking_source,
                        "thought_capture_step": "llm_chat_general" if str(trace_verb or "").strip() == "chat_general" else None,
                        **(gateway_meta if isinstance(gateway_meta, dict) else {}),
                    }

                    chat_row = {
                        "id": trace_id,
                        "correlation_id": trace_id,
                        "source": "hub_ws",
                        "prompt": transcript,
                        "response": orion_response_text,
                        "user_id": data.get("user_id"),
                        "session_id": data.get("session_id"),
                        "spark_meta": spark_meta,
                    }
                    selected_reasoning_trace, _ = select_reasoning_trace_for_history(
                        correlation_id=trace_id,
                        reasoning_trace=explicit_reasoning_trace,
                        metacog_traces=metacog_traces,
                        reasoning_content=reasoning_content,
                        session_id=publish_session_id,
                        message_id=f"{trace_id}:assistant",
                        model=((gateway_meta or {}).get("model") if isinstance(gateway_meta, dict) else None),
                    )
                    if _thought_debug_enabled():
                        logger.info(
                            "THOUGHT_DEBUG_HUB stage=ws_chat_history_turn_payload corr=%s keys=%s shape=%s",
                            trace_id,
                            sorted(list((selected_reasoning_trace or {}).keys())) if isinstance(selected_reasoning_trace, dict) else [],
                            {
                                "reasoning_trace_exists": bool(selected_reasoning_trace),
                                "reasoning_trace_content_len": _debug_len((selected_reasoning_trace or {}).get("content") if isinstance(selected_reasoning_trace, dict) else None),
                                "reasoning_content_exists": bool(str(reasoning_content or "").strip()),
                                "reasoning_content_len": _debug_len(reasoning_content),
                                "inline_think_content_exists": bool(str(inline_think_content or "").strip()),
                                "inline_think_content_len": _debug_len(inline_think_content),
                                "thinking_source": thinking_source,
                                "metacog_traces_exists": bool(metacog_traces),
                            },
                        )

                    # 1. SQL Log (turn-level row: prompt + response)
                    env_turn = build_chat_turn_envelope(
                        prompt=transcript,
                        response=orion_response_text,
                        session_id=publish_session_id,
                        correlation_id=trace_id,
                        user_id=data.get("user_id"),
                        response_identity=(
                            ((gateway_meta or {}).get("model") or (gateway_meta or {}).get("fcc_model_label"))
                            if isinstance(gateway_meta, dict)
                            else None
                        ),
                        source_label="hub_ws",
                        spark_meta=spark_meta,
                        turn_id=trace_id,
                        memory_status="accepted",
                        memory_tier="ephemeral",
                        client_meta=enriched_client_meta,
                        reasoning_content=reasoning_content,
                        inline_think_content=inline_think_content,
                        thinking_source=thinking_source,
                        reasoning_trace=selected_reasoning_trace,
                    )
                    wrote_chat_history = bool(bus) and (not no_write)
                    if str(trace_verb or "").strip() == "chat_general":
                        logger.info(
                            "chat_general_thought_capture corr=%s step=llm_chat_general think_len=%s source=%s wrote_chat_history=%s",
                            trace_id,
                            len(str(inline_think_content or "").strip()),
                            thinking_source,
                            wrote_chat_history,
                        )
                    _schedule_publish(publish_chat_turn(bus, env_turn), "chat.history turn")
                    logger.info("Published chat.history turn row -> %s", settings.chat_history_turn_channel)
                    if metacog_traces:
                        from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

                        for trace in metacog_traces:
                            coerced_trace = _coerce_metacog_trace(trace)
                            if coerced_trace is None:
                                continue
                            trace_debug = trace if isinstance(trace, dict) else coerced_trace.model_dump(mode="json")
                            if _thought_debug_enabled():
                                logger.info(
                                    "THOUGHT_DEBUG_METACOG_PUB stage=hub_ws_prepare corr=%s trace_role=%s trace_stage=%s model=%s content_len=%s content_snippet=%r",
                                    trace_id,
                                    trace_debug.get("trace_role") or trace_debug.get("role"),
                                    trace_debug.get("trace_stage") or trace_debug.get("stage"),
                                    trace_debug.get("model"),
                                    _debug_len(trace_debug.get("content")),
                                    _debug_snippet(trace_debug.get("content")),
                                )
                            trace_env = BaseEnvelope(
                                kind="metacognitive.trace.v1",
                                source=ServiceRef(
                                    name=settings.SERVICE_NAME,
                                    node=settings.NODE_NAME,
                                    version=settings.SERVICE_VERSION,
                                ),
                                correlation_id=trace_id,
                                payload=coerced_trace,
                            )
                            _schedule_publish(bus.publish("orion:metacog:trace", trace_env), "metacog.trace")
                        if _thought_debug_enabled() and not any(isinstance(t, dict) for t in metacog_traces):
                            logger.info("THOUGHT_DEBUG_METACOG_PUB stage=hub_ws_skipped corr=%s reason=no_valid_trace_dicts", trace_id)
                        logger.info(
                            "hub_metacog_published corr=%s source=ws channel=%s traces=%s",
                            trace_id,
                            "orion:metacog:trace",
                            len(metacog_traces),
                        )
                    if is_social_room_payload(data):
                        _schedule_publish(
                            publish_social_room_turn(
                                bus,
                                prompt=transcript,
                                response=orion_response_text,
                                session_id=publish_session_id,
                                correlation_id=trace_id,
                                user_id=data.get("user_id"),
                                source_label="hub_ws",
                                recall_profile=recall_payload.get("profile"),
                                trace_verb=trace_verb,
                                client_meta=enriched_client_meta,
                                memory_digest=memory_digest,
                            ),
                            "chat.social turn",
                        )
                    # 2026-07-28: spark.candidate publish removed (spark-
                    # introspector retirement). This chat-history/chat-turn
                    # publish above already feeds orion-vector-host's real
                    # OrionTissue physics feed (app/tissue_feed.py) in-process
                    # on every semantic upsert -- no separate candidate event
                    # is needed to drive the Cognitive EKG anymore.

                except Exception as e:
                    logger.warning(f"Failed to log/introspect chat: {e}")

                # Publish assistant reply into chat history
                try:
                    gateway_meta = {}
                    if used_agent_claude_lane:
                        agent_meta = route_debug.get("agent_claude") if isinstance(route_debug, dict) else {}
                        gateway_meta = agent_meta if isinstance(agent_meta, dict) else {}
                    elif resp is not None and hasattr(resp, "cortex_result") and resp.cortex_result:
                        gateway_meta = resp.cortex_result.metadata or {}
                    history_correlation_id = trace_id
                    if resp is not None and hasattr(resp, "cortex_result") and resp.cortex_result:
                        history_correlation_id = (
                            getattr(resp.cortex_result, "correlation_id", None) or trace_id
                        )
                    selected_reasoning_trace, _ = select_reasoning_trace_for_history(
                        correlation_id=trace_id,
                        reasoning_trace=explicit_reasoning_trace,
                        metacog_traces=metacog_traces,
                        reasoning_content=reasoning_content,
                        session_id=publish_session_id,
                        message_id=f"{trace_id}:assistant",
                        model=((gateway_meta or {}).get("model") if isinstance(gateway_meta, dict) else None)
                        or (
                            route_debug.get("agent_claude", {}).get("fcc_model_label")
                            if isinstance(route_debug, dict)
                            and isinstance(route_debug.get("agent_claude"), dict)
                            else None
                        ),
                    )
                    assistant_env = build_chat_history_envelope(
                        content=orion_response_text,
                        role="assistant",
                        session_id=publish_session_id,
                        correlation_id=history_correlation_id,
                        speaker=gateway_meta.get("speaker") or settings.SERVICE_NAME,
                        model=gateway_meta.get("model") or gateway_meta.get("fcc_model_label"),
                        provider=gateway_meta.get("provider"),
                        tags=[client_mode if used_agent_claude_lane else mode, trace_verb],
                        message_id=f"{trace_id}:assistant",
                        memory_status="accepted",
                        memory_tier="ephemeral",
                        client_meta=enriched_client_meta,
                        reasoning_trace=selected_reasoning_trace,
                    )
                    _schedule_publish(publish_chat_history(bus, [assistant_env]), "chat.history assistant")
                except Exception as e:
                    logger.warning("Failed to publish assistant chat history: %s", e, exc_info=True)

            # 4. TTS
            will_tts = dispatch_tts_reply(
                text=orion_response_text,
                disable_tts=disable_tts,
                tts_client=tts_client,
                tts_q=tts_q,
                correlation_id=trace_id,
                session_id=session_id,
                lane="classic",
                extra_gate=not workflow_metadata_only,
                log_extra={"workflow_metadata_only": workflow_metadata_only},
            )
            if not will_tts and not disable_tts and orion_response_text:
                tts_debug_payload = await _with_biometrics(
                    {
                        "tts_debug": {
                            "stage": "hub_decision",
                            "will_tts": False,
                            "response_len": len(orion_response_text or ""),
                            "workflow_metadata_only": workflow_metadata_only,
                            "disable_tts": disable_tts,
                            "has_tts_client": bool(tts_client),
                        },
                    },
                    cache=biometrics_cache,
                )
                if not await _safe_ws_send_json(websocket, tts_debug_payload):
                    continue

            if orion_response_text and not workflow_metadata_only:
                history.append({"role": "assistant", "content": orion_response_text})

            # Keep history bounded (system + last 2*turns messages)
            try:
                keep_msgs = 1 + (2 * max(0, int(turns)))
                if len(history) > keep_msgs:
                    history[:] = history[:1] + history[-(2 * turns):]
            except Exception:
                pass
            await websocket.send_json(await _with_biometrics({"state": "idle"}, cache=biometrics_cache))

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
        corr = active_turn.get("correlation_id")
        kind = active_turn.get("kind")
        if corr:
            await cancel_in_flight_turn(
                bus=rpc_bus or bus,
                correlation_id=str(corr),
                kind=str(kind or "orion"),
                reason="client_disconnect",
            )
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        corr = active_turn.get("correlation_id")
        kind = active_turn.get("kind")
        if corr:
            await cancel_in_flight_turn(
                bus=rpc_bus or bus,
                correlation_id=str(corr),
                kind=str(kind or "orion"),
                reason="ws_error",
            )
    finally:
        _ACTIVE_TURNS_BY_CONNECTION.pop(connection_id, None)
        if endogenous_outreach is not None:
            endogenous_outreach.unregister_connection(connection_id)
        _room_relay = getattr(scripts.main, "room_claude_relay", None)
        if _room_relay is not None:
            _room_relay.unregister_connection(connection_id)
        drain_task.cancel()
        if notification_cache is not None:
            notification_cache.unregister_queue(tts_q)
        if presence_state:
            presence_state.disconnected()
