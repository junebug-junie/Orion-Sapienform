"""Per-chat-turn affect bracket -- one AffectGPT capture before an Orion-mode
turn runs, one after its reply is handed back.

Juniper's ask, 2026-08-25, immediately downstream of turning the microphone
on for the unified turn: "I want it to trigger affect service to also record
affect. then after it returns the chat message, I want it to trigger a follow
on affect record."

What this is FOR, concretely: PR #1865 already lets a recent affect read
colour a turn's prompt, but the only things that ever produced a read were
Juniper pressing "Check now" or leaving the ambient toggle on -- both
untethered from any particular conversation. Bracketing a turn produces the
one thing neither of those can: a *matched pair* around a known stimulus, so
"how did Juniper's affect move across this exchange" becomes answerable from
stored events instead of guessed. That pairing is what
``chat_correlation_id`` (JuniperMultimodalAffectV1) exists to make joinable;
observed_at-proximity cannot do it, because a concurrent ambient tick lands
in the same time window and is indistinguishable by timestamp alone.

Three design constraints this module exists to satisfy:

**Never block the turn.** ``capture_and_assess`` is synchronous and can take
up to ~195s worst case (retina's ~8s clip + its own timeout ceiling, then
AffectGPT inference: ~20s warm, more cold). Hanging a chat turn on that would
be indefensible, so every fire here is a detached ``asyncio.create_task`` and
the turn never awaits it. The direct consequence, stated plainly rather than
papered over: **the pre-turn capture does NOT colour the turn that fired it.**
It cannot -- the reply is usually already streaming before the GPU has
answered. What it does is land inside ``orion/situational/juniper_affect_state``'s
300s TTL mirror in time for the NEXT turn, and give the post-turn capture
something to be compared against.

**Share the existing exclusive capture slot.** These captures drive the same
single physical webcam+mic on carbon as the manual button and the ambient
loop, so they claim ``vision_affect_ambient``'s lock, not a second one. A
caller that loses the slot is dropped, never queued and never retried --
the same no-retry policy vision_affect_ambient's own module docstring
already establishes for a live recording trigger, and for the same reason:
hammering retries at a camera is the wrong instinct. A dropped post-turn
capture is a real, logged gap, not a silent success.

**Off by default at the service boundary, not just in the UI.** Gated by
``AFFECT_CHAT_TURN_SCOPE`` (off | voice | all, default "voice"). "voice"
means only turns Juniper actually spoke -- the mic press is an explicit,
per-turn, physical consent action in a way that typing into an already-open
tab is not, and that is the whole reason it is the default rather than
"all". Widening to every Orion-mode text turn is one env change, deliberately
left as a conscious operator decision.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

import requests

from . import vision_affect_ambient

logger = logging.getLogger("orion-hub.chat_turn_affect")

# The two labels this module is allowed to publish. Kept as a module
# constant rather than inlined at the two call sites so the service-side
# Literal (orion/schemas/affectgpt.py::JuniperMultimodalAffectV1.trigger)
# has exactly one counterpart here to stay in sync with.
TRIGGER_PRE = "chat_turn_pre"
TRIGGER_POST = "chat_turn_post"

_VALID_SCOPES = frozenset({"off", "voice", "all"})

# Strong references to in-flight fire-and-forget tasks.
#
# NOT bookkeeping/decoration: asyncio only holds a WEAK reference to a running
# task, so a task nobody keeps a reference to can be garbage-collected
# mid-flight and simply vanish (documented behaviour -- see the warning under
# asyncio.create_task in the stdlib docs). Both call sites in
# websocket_handler.py deliberately discard the return value of fire(), which
# is exactly the shape that triggers it. Without this set, an affect capture
# could disappear silently partway through, leaving the shared capture lock
# held forever and wedging every subsequent capture -- manual, ambient and
# chat-turn alike -- behind a task that no longer exists.
#
# Self-trimming via the done-callback below, so this never grows: at most two
# entries are ever live at once (one turn's pre and post legs), and the
# capture lock already prevents more than one from actually running.
_INFLIGHT: set[asyncio.Task] = set()


def resolve_scope(settings: Any) -> str:
    """Normalize AFFECT_CHAT_TURN_SCOPE. An unrecognized value falls back to
    "off", NOT to the "voice" default: a typo in an env var must never
    silently start recording Juniper's webcam. Fail-closed is the only
    defensible direction for this particular knob.
    """
    raw = str(getattr(settings, "AFFECT_CHAT_TURN_SCOPE", "") or "").strip().lower()
    if raw in _VALID_SCOPES:
        return raw
    if raw:
        logger.warning(
            "[HUB] chat_turn_affect_bad_scope value=%r -- falling back to 'off'", raw
        )
    return "off"


def should_fire(settings: Any, *, is_voice_turn: bool) -> bool:
    """Whether this turn is in scope. Split out from the fire path so a test
    can assert the policy without touching HTTP, threads, or the lock."""
    scope = resolve_scope(settings)
    if scope == "off":
        return False
    if scope == "voice":
        return bool(is_voice_turn)
    return True  # "all"


def _capture_blocking(
    *, base_url: str, timeout_sec: float, trigger: str, correlation_id: str
) -> None:
    """Runs in a worker thread (asyncio.to_thread). Claims the shared slot,
    calls the ONE shared HTTP call site, and always releases via end_capture
    so a failure here can never strand the lock and wedge every later
    capture (manual, ambient, or chat-turn) behind it.
    """
    if not vision_affect_ambient.try_begin_capture(trigger):
        # Not an error worth raising: a capture is already in flight (the
        # ambient loop, the manual button, or this turn's own sibling
        # fire). Logged so a missing half of a pre/post pair is explainable
        # after the fact rather than looking like the fire never happened.
        logger.info(
            "[HUB] chat_turn_affect_skipped trigger=%s corr=%s reason=capture_in_progress",
            trigger,
            correlation_id,
        )
        return
    try:
        body: Dict[str, Any] = vision_affect_ambient.call_capture_and_assess(
            base_url,
            timeout_sec,
            trigger,
            chat_correlation_id=correlation_id,
        )
        ok, error = vision_affect_ambient.result_ok_and_error(body)
        raw_response, video_sha256 = vision_affect_ambient.result_content(body)
        vision_affect_ambient.end_capture(
            ok=ok, error=error, raw_response=raw_response, video_sha256=video_sha256
        )
        logger.info(
            "[HUB] chat_turn_affect_done trigger=%s corr=%s ok=%s error=%s",
            trigger,
            correlation_id,
            ok,
            error,
        )
    except requests.RequestException as exc:
        vision_affect_ambient.end_capture(ok=False, error=str(exc))
        logger.warning(
            "[HUB] chat_turn_affect_transport_error trigger=%s corr=%s error=%s",
            trigger,
            correlation_id,
            exc,
        )
    except Exception as exc:  # never let a detached task die unexplained
        vision_affect_ambient.end_capture(ok=False, error=str(exc))
        logger.warning(
            "[HUB] chat_turn_affect_error trigger=%s corr=%s error=%s",
            trigger,
            correlation_id,
            exc,
        )


def fire(
    *,
    settings: Any,
    trigger: str,
    correlation_id: str,
    is_voice_turn: bool,
) -> Optional[asyncio.Task]:
    """Fire-and-forget one capture. Returns the task (tests await it; the
    turn path ignores it) or None when nothing was fired.

    Deliberately synchronous and non-async: the callers are in the middle of
    ``websocket_handler``'s turn path and must not gain an await point here.
    """
    if not should_fire(settings, is_voice_turn=is_voice_turn):
        return None
    base = str(getattr(settings, "JUNIPER_AFFECTIVE_STATE_BASE_URL", "") or "").strip().rstrip("/")
    if not base:
        # Same honest-degradation contract the manual route already has
        # (it 503s on this) -- here there is no caller to tell, so it is a
        # log line, not a silent no-op.
        logger.info(
            "[HUB] chat_turn_affect_unconfigured trigger=%s corr=%s reason=base_url_not_set",
            trigger,
            correlation_id,
        )
        return None
    timeout_sec = float(getattr(settings, "JUNIPER_AFFECTIVE_STATE_TIMEOUT_SEC", 240.0))
    task = asyncio.create_task(
        asyncio.to_thread(
            _capture_blocking,
            base_url=base,
            timeout_sec=timeout_sec,
            trigger=trigger,
            correlation_id=correlation_id,
        ),
        name=f"chat-turn-affect-{trigger}-{correlation_id}",
    )
    _INFLIGHT.add(task)
    task.add_done_callback(_INFLIGHT.discard)
    logger.info(
        "[HUB] chat_turn_affect_fired trigger=%s corr=%s voice=%s",
        trigger,
        correlation_id,
        is_voice_turn,
    )
    return task
