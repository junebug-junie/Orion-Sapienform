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

**Consent model: the mic press is the per-turn act, so this ships ON for
spoken turns.** Gated by ``AFFECT_CHAT_TURN_SCOPE`` (off | voice | all),
shipped default ``voice`` -- meaning it fires on turns Juniper actually
spoke, and no others, from the first boot after deploy. Saying it plainly
because an earlier draft of this docstring claimed "off by default at the
service boundary" while the code shipped ``voice``, which was simply false
(review finding, 2026-08-26).

Why ``voice`` rather than ``off``: Juniper asked for this directly, and the
right analogue is the manual "Check now" button, not the ambient toggle.
The ambient loop resets ``state.enabled`` to False on every restart
precisely because it records with no human in the loop for as long as it is
on; the manual button needs no such reset because each capture is preceded
by a deliberate human action. This path is the latter shape -- pressing the
microphone is an explicit, physical, per-turn act, and a typed turn (which
has no such act) is excluded for exactly that reason.

What that consent argument does NOT cover, stated rather than buried:
``all`` would fire on typed turns, where no per-turn physical act exists,
and there is no UI toggle or per-session gate here -- ``off`` is an env
change plus a restart, not a click. Widening is deliberately left as a
conscious operator decision.

The scope resolver fails CLOSED: anything not exactly off/voice/all becomes
``off``, never the ``voice`` default. A typo must not start a camera.
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

# The in-flight PRE leg, keyed by chat correlation_id.
#
# Exists because the post leg would otherwise lose the shared capture slot
# to its own pre leg on most turns, and the matched pair this whole module
# exists to produce would simply not exist (review finding, 2026-08-26).
# The arithmetic: _capture_blocking holds the lock for the entire
# capture_and_assess round trip -- ~8s retina clip plus ~20s warm AffectGPT
# inference, up to ~195s cold or degraded. Any turn that finishes inside
# that window fires its post leg straight into a held lock and is dropped.
# Plenty of Orion turns finish in well under 28s.
#
# So the post leg AWAITS the pre leg rather than racing it. That is a real
# behavioural choice, not just a retry: the post capture is meant to read
# Juniper AFTER the reply landed, and starting it a few seconds later than
# the reply is fine, whereas not taking it at all is the failure mode that
# makes the pair useless. Still bounded -- the pre leg has its own HTTP
# timeout, so this cannot wait forever.
_PENDING_PRE: dict[str, asyncio.Task] = {}


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
    *,
    base_url: str,
    timeout_sec: float,
    trigger: str,
    correlation_id: str,
    subtitle: Optional[str] = None,
) -> None:
    """Runs in a worker thread (asyncio.to_thread). Claims the shared slot,
    calls the ONE shared HTTP call site, and always releases via end_capture
    so a failure here can never strand the lock and wedge every later
    capture (manual, ambient, or chat-turn) behind it.
    """
    if not vision_affect_ambient.try_begin_capture(trigger, record_state=False):
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
            subtitle=subtitle,
        )
        ok, error = vision_affect_ambient.result_ok_and_error(body)
        raw_response, video_sha256 = vision_affect_ambient.result_content(body)
        vision_affect_ambient.end_capture(
            ok=ok,
            error=error,
            raw_response=raw_response,
            video_sha256=video_sha256,
            record_state=False,
        )
        logger.info(
            "[HUB] chat_turn_affect_done trigger=%s corr=%s ok=%s error=%s",
            trigger,
            correlation_id,
            ok,
            error,
        )
    except requests.RequestException as exc:
        vision_affect_ambient.end_capture(ok=False, error=str(exc), record_state=False)
        logger.warning(
            "[HUB] chat_turn_affect_transport_error trigger=%s corr=%s error=%s",
            trigger,
            correlation_id,
            exc,
        )
    except Exception as exc:  # never let a detached task die unexplained
        vision_affect_ambient.end_capture(ok=False, error=str(exc), record_state=False)
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
    subtitle: Optional[str] = None,
) -> Optional[asyncio.Task]:
    """Fire-and-forget one capture. Returns the task (tests await it; the
    turn path ignores it) or None when nothing was fired.

    Deliberately synchronous and non-async: the callers are in the middle of
    ``websocket_handler``'s turn path and must not gain an await point here.
    """
    if not should_fire(settings, is_voice_turn=is_voice_turn):
        return None
    if not is_voice_turn:
        # Scope "all" fires on typed turns too, but a typed message was never
        # spoken. Passing it through would render "the person said this around
        # the time these frames were captured: ..." to a VL model reading her
        # face, about text she silently typed. The frames are still worth
        # reading on a text turn; the words are not hers to attribute to speech.
        subtitle = None
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
    # Guarded because the POST-turn call site is inside websocket_handler's
    # `finally` block. An exception escaping from there does not just lose
    # the capture -- it REPLACES whatever exception the turn was already
    # unwinding with, so a real turn failure would be reported as an affect
    # error instead. asyncio.create_task raises RuntimeError when there is
    # no running loop or the loop is closing, which is exactly the state a
    # socket teardown can be in. An advisory capture must never be able to
    # corrupt the turn's own error reporting.
    async def _run() -> None:
        # The post leg waits for this turn's own pre leg to finish before
        # asking for the slot. See _PENDING_PRE. Only ever waits on THIS
        # turn's pre leg -- never on an unrelated capture -- so a busy
        # ambient loop still just costs this leg its slot, as designed.
        if trigger == TRIGGER_POST:
            pre = _PENDING_PRE.get(correlation_id)
            if pre is not None and not pre.done():
                logger.info(
                    "[HUB] chat_turn_affect_post_waiting_for_pre corr=%s",
                    correlation_id,
                )
                # shield=False on purpose: if this post task is cancelled
                # (Hub shutting down), stop waiting rather than pinning
                # shutdown behind a GPU call.
                try:
                    await asyncio.wait_for(asyncio.shield(pre), timeout=timeout_sec)
                except asyncio.TimeoutError:
                    logger.warning(
                        "[HUB] chat_turn_affect_post_gave_up_waiting corr=%s",
                        correlation_id,
                    )
                except Exception:
                    # The pre leg failing is not a reason to skip the post
                    # leg -- it is a reason the post leg is the only read
                    # this turn will have.
                    pass
        await asyncio.to_thread(
            _capture_blocking,
            base_url=base,
            timeout_sec=timeout_sec,
            trigger=trigger,
            correlation_id=correlation_id,
            subtitle=subtitle,
        )

    try:
        task = asyncio.create_task(
            _run(),
            name=f"chat-turn-affect-{trigger}-{correlation_id}",
        )
    except RuntimeError as exc:
        logger.warning(
            "[HUB] chat_turn_affect_not_scheduled trigger=%s corr=%s error=%s",
            trigger,
            correlation_id,
            exc,
        )
        return None
    _INFLIGHT.add(task)
    task.add_done_callback(_INFLIGHT.discard)
    if trigger == TRIGGER_PRE:
        _PENDING_PRE[correlation_id] = task
        # Keyed by correlation_id, so this dict must self-clean or it is a
        # slow leak across the life of a long-running Hub process.
        task.add_done_callback(lambda _t: _PENDING_PRE.pop(correlation_id, None))
    logger.info(
        "[HUB] chat_turn_affect_fired trigger=%s corr=%s voice=%s",
        trigger,
        correlation_id,
        is_voice_turn,
    )
    return task
