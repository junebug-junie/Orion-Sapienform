"""Ambient (recurring) AffectGPT capture -- the Vision panel's toggle.

Design correction, 2026-08-22: this was originally shipped as a one-shot
"Affect check" button, which silently substituted for the toggle Juniper had
actually asked for and approved ("a toggle that periodically grabs a clip...
while on"). This module is the real toggle: flip it on, it keeps calling
orion-juniper-affective-state's ``capture_and_assess`` every
``AFFECT_AMBIENT_INTERVAL_SEC`` while on; flip it off, it stops. The
one-shot button stays too (renamed "Check now" in the UI) -- it was correct
on its own terms, just not a substitute for this.

**Hub owns this loop, not the orchestrator (circe).** Explicit direction
from Juniper, 2026-08-22, overriding an earlier draft of this design that
put the loop on circe. A browser-side ``setInterval`` was never actually
built and would have been wrong anyway (dies on tab close, multiplies with
multiple open Hub tabs) -- this is a real server-side loop, living in Hub's
own FastAPI process, following the exact shape
``scripts/main.py``'s ``_run_substrate_topic_foundry_scheduler`` already
uses (a background ``asyncio.create_task`` loop started at startup,
cancelled at shutdown) -- except THIS one's ``enabled`` flag is
runtime-toggleable via an HTTP route, not just an env-set-at-startup switch,
which is the whole point of a UI toggle.

**Fails closed on restart by construction, not by extra code.** ``enabled``
is a plain in-memory flag on this module's ``state`` singleton -- a Hub
restart resets it to False for free, with no persistence layer needed.
Deliberate (Juniper approved this explicitly): never silently resume
recording Juniper's face/voice after a crash or redeploy: an operator has to
consciously flip it back on.

**No retries on a failed tick, by design (Juniper's explicit instruction,
2026-08-22).** A failed attempt just waits for the next scheduled one --
hammering retries on every failure would be the wrong instinct for a live
webcam+mic recording trigger. A collision with the manual "Check now" button
(see try_begin_capture below) is handled the same way -- it's a real failure
for that cycle, not something to retry immediately.

**Manual and ambient captures share one exclusive slot (review finding,
2026-08-22).** Originally the manual route bypassed this module's state
entirely and only retina's own device-level lock caught a collision (a
confusing generic "busy" error, and the ambient loop losing its cycle with
no record of why). try_begin_capture()/end_capture() below is a real
threading.Lock, not just the GIL-atomic boolean flag this started as --
the manual route runs in FastAPI's threadpool (a real OS thread), the
ambient loop runs on the asyncio event loop thread, so this is a genuine
cross-thread race, not just a single-threaded check-then-act. Both paths
now update the SAME state, so the status line reflects manual activity too,
not just ambient ticks.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger("orion-hub.vision_affect_ambient")


@dataclass
class AffectAmbientState:
    enabled: bool = False
    tick_in_progress: bool = False
    tick_count: int = 0
    last_attempt_at: Optional[float] = None
    last_trigger: Optional[str] = None
    last_result_ok: Optional[bool] = None
    last_error: Optional[str] = None
    # The actual content of the last SUCCESSFUL attempt (manual or ambient)
    # -- added 2026-08-22 for the Vision panel's "Carbon (affect snapshot)"
    # dropdown option (Juniper's ask: show the most recent affect result,
    # not just whether the last tick succeeded). Left as-is (not cleared)
    # on a failed attempt -- a failure doesn't erase the last real reading.
    last_raw_response: Optional[str] = None
    last_video_sha256: Optional[str] = None

    def status_payload(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "tick_in_progress": self.tick_in_progress,
            "tick_count": self.tick_count,
            "last_attempt_at": self.last_attempt_at,
            "last_trigger": self.last_trigger,
            "last_result_ok": self.last_result_ok,
            "last_error": self.last_error,
            "last_raw_response": self.last_raw_response,
            "last_video_sha256": self.last_video_sha256,
        }


# Module-level singletons, same convention as biometrics_cache.py's own
# process-wide cache object -- one Hub process, one ambient state, no need
# for a class instance threaded through every call site.
state = AffectAmbientState()
# threading.Lock, NOT asyncio.Lock -- the manual "Check now" route runs
# synchronously in FastAPI's threadpool (a real OS thread), while the
# ambient loop runs on the asyncio event loop thread. An asyncio.Lock only
# protects against concurrent coroutines on the SAME event loop; it does
# nothing for a genuine cross-thread race. This does.
_capture_lock = threading.Lock()


def try_begin_capture(trigger: str, *, record_state: bool = True) -> bool:
    """Non-blocking: returns True if this caller won the exclusive capture
    slot (manual, ambient, or a chat-turn bracket leg -- whichever asks
    first), False if another capture is already in flight. Never queues --
    a caller that loses just reports "busy" and, for the ambient loop,
    waits for its next scheduled tick rather than retrying immediately.

    ``record_state=False`` takes the mutex WITHOUT touching this module's
    observable state, and is what the chat-turn bracket
    (scripts/chat_turn_affect.py) uses. Two real problems it avoids, both
    review findings 2026-08-26:

    1. ``affect_ambient_loop`` computes whether a tick is due from
       ``state.last_attempt_at`` against a 300s interval. A chat-turn
       capture bumping that field resets the ambient due-clock, so a
       conversation with a spoken turn more often than every five minutes
       would leave the ambient toggle reading "enabled" while never
       actually firing a tick of its own.
    2. ``last_trigger`` / ``last_result_ok`` / ``last_raw_response`` drive
       the Vision panel's status line and its "Carbon (affect snapshot)"
       view. Those exist to report what the OPERATOR started. The manual
       button has the same coupling, but it is human-paced and rare; a
       per-turn automatic capture would take the panel over.

    The mutex itself is still shared by every caller -- that part is the
    whole point, since all of them drive one physical camera.
    """
    acquired = _capture_lock.acquire(blocking=False)
    if not acquired:
        return False
    if record_state:
        state.tick_in_progress = True
        state.last_trigger = trigger
        state.last_attempt_at = time.time()
        state.tick_count += 1
    return True


def end_capture(
    *,
    ok: bool,
    error: Optional[str],
    raw_response: Optional[str] = None,
    video_sha256: Optional[str] = None,
    record_state: bool = True,
) -> None:
    """``record_state`` must match the value the paired try_begin_capture()
    was given -- otherwise a chat-turn capture would still write its result
    into the operator-facing panel fields it deliberately did not claim.
    The lock is ALWAYS released either way; that is not optional."""
    if not record_state:
        try:
            _capture_lock.release()
        except RuntimeError:
            logger.warning("[HUB] affect_end_capture_release_without_lock")
        return
    state.last_result_ok = ok
    state.last_error = error
    if ok:
        # Only overwrite on success -- a failed attempt's None/blank must
        # not erase the last real reading the "affect snapshot" view shows.
        state.last_raw_response = raw_response
        state.last_video_sha256 = video_sha256
    state.tick_in_progress = False
    try:
        _capture_lock.release()
    except RuntimeError:
        # Released without being held -- should not happen given every
        # caller only reaches here after a successful try_begin_capture(),
        # but never let bookkeeping crash the caller's own error handling.
        logger.warning("[HUB] affect_ambient_end_capture_release_without_lock")


def call_capture_and_assess(
    base_url: str,
    timeout_sec: float,
    trigger: str,
    *,
    chat_correlation_id: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> Dict[str, Any]:
    """Blocking HTTP call to the orchestrator -- callers off the event loop
    (the ambient loop below) run this via asyncio.to_thread. Shared by the
    manual "Check now" route (api_routes.py), this module's own tick, and
    the per-chat-turn bracket (scripts/chat_turn_affect.py) so there is
    exactly ONE call site for "hit capture_and_assess", not three that can
    drift.

    ``chat_correlation_id`` is sent only when supplied (chat_turn_pre/
    chat_turn_post callers) -- omitted entirely otherwise rather than sent
    as an explicit null, so the manual/ambient request bodies on the wire
    are byte-identical to what they were before this parameter existed.

    ``subtitle`` (2026-08-26) is the transcript Hub ALREADY has from the
    browser microphone. Passing it is what stops the affect capture needing an
    audio recording of its own -- see chat_turn_affect.fire()'s own comment for
    why that reverses an earlier deliberate choice.

    Always returns a dict shaped like the orchestrator's real response
    (``{"capture": ..., "result": {"ok": ..., "error": ...}, "event": ...}``)
    even on a malformed/non-dict reply -- review finding, 2026-08-22: this
    used to silently degrade to a bare ``{}`` on that path, dropping the
    error signal for both callers (the manual route's caller saw neither
    ``ok`` nor ``error``; the ambient loop's status line showed a generic
    "failed (unknown)" with no diagnostic value).
    """
    body: Dict[str, Any] = {"trigger": trigger}
    if chat_correlation_id:
        body["chat_correlation_id"] = chat_correlation_id
    # Sent only when there is real text, and stripped first -- a whitespace-only
    # value must not count as a transcript (the same hole that was found and
    # fixed in the retired backend's own subtitle resolution). Omitted entirely
    # otherwise, so the manual/ambient request bodies stay byte-identical to
    # what they were before this parameter existed.
    if subtitle and subtitle.strip():
        body["subtitle"] = subtitle.strip()
    resp = requests.post(
        f"{base_url.rstrip('/')}/v1/juniper/affect/capture_and_assess",
        json=body,
        timeout=timeout_sec,
    )
    resp.raise_for_status()
    parsed = resp.json()
    if isinstance(parsed, dict):
        return parsed
    return {"result": {"ok": False, "error": "invalid_response"}}


def result_ok_and_error(body: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Shared by run_ambient_tick below and the manual "Check now" route
    (api_routes.py) -- one place that decides "was this attempt actually
    ok" from call_capture_and_assess()'s response, not two that can drift."""
    result = body.get("result") if isinstance(body, dict) else None
    ok = bool(result.get("ok")) if isinstance(result, dict) else False
    error = None if ok else ((result or {}).get("error") if isinstance(result, dict) else "invalid_response")
    return ok, error


def result_content(body: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """(raw_response, video_sha256) for the "Carbon (affect snapshot)"
    dropdown option -- same call_capture_and_assess() response shape as
    result_ok_and_error above, just pulling different fields out of it."""
    result = body.get("result") if isinstance(body, dict) else None
    capture = body.get("capture") if isinstance(body, dict) else None
    raw_response = result.get("raw_response") if isinstance(result, dict) else None
    video_sha256 = capture.get("video_sha256") if isinstance(capture, dict) else None
    return raw_response, video_sha256


async def run_ambient_tick(base_url: str, timeout_sec: float) -> None:
    """One ambient attempt. Always off the event loop (asyncio.to_thread) --
    the blocking requests.post above can take up to ~195s worst case; a
    background asyncio.create_task loop calling it directly would freeze
    every other Hub request for that whole window.

    Skips (does not retry) if the manual "Check now" route currently holds
    the capture slot -- the loop's own next scheduled tick will try again;
    this cycle is just logged as a collision, same as any other failure.
    """
    if not try_begin_capture("ambient"):
        logger.info("[HUB] affect_ambient_tick_skipped reason=capture_in_progress")
        return
    try:
        body = await asyncio.to_thread(call_capture_and_assess, base_url, timeout_sec, "ambient")
        ok, error = result_ok_and_error(body)
        raw_response, video_sha256 = result_content(body)
        logger.info(f"[HUB] affect_ambient_tick ok={ok} tick_count={state.tick_count}")
        end_capture(ok=ok, error=error, raw_response=raw_response, video_sha256=video_sha256)
    except Exception as exc:  # advisory loop -- never crash Hub, never retry immediately
        logger.warning(f"[HUB] affect_ambient_tick_error error={exc}")
        end_capture(ok=False, error=str(exc))


async def affect_ambient_loop(
    *, base_url: str, interval_sec: float, timeout_sec: float, poll_sec: float
) -> None:
    """Always running once started (see scripts/main.py startup wiring) --
    gated by state.enabled, which the toggle route flips at any time.
    Toggling off takes effect within poll_sec (default 5s), not up to
    interval_sec (default 5min) later, because the flag is checked on the
    short poll cadence, not the long capture cadence.
    """
    while True:
        try:
            if state.enabled and not state.tick_in_progress:
                due = (
                    state.last_attempt_at is None
                    or (time.time() - state.last_attempt_at) >= interval_sec
                )
                if due:
                    await run_ambient_tick(base_url, timeout_sec)
        except Exception as exc:  # the loop itself must never die
            logger.error(f"[HUB] affect_ambient_loop_error error={exc}")
        await asyncio.sleep(poll_sec)
