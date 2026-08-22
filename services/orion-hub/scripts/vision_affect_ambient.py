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
webcam+mic recording trigger.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from loguru import logger


@dataclass
class AffectAmbientState:
    enabled: bool = False
    tick_in_progress: bool = False
    tick_count: int = 0
    last_attempt_at: Optional[float] = None
    last_result_ok: Optional[bool] = None
    last_error: Optional[str] = None

    def status_payload(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "tick_in_progress": self.tick_in_progress,
            "tick_count": self.tick_count,
            "last_attempt_at": self.last_attempt_at,
            "last_result_ok": self.last_result_ok,
            "last_error": self.last_error,
        }


# Module-level singleton, same convention as biometrics_cache.py's own
# process-wide cache object -- one Hub process, one ambient state, no need
# for a class instance threaded through every call site.
state = AffectAmbientState()


def call_capture_and_assess(base_url: str, timeout_sec: float, trigger: str) -> Dict[str, Any]:
    """Blocking HTTP call to the orchestrator -- callers off the event loop
    (the ambient loop below) run this via asyncio.to_thread. Shared by the
    manual "Check now" route (api_routes.py) and this module's own tick so
    there is exactly ONE call site for "hit capture_and_assess", not two
    that can drift.
    """
    resp = requests.post(
        f"{base_url.rstrip('/')}/v1/juniper/affect/capture_and_assess",
        json={"trigger": trigger},
        timeout=timeout_sec,
    )
    resp.raise_for_status()
    parsed = resp.json()
    return parsed if isinstance(parsed, dict) else {}


async def run_ambient_tick(base_url: str, timeout_sec: float) -> None:
    """One ambient attempt. Always off the event loop (asyncio.to_thread) --
    the blocking requests.post above can take up to ~195s worst case; a
    background asyncio.create_task loop calling it directly would freeze
    every other Hub request for that whole window.
    """
    state.tick_in_progress = True
    state.last_attempt_at = time.time()
    state.tick_count += 1
    try:
        body = await asyncio.to_thread(call_capture_and_assess, base_url, timeout_sec, "ambient")
        result = body.get("result") if isinstance(body, dict) else None
        ok = bool(result.get("ok")) if isinstance(result, dict) else False
        state.last_result_ok = ok
        state.last_error = None if ok else (result or {}).get("error")
        logger.info(f"[HUB] affect_ambient_tick ok={ok} tick_count={state.tick_count}")
    except Exception as exc:  # advisory loop -- never crash Hub, never retry immediately
        state.last_result_ok = False
        state.last_error = str(exc)
        logger.warning(f"[HUB] affect_ambient_tick_error error={exc}")
    finally:
        state.tick_in_progress = False


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
