"""GPU/CUDA liveness watchdog for the whisper-tts service.

Real incident, 2026-08-26: this container's Coqui TTS backend crashed on
first real use with "CUDA is not available on this machine", while STT
(openai-whisper, same container, same GPU) kept working -- STT has an
explicit CPU fallback (``"cuda" if torch.cuda.is_available() else "cpu"``,
stt.py) and its already-running model had likely established a CUDA context
before the break; Coqui's TTS library has no such fallback and hard-asserts
CUDA. ``nvidia-smi`` inside the container returned "Failed to initialize
NVML: Unknown Error" -- a known docker+nvidia-container-toolkit staleness
quirk, usually triggered by something at the host level (another GPU
container being rebuilt/restarted, a driver/persistenced reload), not by
anything this service's own code does. A plain container restart fixed it
immediately (confirmed live): the container's device cgroup mapping gets
re-established fresh on boot.

This module cannot fix the underlying host/runtime quirk -- that is outside
this service's code, and outside Python's reach entirely. What it CAN do is
stop the failure being SILENT: this container already runs with
``restart: unless-stopped`` (docker-compose.yml), so the moment this
watchdog detects ``torch.cuda.is_available()`` going from True to False, a
clean process exit turns a silent multi-hour outage (found only when a
human notices TTS is broken) into a self-healing few-second blip -- the same
outcome the manual restart already validated in production, just automatic.

Deliberately does NOT try to detect "never was available at all" -- that is
`main.py`'s ``_require_cuda_or_die()``, called once at startup, which turns
THAT case into an immediate boot-time crash (also a restart, just triggered
by a different, narrower check at a different moment). This loop's job is
the complementary one: catch a LATER transition, mid-uptime, which a
one-shot startup check cannot see by definition.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
from typing import Callable

logger = logging.getLogger("orion-whisper-tts.cuda_watchdog")


def should_trigger_restart(consecutive_failures: int, failure_threshold: int) -> bool:
    """Pure decision function -- kept separate from the loop below so it is
    trivially testable with no asyncio, torch, or signal involved.

    A single failed check is not acted on immediately: an isolated NVML
    hiccup (transient driver contention, a poll landing mid-reinit) should
    not restart an otherwise-healthy service. Requiring ``failure_threshold``
    CONSECUTIVE failures before acting mirrors the debounce-before-acting
    shape this repo already uses elsewhere for an analogous
    not-ready-vs-actually-broken distinction
    (``curiosity_investigation.py``'s ``_consecutive_not_ready``).
    """
    return consecutive_failures >= failure_threshold


def restart_process() -> None:
    """The real ``on_trigger`` used in production: send SIGTERM to our own
    process.

    Deliberately NOT ``os._exit()``: SIGTERM lets uvicorn run its normal
    shutdown sequence (this service's own ``shutdown()`` handler cancels the
    other background tasks and closes the bus connection cleanly) before the
    process actually exits -- so ``restart: unless-stopped`` restarts a
    container that shut down cleanly, not one that was killed mid-write.
    """
    os.kill(os.getpid(), signal.SIGTERM)


async def cuda_watchdog_loop(
    *,
    poll_sec: float,
    failure_threshold: int,
    is_cuda_available: Callable[[], bool],
    on_trigger: Callable[[], None],
) -> None:
    """Runs for the life of the process, cancelled on shutdown -- same
    convention ``heartbeat_loop`` (main.py) already uses.

    ``is_cuda_available`` and ``on_trigger`` are injected rather than calling
    ``torch.cuda.is_available()`` / ``restart_process()`` directly, so tests
    never need a real CUDA runtime and never actually kill a process --
    production wires the real functions; tests wire fakes and assert on
    calls.
    """
    consecutive_failures = 0
    logger.info(
        "[WHISPER-TTS] cuda_watchdog_started poll_sec=%s failure_threshold=%s",
        poll_sec,
        failure_threshold,
    )
    try:
        while True:
            await asyncio.sleep(poll_sec)
            try:
                available = is_cuda_available()
            except Exception as exc:
                # The check itself failing is not the same signal as the
                # check succeeding and reporting False -- log it, but do not
                # let it count toward the failure threshold or crash this
                # loop. A broken CHECK is not proof of a broken GPU.
                logger.warning(
                    "[WHISPER-TTS] cuda_watchdog_check_error error=%s", exc
                )
                continue

            if available:
                if consecutive_failures:
                    logger.info(
                        "[WHISPER-TTS] cuda_watchdog_recovered "
                        "after %d failed check(s), no restart triggered",
                        consecutive_failures,
                    )
                consecutive_failures = 0
                continue

            consecutive_failures += 1
            logger.warning(
                "[WHISPER-TTS] cuda_watchdog_cuda_unavailable "
                "consecutive=%d threshold=%d",
                consecutive_failures,
                failure_threshold,
            )
            if should_trigger_restart(consecutive_failures, failure_threshold):
                logger.critical(
                    "[WHISPER-TTS] cuda_watchdog_triggering_restart "
                    "consecutive_failures=%d -- torch.cuda.is_available() has "
                    "been False for %d consecutive check(s). This container "
                    "runs with `restart: unless-stopped`; a container restart "
                    "is the known fix for this NVML staleness (confirmed live "
                    "2026-08-26). Exiting so the supervisor restarts us.",
                    consecutive_failures,
                    consecutive_failures,
                )
                on_trigger()
                return
    except asyncio.CancelledError:
        logger.info("[WHISPER-TTS] cuda_watchdog_stopping")
