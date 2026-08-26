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

Deliberately does NOT try to detect "never was available at all" as a
separate crash path -- that used to be `main.py`'s ``_require_cuda_or_die()``
raising at startup, until a review (2026-08-26) caught that a hard crash
there took down the ENTIRE process, including STT, which does not need CUDA
at all. STT surviving a CUDA outage that killed TTS is exactly the
resilience this whole feature exists to preserve; a boot-time crash
undid it. ``_require_cuda_or_die()`` is now advisory-only (logs CRITICAL,
never raises) -- THIS loop is the single enforcement mechanism for both
"broken at boot" and "broken mid-uptime": a boot-broken GPU simply fails
its first check almost immediately and restarts on the normal threshold,
same as a break mid-uptime. One mechanism, not two that can drift apart.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
from typing import Awaitable, Callable, Union

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


async def _check_with_timeout(
    is_cuda_available: Callable[[], bool], timeout_sec: float
) -> bool:
    """Runs ``is_cuda_available`` off the event loop with a bound.

    Review finding, 2026-08-26: a plain synchronous call on the loop has two
    failure modes a naive `except Exception` around it does not cover. A
    genuine NVML/driver wedge -- exactly the state this watchdog exists to
    detect -- is a documented way for an NVML call to HANG rather than
    fast-return. Called in-line on the single shared event loop, that hang
    freezes `heartbeat_loop`, `listener_task`, and `stt_task` right along
    with this one, and the failure-counting logic below never even runs --
    the watchdog cannot act during its own target scenario. Running it via
    `asyncio.to_thread` keeps the loop free; `asyncio.wait_for` turns an
    actual hang into a real, counted failure instead of an unbounded stall.
    """
    return await asyncio.wait_for(
        asyncio.to_thread(is_cuda_available), timeout=timeout_sec
    )


async def cuda_watchdog_loop(
    *,
    poll_sec: float,
    failure_threshold: int,
    is_cuda_available: Callable[[], bool],
    on_trigger: Union[Callable[[], None], Callable[[], Awaitable[None]]],
    check_timeout_sec: float | None = None,
) -> None:
    """Runs for the life of the process, cancelled on shutdown -- same
    convention ``heartbeat_loop`` (main.py) already uses.

    ``is_cuda_available`` and ``on_trigger`` are injected rather than calling
    ``torch.cuda.is_available()`` / ``restart_process()`` directly, so tests
    never need a real CUDA runtime and never actually kill a process --
    production wires the real functions; tests wire fakes and assert on
    calls. ``on_trigger`` may be sync or async; both are awaited uniformly.

    ``check_timeout_sec`` defaults to ``poll_sec`` -- a check is not allowed
    to run longer than the gap between checks, or the next tick's `sleep`
    would start immediately into an already-overdue check.
    """
    consecutive_failures = 0
    timeout_sec = check_timeout_sec if check_timeout_sec is not None else poll_sec
    logger.info(
        "[WHISPER-TTS] cuda_watchdog_started poll_sec=%s failure_threshold=%s "
        "check_timeout_sec=%s",
        poll_sec,
        failure_threshold,
        timeout_sec,
    )
    try:
        while True:
            await asyncio.sleep(poll_sec)
            # Review finding, 2026-08-26: the ENTIRE tick body used to sit
            # outside any guard beyond the check-specific one below. Any
            # other unexpected exception in this body (including on_trigger
            # itself, e.g. os.kill raising under a restricted sandbox
            # runtime) killed this fire-and-forget task silently -- no
            # supervision, no self-restart of the watchdog, and the very
            # failure class this feature exists to close would then recur
            # with nothing left watching for it.
            try:
                try:
                    available = await _check_with_timeout(
                        is_cuda_available, timeout_sec
                    )
                except asyncio.TimeoutError:
                    # A hang IS evidence of the wedged state this watchdog
                    # targets -- counted as a real failure, not skipped the
                    # way a raised exception is (see the comment below).
                    logger.warning(
                        "[WHISPER-TTS] cuda_watchdog_check_timeout "
                        "after %ss -- treating as a failed check",
                        timeout_sec,
                    )
                    available = False
                except Exception as exc:
                    # The check RAISING is a different failure mode from the
                    # check succeeding and reporting False -- log it, but do
                    # not let it count toward the threshold or crash this
                    # loop. A broken CHECK is not proof of a broken GPU (a
                    # hang, handled above, is treated differently on
                    # purpose).
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
                        "consecutive_failures=%d -- torch.cuda.is_available() "
                        "has been False (or hung) for %d consecutive "
                        "check(s). This container runs with `restart: "
                        "unless-stopped`; a container restart is the known "
                        "fix for this NVML staleness (confirmed live "
                        "2026-08-26). Exiting so the supervisor restarts us.",
                        consecutive_failures,
                        consecutive_failures,
                    )
                    result = on_trigger()
                    if asyncio.iscoroutine(result):
                        await result
                    return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Last-resort net around the WHOLE tick, not just the check.
                # Logged and the loop continues rather than the task dying
                # unsupervised -- an outage this feature exists to catch
                # must never be made worse by the catcher itself falling
                # over.
                logger.error(
                    "[WHISPER-TTS] cuda_watchdog_unexpected_error error=%s",
                    exc,
                    exc_info=True,
                )
    except asyncio.CancelledError:
        logger.info("[WHISPER-TTS] cuda_watchdog_stopping")
