"""Settle a declared power intent against a fast GPU sample window.

A workload publishes PowerIntentV1 before it draws; this opens a ~1 Hz window on the
named GPU, closes it at the declared duration or the deadline, and publishes what the
meter actually saw.

WHY NOT REUSE `collect_gpu_stats()`. The standing path shells out to
`/orion/sensors/gpu_host_stats.sh`, `time.sleep(1)`s, then scans a directory for the
newest CSV. That is fine at a 31s cadence and unusable at 1 Hz: the sleep alone eats
the interval, and the whole point of this module is resolution the standing sampler does
not have. Measured 2026-08-28 on circe -- 332 diffusion jobs in three days, 4 caught by
the 31s sampler. So this queries nvidia-smi directly for one card and parses one number.

THE SAMPLER IS INJECTED. Tests drive real settlement arithmetic without a GPU, and the
production sampler stays a thin, separately-reviewable function.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Awaitable, Callable, List, Optional

from orion.schemas.power import PowerIntentSettledV1, PowerIntentV1

logger = logging.getLogger("orion.biometrics.power_intent")

# Floor on the window so a zero/negative declared duration cannot spin.
_MIN_WINDOW_SEC = 0.5
# Ceiling regardless of what was declared -- a deadline far in the future must not pin a
# 1 Hz sampler for hours. The deadline still applies; this is the backstop under it.
_MAX_WINDOW_SEC = 300.0


def sample_gpu_watts(gpu_index: int, *, timeout_sec: float = 2.0) -> Optional[float]:
    """One instantaneous power reading for one card, or None.

    None means "could not read", never 0.0. A card that genuinely draws nothing does not
    exist, so a zero here would be a fabricated measurement -- and the settlement's
    `outcome` field exists precisely so unread windows stay distinguishable from quiet
    ones.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [
                exe,
                f"--id={int(gpu_index)}",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 -- any failure is "could not read"
        return None
    try:
        watts = float(out.splitlines()[0].strip())
    except (ValueError, IndexError):
        return None
    return watts if watts >= 0.0 else None


def _window_seconds(intent: PowerIntentV1, now: datetime) -> float:
    """How long to sample: the declared duration, clamped, and never past the deadline."""
    declared = max(float(intent.expected_duration_sec or 0.0), _MIN_WINDOW_SEC)
    declared = min(declared, _MAX_WINDOW_SEC)
    remaining = (intent.deadline - now).total_seconds()
    return max(0.0, min(declared, remaining))


def summarize(
    intent: PowerIntentV1,
    samples: List[float],
    *,
    baseline: Optional[float],
    window_start: datetime,
    window_end: datetime,
    hit_deadline: bool,
) -> PowerIntentSettledV1:
    """Pure settlement arithmetic. No I/O, no clock."""
    elapsed = max(0.0, (window_end - window_start).total_seconds())

    if not samples:
        # NOT 0.0 W. "We did not see" and "we saw nothing drawn" are opposite claims.
        return PowerIntentSettledV1(
            intent_id=intent.intent_id,
            workload_kind=intent.workload_kind,
            node=intent.node,
            gpu_index=intent.gpu_index,
            outcome="deadline_expired" if hit_deadline else "no_samples",
            window_start=window_start,
            window_end=window_end,
            sample_count=0,
            achieved_sample_hz=None,
            baseline_watts=baseline,
            expected_watts=intent.expected_watts,
            residual_watts=None,
        )

    peak = max(samples)
    mean = sum(samples) / len(samples)
    residual = None
    if intent.expected_watts is not None:
        residual = peak - float(intent.expected_watts)

    return PowerIntentSettledV1(
        intent_id=intent.intent_id,
        workload_kind=intent.workload_kind,
        node=intent.node,
        gpu_index=intent.gpu_index,
        outcome="settled",
        window_start=window_start,
        window_end=window_end,
        sample_count=len(samples),
        # The rate ACHIEVED, not the rate configured. A window that only managed two
        # samples of an eight-second burst is arithmetic, not measurement, and this is
        # what lets a reader tell which one they are holding.
        achieved_sample_hz=(len(samples) / elapsed) if elapsed > 0 else None,
        actual_peak_watts=peak,
        actual_mean_watts=mean,
        energy_joules=mean * elapsed if elapsed > 0 else None,
        baseline_watts=baseline,
        expected_watts=intent.expected_watts,
        residual_watts=residual,
    )


async def settle(
    intent: PowerIntentV1,
    *,
    sampler: Callable[[int], Optional[float]],
    sample_interval_sec: float = 1.0,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> PowerIntentSettledV1:
    """Open the window, sample, close it, and return the settlement."""
    if intent.gpu_index is None:
        # Node-scoped intents settle against wall power, which this module does not
        # own. Report honestly rather than inventing a GPU window.
        start = now_fn()
        return summarize(
            intent, [], baseline=None, window_start=start, window_end=start,
            hit_deadline=False,
        )

    gpu = int(intent.gpu_index)
    # asyncio.to_thread: the default sampler is a synchronous subprocess.run of
    # nvidia-smi (up to a 2.0s timeout). Called bare on the biometrics event loop
    # it blocks the SystemHealth heartbeat, the iLO/PDU pollers, and the hub's
    # intake Hunter -- roughly 20 times per generation. The first live settlement
    # measured achieved_sample_hz=0.894 against a configured 1.0s interval, i.e.
    # ~0.11s of loop-blocking per sample. to_thread accepts a plain sync callable,
    # so injected test samplers keep working unchanged.
    baseline = await asyncio.to_thread(sampler, gpu)  # before the window opens
    start = now_fn()
    window = _window_seconds(intent, start)

    samples: List[float] = []
    hit_deadline = False
    while True:
        now = now_fn()
        if now >= intent.deadline:
            hit_deadline = True
            break
        if (now - start).total_seconds() >= window:
            break
        value = await asyncio.to_thread(sampler, gpu)
        if value is not None:
            samples.append(value)
        await sleep_fn(sample_interval_sec)

    end = now_fn()
    settled = summarize(
        intent, samples, baseline=baseline, window_start=start, window_end=end,
        hit_deadline=hit_deadline,
    )
    logger.info(
        "power_intent_settled intent=%s workload=%s gpu=%s outcome=%s samples=%d "
        "peak=%s baseline=%s residual=%s",
        settled.intent_id, settled.workload_kind, settled.gpu_index, settled.outcome,
        settled.sample_count, settled.actual_peak_watts, settled.baseline_watts,
        settled.residual_watts,
    )
    return settled
