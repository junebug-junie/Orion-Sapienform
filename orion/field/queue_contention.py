"""Queue contention score: shared agent/curiosity capacity backlog.

Measures how backed up the reading-seed pipeline, durable GPU demands, and GPU
pool waiting leases are, two ways per source:

* **depth** -- how many items wait, relative to that source's own recent EWMA
  baseline;
* **oldest wait** (2026-09-25) -- how long the oldest waiting item has waited,
  relative to a fixed per-source expected wait.

Used to inform Orion's Cursor-vs-local hire decision.

Why the second half exists: a depth-vs-own-baseline score cannot see a FROZEN
queue. Live 2026-09-25, the seed queue held 144 pending items with the oldest
from 2026-09-07 and nothing done since 09-14, yet the score sat at 0.14 and
was decaying toward 0 because the EWMA baseline was converging on the frozen
count. A stuck queue and a quiet queue read the same to a relative depth
score; only the age of what is waiting tells them apart. The expected waits
are fixed config, not an EWMA, on purpose: a baseline that adapts would learn
"stuck" as normal the same way the depth baseline did.

NOT a rebadge of ``gpu_pressure`` (node biometrics / strain), 
``sustained_load_pressure`` (field-channel ``loaded_steady`` regime), or
``cortex_exec_step_load`` (execution step telemetry) — different producers,
different theory. Independence gate:
``docs/superpowers/specs/2026-09-20-queue-contention-metric-gate.md``.

Score formula (locked):

```text
ratio     = count / max(ewma, floor)
depth_sub = clip(10 * (ratio - 1) / 4, 0, 10)          # 1x→0, 5x→10
age_ratio = oldest_wait_sec / expected_wait_sec
age_sub   = clip(10 * (age_ratio - 1) / 4, 0, 10)      # 1x→0, 5x→10
score  = max(all depth_subs and age_subs)
driver = "<source>" if a depth sub wins,
         "<source>:oldest_wait" if an age sub wins,
         None if every sub is ~0
```

Rest point: an empty queue has no oldest item (age 0) and a queue whose oldest
item is younger than its expected wait gives ``age_ratio <= 1`` -> ``age_sub``
is exactly 0.0 (clip), not a floor. The age is a fresh read every tick, never
carried forward or decayed, so it cannot sink to a fake 0 either.

EWMA mean uses ``orion.bus.ewma.compute_ewma_update`` (variance unused here;
prev_variance always 0). Score is against the *prior* baseline before this
tick's count is absorbed — same z-score-before-absorb convention as that
helper. First observation (prev_n==0) establishes the baseline with sub=0
rather than fabricating a spike against an empty history.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from orion.bus.ewma import compute_ewma_update

SOURCE_SEED = "world_pulse_seed_pending"
SOURCE_DURABLE = "durable_demand_pending"
# Leases queued or backlogged in orion-gpu-pool (gpu_pool_leases). Replaces "gateway_waiting" (the
# LLM gateway's in-process admission ledger, deleted when the gateway cut over to the pool,
# 2026-09-24). Same meaning -- work waiting for a GPU -- now counted where the waiting happens.
# Gate: docs/superpowers/specs/2026-09-20-queue-contention-metric-gate.md, "Re-point" section.
SOURCE_GPU_POOL = "gpu_pool_waiting"

SOURCE_KEYS: tuple[str, ...] = (SOURCE_SEED, SOURCE_DURABLE, SOURCE_GPU_POOL)

# Driver suffix when a source's oldest-wait sub (not its depth sub) wins the max().
OLDEST_WAIT_SUFFIX = ":oldest_wait"

# Per-source expected wait (seconds) for the oldest waiting item. Knobs, not findings: each is
# anchored to live data in docs/superpowers/specs/2026-09-20-queue-contention-metric-gate.md,
# "Oldest-wait component (2026-09-25)". age_sub is 0 up to 1x and 10 at 5x.
# Seeds: 48h ~= p90 claim wait of every seed ever completed (175,596s, n=15); 10/10 at 10 days.
DEFAULT_SEED_EXPECTED_WAIT_SEC = 172800.0
# Durable demands: 12h ~= p90 time-to-first-grant over 138 granted demands (45,086s); 10/10 at 60h.
DEFAULT_DURABLE_EXPECTED_WAIT_SEC = 43200.0
# GPU pool: 60s = 10x the worst grant wait seen over 1,275 leases (5.7s); 10/10 at 300s, which is
# the deadline most callers put on a lease -- the oldest waiter is about to give up.
DEFAULT_GPU_POOL_EXPECTED_WAIT_SEC = 60.0

DEFAULT_EXPECTED_WAIT_SEC: dict[str, float] = {
    SOURCE_SEED: DEFAULT_SEED_EXPECTED_WAIT_SEC,
    SOURCE_DURABLE: DEFAULT_DURABLE_EXPECTED_WAIT_SEC,
    SOURCE_GPU_POOL: DEFAULT_GPU_POOL_EXPECTED_WAIT_SEC,
}

DEFAULT_FLOOR = 1.0
# Digester tick ~2s (RECEIPT_POLL_INTERVAL_SEC); half-life ~24h per gate doc.
DEFAULT_HALF_LIFE_SEC = 86400.0

_DRIVER_EPS = 1e-12


@dataclass(frozen=True)
class QueueContentionReading:
    """One tick's queue-contention score against prior EWMA baselines."""

    score: float
    """Overall 0–10 pressure; ``max()`` of per-source subs, not an average."""

    driver: str | None
    """Source key that produced ``score``, or None when every sub is ~0."""

    raw: dict[str, float]
    """Counts actually scored this tick (omit unreachable fail-open sources)."""

    oldest_wait_sec: dict[str, float]
    """Oldest-item wait actually scored this tick (omit unreachable fail-open sources)."""

    subs: dict[str, float]
    """Every sub-score that competed for the max, keyed like ``driver``."""

    ewma: dict[str, float]
    """Updated per-source EWMA means after absorbing ``raw``."""

    ewma_n: dict[str, int]
    """Updated per-source observation counts after this tick."""


def ewma_alpha(*, dt_sec: float, half_life_sec: float = DEFAULT_HALF_LIFE_SEC) -> float:
    """EWMA smoothing from tick interval and half-life.

    ``alpha = 1 - exp(-ln(2) * dt / half_life)`` — standard half-life form so
    a ~24h half-life at a ~2s digester tick is a single documented constant
    rather than a magic 1e-5.
    """
    if half_life_sec <= 0.0 or dt_sec <= 0.0:
        raise ValueError("dt_sec and half_life_sec must be positive")
    return 1.0 - math.exp(-math.log(2.0) * dt_sec / half_life_sec)


def _clip(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def _sub_score(count: float, baseline: float, *, floor: float) -> float:
    ratio = count / max(baseline, floor)
    return _clip(10.0 * (ratio - 1.0) / 4.0, 0.0, 10.0)


def _age_sub_score(oldest_wait_sec: float, expected_wait_sec: float) -> float:
    ratio = max(oldest_wait_sec, 0.0) / expected_wait_sec
    return _clip(10.0 * (ratio - 1.0) / 4.0, 0.0, 10.0)


def score_queue_contention(
    counts: Mapping[str, float],
    prev_ewma: Mapping[str, float],
    prev_n: Mapping[str, int],
    *,
    alpha: float,
    floor: float = DEFAULT_FLOOR,
    oldest_wait_sec: Mapping[str, float] | None = None,
    expected_wait_sec: Mapping[str, float] | None = None,
) -> QueueContentionReading:
    """Score current queue depths (vs prior EWMA) and oldest waits (vs expected wait).

    Only keys present in ``counts`` / ``oldest_wait_sec`` participate (fail-open
    readers omit a key rather than inventing 0). Unknown keys outside
    ``SOURCE_KEYS`` are ignored. ``expected_wait_sec`` falls back per key to
    ``DEFAULT_EXPECTED_WAIT_SEC``.
    """
    ages_in = oldest_wait_sec or {}
    expected = {**DEFAULT_EXPECTED_WAIT_SEC, **(expected_wait_sec or {})}
    ages: dict[str, float] = {}
    raw: dict[str, float] = {}
    # Only live sources carry forward: a retired source's baseline (e.g. "gateway_waiting") is dropped,
    # not left riding along in FieldStateV1 forever where a generic reader could mistake it for signal.
    ewma: dict[str, float] = {k: v for k, v in prev_ewma.items() if k in SOURCE_KEYS}
    ewma_n: dict[str, int] = {k: v for k, v in prev_n.items() if k in SOURCE_KEYS}
    subs: dict[str, float] = {}

    for key in SOURCE_KEYS:
        if key not in counts:
            continue
        count = float(counts[key])
        raw[key] = count
        n_prev = int(prev_n.get(key, 0))
        ewma_prev = float(prev_ewma.get(key, 0.0))

        if n_prev == 0:
            sub = 0.0
        else:
            sub = _sub_score(count, ewma_prev, floor=floor)
        subs[key] = sub

        update = compute_ewma_update(
            prev_ewma=ewma_prev,
            prev_variance=0.0,
            prev_count=n_prev,
            value=count,
            alpha=alpha,
        )
        ewma[key] = update.ewma
        ewma_n[key] = n_prev + 1

    # Oldest-wait subs. No warm-up: the expected wait is fixed config, not a learned baseline,
    # so the very first tick after a restart can already say "this queue is stuck".
    for key in SOURCE_KEYS:
        if key not in ages_in:
            continue
        exp_wait = float(expected[key])
        if not exp_wait > 0.0:
            # Bad override: skip this source's age sub rather than crash the digester tick.
            continue
        age = max(float(ages_in[key]), 0.0)
        ages[key] = age
        subs[key + OLDEST_WAIT_SUFFIX] = _age_sub_score(age, exp_wait)

    if not subs:
        return QueueContentionReading(
            score=0.0,
            driver=None,
            raw=raw,
            oldest_wait_sec=ages,
            subs=subs,
            ewma=ewma,
            ewma_n=ewma_n,
        )

    score = max(subs.values())
    driver: str | None = None
    if score > _DRIVER_EPS:
        # Stable tie-break: SOURCE_KEYS order, depth before oldest-wait (first max wins).
        order = [k for src in SOURCE_KEYS for k in (src, src + OLDEST_WAIT_SUFFIX)]
        driver = max(subs, key=lambda k: (subs[k], -order.index(k)))

    return QueueContentionReading(
        score=score,
        driver=driver,
        raw=raw,
        oldest_wait_sec=ages,
        subs=subs,
        ewma=ewma,
        ewma_n=ewma_n,
    )


def driver_source(driver: str | None) -> tuple[str | None, bool]:
    """Split a driver into ``(source_key, is_oldest_wait)``; ``(None, False)`` for no driver."""
    if not driver:
        return None, False
    if driver.endswith(OLDEST_WAIT_SUFFIX):
        return driver[: -len(OLDEST_WAIT_SUFFIX)], True
    return driver, False
