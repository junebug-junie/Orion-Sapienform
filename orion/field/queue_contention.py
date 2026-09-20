"""Queue contention score: shared agent/curiosity capacity backlog vs EWMA.

Measures how backed up the reading-seed pipeline, durable GPU lease waits, and
LLM gateway admission waiting are relative to each source's own recent EWMA
baseline. Used to inform Orion's Cursor-vs-local hire decision.

NOT a rebadge of ``gpu_pressure`` (node biometrics / strain), 
``sustained_load_pressure`` (field-channel ``loaded_steady`` regime), or
``cortex_exec_step_load`` (execution step telemetry) — different producers,
different theory. Independence gate:
``docs/superpowers/specs/2026-09-20-queue-contention-metric-gate.md``.

Score formula (locked):

```text
ratio = count / max(ewma, floor)
sub   = clip(10 * (ratio - 1) / 4, 0, 10)   # 1x→0, 5x→10
score = max(subs)
driver = argmax(subs)  # None if all ~0
```

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
SOURCE_GATEWAY = "gateway_waiting"

SOURCE_KEYS: tuple[str, ...] = (SOURCE_SEED, SOURCE_DURABLE, SOURCE_GATEWAY)

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


def score_queue_contention(
    counts: Mapping[str, float],
    prev_ewma: Mapping[str, float],
    prev_n: Mapping[str, int],
    *,
    alpha: float,
    floor: float = DEFAULT_FLOOR,
) -> QueueContentionReading:
    """Score current queue depths against prior per-source EWMA baselines.

    Only keys present in ``counts`` participate (fail-open readers omit a key
    rather than inventing 0). Unknown keys outside ``SOURCE_KEYS`` are ignored.
    """
    raw: dict[str, float] = {}
    ewma: dict[str, float] = dict(prev_ewma)
    ewma_n: dict[str, int] = dict(prev_n)
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

    if not subs:
        return QueueContentionReading(
            score=0.0, driver=None, raw=raw, ewma=ewma, ewma_n=ewma_n
        )

    score = max(subs.values())
    driver: str | None = None
    if score > _DRIVER_EPS:
        # Stable tie-break: SOURCE_KEYS order (first max wins).
        driver = max(subs, key=lambda k: (subs[k], -SOURCE_KEYS.index(k)))

    return QueueContentionReading(
        score=score,
        driver=driver,
        raw=raw,
        ewma=ewma,
        ewma_n=ewma_n,
    )
