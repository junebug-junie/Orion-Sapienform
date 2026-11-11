"""The GPU pool's background queue served by value (attention patch 3b). Pure; no I/O.

Spec: docs/superpowers/specs/2026-11-07-attention-patch3-market-decides-design.md, section 1
"Capacity by value", decided 2026-11-10 (defaults 2, 3 and 5).

When two of Orion's own discretionary jobs wait for the same kind of GPU seat, the pool serves the one
the body-economy market valued higher, instead of the one that arrived first. Only the ``background``
priority class is reordered: Juniper's turns (interactive), perception, system and urgent work keep
their places, and started work is never preempted (this only orders what is still waiting).

The rule (``pick``), over the waiting holds in arrival order:

* a hold is scored ``net + aging x hours waited`` (``net`` is the market decision's net the lease
  request carried);
* a later hold is served before an earlier one only when its score is at least ``margin`` higher;
* two holds of the same family keep arrival order: only each family's EARLIEST waiting hold competes;
* a hold with no bid (upkeep: compactors, orion-day, one-shot requests, any caller that sent no net)
  keeps its arrival place: nothing that arrived after it may jump it, and it jumps nobody.

``sequence`` applies ``pick`` repeatedly, so the scheduler's grant loop and the queue position a caller
sees follow the same order. ``services/orion-gpu-pool/evals/value_queue_replay.py`` (replay D) replays
the same ``pick`` against 14 days of real holds.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional, Sequence, TypeVar

VALUE_PRIORITY = "background"
DEFAULT_MARGIN = 0.10
DEFAULT_AGING_PER_H = 0.10

T = TypeVar("T")


@dataclass(frozen=True)
class ValueOrderConfig:
    enabled: bool = True
    margin: float = DEFAULT_MARGIN
    aging_per_h: float = DEFAULT_AGING_PER_H


def family_key(family: Optional[str]) -> Optional[str]:
    """Stages of one job are one family (``reading(stage2)`` -> ``reading``)."""
    return family.split("(")[0] if family else family


def score(net: float, arrived: datetime, now: datetime, aging_per_h: float) -> float:
    return float(net) + aging_per_h * max(0.0, (now - arrived).total_seconds()) / 3600.0


def pick(waiting: Sequence[T], now: datetime, *, net: Callable[[T], Optional[float]],
         family: Callable[[T], Optional[str]], arrived: Callable[[T], datetime],
         aging_per_h: float = DEFAULT_AGING_PER_H, margin: float = DEFAULT_MARGIN) -> T:
    """The waiting item the value-ordered queue serves at ``now``. ``waiting`` must be non-empty.
    Ties and gaps under ``margin`` keep arrival order (the input order breaks equal arrivals)."""
    order = sorted(waiting, key=arrived)            # stable: equal arrivals keep the caller's order
    heads: list[T] = []
    seen: set = set()
    for o in order:
        n = net(o)
        # Upkeep (no net) is its own family: two upkeep holds never merge into one head.
        key = ("upkeep", id(o)) if n is None else family_key(family(o))
        if key not in seen:
            seen.add(key)
            heads.append(o)
    best = heads[0]
    for o in heads[1:]:
        nb, no = net(best), net(o)
        if nb is None or no is None:
            break                                   # upkeep keeps its place: nothing after it jumps it
        if score(no, arrived(o), now, aging_per_h) >= score(nb, arrived(best), now, aging_per_h) + margin:
            best = o
    return best


def sequence(waiting: Sequence[T], now: datetime, *, net: Callable[[T], Optional[float]],
             family: Callable[[T], Optional[str]], arrived: Callable[[T], datetime],
             aging_per_h: float = DEFAULT_AGING_PER_H, margin: float = DEFAULT_MARGIN) -> list[T]:
    """Every waiting item in serve order (``pick`` applied until the queue is empty). O(n^2)."""
    rest = list(waiting)
    out: list[T] = []
    while rest:
        best = pick(rest, now, net=net, family=family, arrived=arrived, aging_per_h=aging_per_h, margin=margin)
        out.append(best)
        rest = [o for o in rest if o is not best]
    return out


__all__ = ["DEFAULT_AGING_PER_H", "DEFAULT_MARGIN", "VALUE_PRIORITY", "ValueOrderConfig", "family_key", "pick",
           "score", "sequence"]
