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
# Operator work (evals, benches; config/gpu_pool.yaml classes ``operator*``, 2026-11-12 leased-evals
# design C2): it never bids, and inside the background class it waits behind every one of Orion's own
# background leases, whatever their arrival order (scheduler ``_order``, flag GPU_POOL_OPERATOR_LAST).
OPERATOR_CLASS_PREFIX = "operator"
DEFAULT_MARGIN = 0.10
DEFAULT_AGING_PER_H = 0.10

T = TypeVar("T")


@dataclass(frozen=True)
class ValueOrderConfig:
    enabled: bool = True
    margin: float = DEFAULT_MARGIN
    aging_per_h: float = DEFAULT_AGING_PER_H


def is_operator_class(work_class: Optional[str]) -> bool:
    """True for the pool's operator work classes (``operator``, ``operator_chat``, ...): work Juniper
    or an agent started (an eval, a bench), not one of Orion's own acts."""
    wc = str(work_class or "")
    return wc == OPERATOR_CLASS_PREFIX or wc.startswith(OPERATOR_CLASS_PREFIX + "_")


def family_key(family: Optional[str]) -> Optional[str]:
    """Stages of one job are one family (``reading(stage2)`` -> ``reading``)."""
    return family.split("(")[0] if family else family


def score(net: float, arrived: datetime, now: datetime, aging_per_h: float) -> float:
    return float(net) + aging_per_h * max(0.0, (now - arrived).total_seconds()) / 3600.0


def _heads(order: Sequence[T], net: Callable[[T], Optional[float]], family: Callable[[T], Optional[str]]) -> list[T]:
    heads: list[T] = []
    seen: set = set()
    for o in order:
        # Upkeep (no net) is its own family: two upkeep holds never merge into one head.
        key = ("upkeep", id(o)) if net(o) is None else family_key(family(o))
        if key not in seen:
            seen.add(key)
            heads.append(o)
    return heads


def _pick_sorted(order: Sequence[T], now: datetime, net, family, arrived, aging_per_h: float, margin: float) -> T:
    """``pick`` over items already in arrival order."""
    heads = _heads(order, net, family)
    best = heads[0]
    for o in heads[1:]:
        nb, no = net(best), net(o)
        if nb is None or no is None:
            break                                   # upkeep keeps its place: nothing after it jumps it
        if score(no, arrived(o), now, aging_per_h) >= score(nb, arrived(best), now, aging_per_h) + margin:
            best = o
    return best


def pick(waiting: Sequence[T], now: datetime, *, net: Callable[[T], Optional[float]],
         family: Callable[[T], Optional[str]], arrived: Callable[[T], datetime],
         aging_per_h: float = DEFAULT_AGING_PER_H, margin: float = DEFAULT_MARGIN) -> T:
    """The waiting item the value-ordered queue serves at ``now``. ``waiting`` must be non-empty and
    hold items that compete for the same seats (the scheduler passes one work class at a time).
    Ties and gaps under ``margin`` keep arrival order (the input order breaks equal arrivals)."""
    return _pick_sorted(sorted(waiting, key=arrived), now, net, family, arrived, aging_per_h, margin)


def sequence(waiting: Sequence[T], now: datetime, *, net: Callable[[T], Optional[float]],
             family: Callable[[T], Optional[str]], arrived: Callable[[T], datetime],
             aging_per_h: float = DEFAULT_AGING_PER_H, margin: float = DEFAULT_MARGIN) -> list[T]:
    """Every waiting item in serve order (``pick`` applied until the queue is empty). Sorted once;
    O(n^2) after that."""
    rest = sorted(waiting, key=arrived)
    out: list[T] = []
    while rest:
        best = _pick_sorted(rest, now, net, family, arrived, aging_per_h, margin)
        out.append(best)
        rest = [o for o in rest if o is not best]
    return out


__all__ = ["DEFAULT_AGING_PER_H", "DEFAULT_MARGIN", "OPERATOR_CLASS_PREFIX", "VALUE_PRIORITY", "ValueOrderConfig",
           "family_key", "is_operator_class", "pick", "score", "sequence"]
