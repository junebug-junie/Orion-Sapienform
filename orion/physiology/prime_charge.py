"""Prime's regulate signals as temporary charges in the body-economy market (Orion Prime step 3).

Design: docs/superpowers/specs/2026-10-29-orion-prime-design.md section 4 (hormones are the existing
``regulate`` verb; receivers declare receptors; a signal nobody handles is a visible error) and
section 11 (refractory per receiver; Prime sees the outcome).

What a signal is. Prime (orion-mind) asks one organ to slow down or quicken. In the market that is a
TEMPORARY CHARGE (slow) or CREDIT (quicken) on that family's bids, added to the want's price charge in
``orion.physiology.choice.decide``:

    net = family_weight x value_rank - lane charges - prime_charge        (slow: prime_charge > 0)

It is bounded so it can never be a veto:

* ``prime_charge = magnitude x family_weight`` with ``magnitude <= MAX_MAGNITUDE`` (0.5). A want whose
  value rank is above ``magnitude`` (above the middle of its own family's history, at 0.5) still beats
  the charge, before lane prices. Hunger keeps growing while a want is deferred (appetite.hunger), so
  its rank rises and a held want outbids the charge by itself: constant deferral limits itself.
* the exploration floor still funds a family under 5% of its tier's funded wants;
* a binding run is held at most to its family's deferrable window (app.run_bids.hold_end), then asks
  the GPU pool anyway.

A quicken credit has the mirror bound: it is never a mandate. It is not applied to a want the market
values at zero (value rank 0: the family just acted, nothing in its history is less hungry), so it can
only lift a want that already has some value over a price. No organ declares ``quicken`` yet:
on curiosity (7 days to 2026-10-30) the only holds were "no hunger yet", which this rule leaves alone, and
lane prices were ~0, so a credit would change nothing there.

Where it lives. One Redis hash per receiving family, ``orion:prime:regulate:<family>`` on the bus
Redis (the same Redis the market already reads the posted prices from). Field ``signal`` is the
``PrimeRegulationV1`` JSON, written by Prime; fields ``w:<want_id>`` are written by the market, one per
want that bid while the signal was in force: what the charge did to it (``PrimeChargeEffectV1``). The
key expires ``GRACE_SEC`` after the signal, so Prime can read the final record after the signal ends,
and a dead Prime can never leave a charge in force (the market also checks ``expires_at`` itself).

Holdback. A ``control`` signal is written the same way; the market computes what the charge WOULD have
done on each bid and records it, but does not apply it. Both arms are graded on the same rule: the
market decides the bid both ways (the second decision on a throwaway copy of its reservation book) and
compares the two real decisions (``effect_between``), so the exploration floor and the capacity line count
the same in both arms. The treated-minus-control difference in curiosity's real starts is the effect.

Pure: no I/O. The market (services/orion-durable-runs/app/physiology_market.py) and Prime
(services/orion-mind/app/prime_loop.py) do the Redis reads and writes.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Iterable, Literal, Mapping, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from orion.physiology.families import RECEPTORS

KEY_PREFIX = "orion:prime:regulate:"
SIGNAL_FIELD = "signal"
WANT_FIELD_PREFIX = "w:"
# The charge never exceeds half the family's weight: a want ranked above the middle of its family's own
# history still beats it (before lane prices). Live 7 days to 2026-10-30, curiosity's bid values ran
# 0.14 .. 1.02 (median 0.77): a 0.5 charge defers the low-hunger bids right after a run closes and none
# of the hungry morning ones (services/orion-durable-runs/evals/prime_charge_replay.py).
MAX_MAGNITUDE = 0.5
DEFAULT_MAGNITUDE = 0.5
# How long the market's record outlives the signal: Prime reads the final record after the signal ends.
GRACE_SEC = 3600.0
# Prime's bounds on a signal's length. At most the family's deferrable window for curiosity (6 h): past
# it a held run asks the pool anyway, so a longer charge could not hold anything longer.
MIN_HOURS = 1.0
MAX_HOURS = 6.0
DEFAULT_HOURS = 3.0
MAX_WANT_FIELDS = 200

Direction = Literal["slow", "quicken"]
DIRECTIONS: tuple[str, ...] = RECEPTORS
Arm = Literal["treated", "control"]
# What the charge did to one want (strongest over its bids):
#   held      slow: without the charge it would have gone ahead (net > 0), with it it did not
#   released  quicken: without the credit it would have waited, with it it went ahead
#   outbid    slow: it went ahead anyway (value beat the charge)
#   unchanged the charge did not change the verdict (quicken on a want that went ahead anyway;
#             slow on one that would have waited anyway)
#   floor     the exploration floor funded it whatever the charge said
Effect = Literal["held", "released", "outbid", "unchanged", "floor"]
CHANGED = frozenset({"held", "released"})
_RANK = {"unchanged": 0, "floor": 1, "outbid": 2, "held": 3, "released": 3}


class PrimeRegulationV1(BaseModel):
    """One regulate signal in force (or held back) for one want family."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["prime.regulation.v1"] = "prime.regulation.v1"
    signal_id: str = Field(min_length=1)
    act_id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    direction: Direction
    arm: Arm
    magnitude: float = Field(gt=0.0, le=MAX_MAGNITUDE)
    why: str = Field(default="", max_length=600)
    opened_at: AwareDatetime
    expires_at: AwareDatetime


class PrimeChargeEffectV1(BaseModel):
    """What one want's bids met while the signal was in force (written by the market)."""

    model_config = ConfigDict(extra="forbid")

    want_id: str
    # The signal the effect belongs to: a bid that read the previous signal just before Prime replaced it
    # never counts for the new one.
    signal_id: Optional[str] = None
    effect: Effect               # the strongest effect over its bids (a run held once, then outbid, was held)
    last: Effect                 # the latest bid's effect
    bids: int = Field(ge=1)
    first_at: AwareDatetime
    last_at: AwareDatetime
    why: str = ""                # the latest decision's own why (names the charge)


def key(family: str) -> str:
    return f"{KEY_PREFIX}{family}"


def active(sig: PrimeRegulationV1, now: datetime) -> bool:
    return sig.opened_at <= now < sig.expires_at


def amount(sig: PrimeRegulationV1, weight: float, *, value_rank: Optional[float] = None) -> float:
    """The signed charge on a bid: + for slow, - for quicken. Bounded by MAX_MAGNITUDE x weight. A quicken
    credit is 0 for a want ranked 0 (never a mandate: it cannot fund a want the market values at nothing)."""
    w = float(weight) if isinstance(weight, (int, float)) and math.isfinite(weight) and weight > 0 else 0.0
    m = min(MAX_MAGNITUDE, max(0.0, float(sig.magnitude))) * w
    if sig.direction == "slow":
        return m
    return 0.0 if value_rank is not None and not value_rank > 0 else -m


def receives(receptors: Iterable[str], direction: str) -> bool:
    return direction in set(receptors or ())


def goes_ahead(decision: object) -> bool:
    """Does this decision let the run go now? Funded, or (Patch D, app.run_bids.bind_verdict) a capacity
    deferral worth its price, which queues at the GPU pool. Anything else holds a binding run."""
    if getattr(decision, "outcome", None) == "funded":
        return True
    net = getattr(decision, "net", None)
    return (getattr(decision, "reason", None) == "capacity_reserved" and isinstance(net, (int, float))
            and math.isfinite(net) and net > 0)


def effect_between(direction: str, with_charge: object, without_charge: object) -> Effect:
    """What the charge changed, from the market's two real decisions on the same bid (one with the charge,
    one without). Same rule for both arms; the exploration floor and capacity count as they really did."""
    w, wo = goes_ahead(with_charge), goes_ahead(without_charge)
    if direction == "slow":
        if w:
            return "floor" if getattr(with_charge, "reason", None) == "exploration_floor" else "outbid"
        return "held" if wo else "unchanged"
    return "released" if w and not wo else "unchanged"


def merge(prev: Optional[PrimeChargeEffectV1], want_id: str, eff: Effect, now: datetime,
          why: str, *, signal_id: Optional[str] = None) -> PrimeChargeEffectV1:
    if prev is not None and prev.signal_id != signal_id:
        prev = None          # left over from an earlier signal: never mixed into this one
    if prev is None:
        return PrimeChargeEffectV1(want_id=want_id, signal_id=signal_id, effect=eff, last=eff, bids=1,
                                   first_at=now, last_at=now, why=why[:300])
    strongest = eff if _RANK[eff] > _RANK[prev.effect] else prev.effect
    return prev.model_copy(update={"effect": strongest, "last": eff, "bids": prev.bids + 1, "last_at": now,
                                   "why": why[:300]})


def parse_signal(raw: object) -> Optional[PrimeRegulationV1]:
    if raw is None:
        return None
    try:
        return PrimeRegulationV1.model_validate_json(raw.decode() if isinstance(raw, bytes) else str(raw))
    except Exception:  # noqa: BLE001 -- an unreadable signal is no signal, never a charge
        return None


def parse_effects(fields: Mapping[object, object], *, signal_id: Optional[str] = None) -> list[PrimeChargeEffectV1]:
    """The market's per-want records; with ``signal_id``, only those written under that signal."""
    out = []
    for k, v in (fields or {}).items():
        name = k.decode() if isinstance(k, bytes) else str(k)
        if not name.startswith(WANT_FIELD_PREFIX):
            continue
        try:
            e = PrimeChargeEffectV1.model_validate_json(v.decode() if isinstance(v, bytes) else str(v))
        except Exception:  # noqa: BLE001
            continue
        if signal_id is None or e.signal_id == signal_id:
            out.append(e)
    return sorted(out, key=lambda e: e.first_at)


def tally(effects: Iterable[PrimeChargeEffectV1]) -> dict[str, int]:
    out = {"wants": 0, "held": 0, "released": 0, "outbid": 0, "unchanged": 0, "floor": 0}
    for e in effects:
        out["wants"] += 1
        out[e.effect] += 1
    return out


def changed_any(effects: Iterable[PrimeChargeEffectV1]) -> bool:
    return any(e.effect in CHANGED for e in effects)


def why_clause(sig: PrimeRegulationV1, *, value: float, lane_charge: float, prime: float, net: float) -> str:
    """The sentence a decision's ``why`` carries, so a refusal or a compliance is readable."""
    if sig.direction == "slow":
        verdict = "beat" if net > 0 else "did not beat"
        return (f"value {value:.2f} - lane charge {lane_charge:.2f} {verdict} Prime's slow charge {prime:.2f} "
                f"(until {sig.expires_at.strftime('%H:%MZ')}: {sig.why[:120]})")
    if prime == 0:
        return (f"Prime's quicken credit does not apply: the want has no value yet (until "
                f"{sig.expires_at.strftime('%H:%MZ')}: {sig.why[:120]})")
    return (f"Prime's quicken credit {abs(prime):.2f} added to value {value:.2f} - lane charge {lane_charge:.2f} "
            f"(until {sig.expires_at.strftime('%H:%MZ')}: {sig.why[:120]})")


def dumps(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), ensure_ascii=False)


__all__ = [
    "CHANGED",
    "DEFAULT_HOURS",
    "DEFAULT_MAGNITUDE",
    "DIRECTIONS",
    "GRACE_SEC",
    "KEY_PREFIX",
    "MAX_HOURS",
    "MAX_MAGNITUDE",
    "MAX_WANT_FIELDS",
    "MIN_HOURS",
    "PrimeChargeEffectV1",
    "PrimeRegulationV1",
    "SIGNAL_FIELD",
    "WANT_FIELD_PREFIX",
    "active",
    "amount",
    "changed_any",
    "dumps",
    "effect_between",
    "goes_ahead",
    "key",
    "merge",
    "parse_effects",
    "parse_signal",
    "receives",
    "tally",
    "why_clause",
]
