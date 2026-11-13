"""Personality rev 4, Move 2: Orion's tendencies set what Orion values in the attention market.

Spec: docs/superpowers/specs/2026-10-16-orion-dynamic-personality-graph-design.md, rev 4 "Move 2" and
"Metric quality gate (care and endorsement)". No shadow (Juniper, 2026-11-13): both terms BIND from
deploy. Kill switch: ``PHYSIOLOGY_PERSONALITY_VALUES=off`` (orion-durable-runs).

Two terms. Family endorsement, source (c), lives in ``orion.physiology.weights``. This module holds the
other term, **care inside a family**, plus the holdback arm and the runaway guard. PURE: no I/O.

**Care.** A want whose object has text (a curiosity question, a reading's query, a PR title, an
outreach topic) is compared with every active tendency's direction (``personality_disposition``
centroid: the mean of ``unit(embed(chosen) - mean(embed(declined)))`` over its choices)::

    care  = clamp(max_t cos(embed(text), direction_t) x strength(weight_t) x cap_t, 0, 1)
    value = intensity x outcome_value x (1 + BETA x care)          BETA = 0.5

``strength(w) = w / (w + WEIGHT_HALF)``: a tendency's weight is "weighted choices, recently"
(unbounded); it reaches half strength at 10 weighted choices, the spec's own line for a tendency that
is real enough to speak (Move 4). The care multiplier enters ``appetite.rank_value`` BEFORE the
percentile. No text, no tendency, or no positive cosine: care is exactly 0, which is exactly today's
value (the rest point is 0 and reachable).

**Holdback.** ``holdback_arm`` puts 20% of wants (deterministic per ``want_id``, the same sha256 draw
as Prime's ``draw_arm``) in the ``holdback`` arm: care 0 and the family weight WITHOUT source (c).
Those decisions bind too: it is a live control arm, not a shadow.

**Runaway guard.** Orion chooses from options the market supplied, and the market ranks them by
Orion's past choices. Three guards (spec):

* ``care_promoted``: a producer that shows Orion its top K options calls ``care_promoted_keys`` with
  each option's score with and without care (``net_without_care`` reads the latter off the market's
  decision) and sets ``PersonalityChoiceV1.care_promoted`` on a choice whose option only reached the
  shown set through care. The reducer weighs it half (``GraphParams.care_promoted_factor``), and so
  does source (c).
* the holdback arm keeps showing options the tendencies would not pick;
* ``runaway_caps``: a tendency whose share of SHOWN options (each shown option of a choice point is
  matched to the tendency it most cares about) is above 60% on each of the last 7 days is capped:
  its care is scaled by ``0.6 / share``. The caller logs every cap; it is never silent.

**Trace.** No schema change to ``WantV1`` or ``WantDecisionV1``: the arm, care and endorsement ride in
the decision's ``warnings`` (``trace_warnings``), which want_ledger stores. ``net_without_care`` parses
them back.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Mapping, Optional, Sequence

BETA = 0.5
HOLDBACK_FRACTION = 0.2
WEIGHT_HALF = 10.0
SHOWN_SHARE_CAP = 0.6
CAP_DAYS = 7
OBJECT_TEXT_MAX = 400

ARM_CARE = "care"
ARM_HOLDBACK = "holdback"

TRACE_PREFIX = "personality:"
# Care-zero reasons (``CareResult.reason``), each one recorded on the decision.
REASON_NO_TEXT = "no_object_text"
REASON_NO_TENDENCY = "no_tendency"
REASON_NO_MATCH = "no_match"
REASON_HOLDBACK = "holdback"
REASON_EMBED_FAILED = "embed_failed"
REASON_UNREAD = "tendencies_unread"


def holdback_arm(want_id: str, fraction: float = HOLDBACK_FRACTION) -> str:
    """``holdback`` for ``fraction`` of want ids, else ``care``. Deterministic: a re-bid of the same
    want draws the same arm (the hash Prime's ``draw_arm`` uses, with its own salt)."""
    u = int(hashlib.sha256(f"personality-holdback|{want_id}".encode("utf-8")).hexdigest()[:8], 16) / 0x100000000
    return ARM_HOLDBACK if u < max(0.0, min(1.0, float(fraction))) else ARM_CARE


def unit(vec: Sequence[float]) -> Optional[tuple[float, ...]]:
    """The unit vector, or None for an empty, non-finite or zero vector."""
    try:
        vals = [float(x) for x in vec]
    except (TypeError, ValueError):
        return None
    if not vals or not all(math.isfinite(x) for x in vals):
        return None
    n = math.sqrt(sum(x * x for x in vals))
    if n <= 0.0:
        return None
    return tuple(x / n for x in vals)


def strength(weight: float) -> float:
    """0..1: a tendency's weight as a strength, half at ``WEIGHT_HALF`` weighted choices."""
    if not math.isfinite(weight) or weight <= 0.0:
        return 0.0
    return weight / (weight + WEIGHT_HALF)


@dataclass(frozen=True)
class Tendency:
    """One active ``personality_disposition`` row: its unit direction, weight and label. ``cap`` is the
    runaway guard's factor (1.0 uncapped)."""
    disposition_id: str
    label: str
    direction: tuple[float, ...]
    weight: float
    cap: float = 1.0


@dataclass(frozen=True)
class CareResult:
    care: float                       # 0..1
    reason: str                       # "" when care > 0, else why it is 0
    tendency_id: Optional[str] = None
    cosine: Optional[float] = None


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def best_tendency(vec: Sequence[float], tendencies: Sequence[Tendency], *, with_cap: bool = True
                  ) -> tuple[Optional[Tendency], float, float]:
    """(tendency, raw cosine, scored) for the tendency with the highest ``cos x strength (x cap)``."""
    u = unit(vec)
    best: tuple[Optional[Tendency], float, float] = (None, 0.0, 0.0)
    if u is None:
        return best
    for t in tendencies:
        c = _cos(u, t.direction)
        s = c * strength(t.weight) * (t.cap if with_cap else 1.0)
        if best[0] is None or s > best[2]:
            best = (t, c, s)
    return best


def care_for(vec: Optional[Sequence[float]], tendencies: Sequence[Tendency]) -> CareResult:
    """Care for a want whose object text embeds to ``vec`` (module docstring). Never raises."""
    if vec is None:
        return CareResult(0.0, REASON_NO_TEXT)
    if not tendencies:
        return CareResult(0.0, REASON_NO_TENDENCY)
    t, c, s = best_tendency(vec, tendencies)
    if t is None or not math.isfinite(s) or s <= 0.0:
        return CareResult(0.0, REASON_NO_MATCH, t.disposition_id if t is not None else None,
                          round(c, 6) if t is not None else None)
    return CareResult(min(1.0, s), "", t.disposition_id, round(c, 6))


def care_multiplier(care: float) -> float:
    """``1 + BETA x care``, with care clamped to 0..1."""
    if not math.isfinite(care):
        return 1.0
    return 1.0 + BETA * min(1.0, max(0.0, care))


def object_text(text: Optional[str]) -> Optional[str]:
    """The want's object text, whitespace-collapsed and cut to ``OBJECT_TEXT_MAX``; None when empty."""
    if not isinstance(text, str):
        return None
    t = " ".join(text.split())
    return t[:OBJECT_TEXT_MAX] if t else None


# --- the runaway guard ---------------------------------------------------------------------------------

def daily_shown_shares(shown: Iterable[tuple[date, Optional[str]]]) -> dict[date, dict[str, float]]:
    """day -> tendency id -> share of that day's shown options matched to it. ``shown``: one entry per
    shown option, (UTC day, the tendency it matched or None)."""
    totals: dict[date, int] = {}
    hits: dict[date, dict[str, int]] = {}
    for d, tid in shown:
        totals[d] = totals.get(d, 0) + 1
        if tid:
            hits.setdefault(d, {})[tid] = hits.setdefault(d, {}).get(tid, 0) + 1
    return {d: {tid: n / totals[d] for tid, n in hits.get(d, {}).items()} for d in totals}


@dataclass(frozen=True)
class Cap:
    disposition_id: str
    mean_share: float     # over the CAP_DAYS days
    factor: float         # SHOWN_SHARE_CAP / mean_share


def runaway_caps(shares: Mapping[date, Mapping[str, float]], today: date) -> dict[str, Cap]:
    """Tendencies above ``SHOWN_SHARE_CAP`` of shown options on EACH of the ``CAP_DAYS`` days ending
    ``today``. A day with no shown options breaks the run (nothing was shown, so nothing ran away)."""
    days = [today - timedelta(days=i) for i in range(CAP_DAYS)]
    if any(d not in shares for d in days):
        return {}
    ids = set(shares[days[0]])
    out: dict[str, Cap] = {}
    for tid in ids:
        vals = [shares[d].get(tid, 0.0) for d in days]
        if all(v > SHOWN_SHARE_CAP for v in vals):
            mean = sum(vals) / len(vals)
            out[tid] = Cap(tid, round(mean, 6), round(SHOWN_SHARE_CAP / mean, 6))
    return out


def apply_caps(tendencies: Sequence[Tendency], caps: Mapping[str, Cap]) -> list[Tendency]:
    return [Tendency(t.disposition_id, t.label, t.direction, t.weight, caps[t.disposition_id].factor)
            if t.disposition_id in caps else t for t in tendencies]


def care_promoted_keys(scores_with_care: Mapping[str, float], scores_without_care: Mapping[str, float], k: int
                       ) -> set[str]:
    """For a producer that shows Orion its top ``k`` options: the option keys inside the top ``k`` by
    score WITH care that are outside it WITHOUT care. Set ``PersonalityChoiceV1.care_promoted`` on a
    choice whose chosen option is one of them. Ties break by key, so the result is deterministic."""
    if k <= 0:
        return set()

    def top(scores: Mapping[str, float]) -> set[str]:
        return {key for key, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]}

    return top(scores_with_care) - top(scores_without_care)


# --- the decision trace ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class PersonalityTerm:
    """What the personality values did to one bid. ``arm`` holdback: care 0 and source (c) left out."""
    arm: str
    care: float = 0.0
    reason: str = ""
    tendency_id: Optional[str] = None
    cosine: Optional[float] = None
    # Source (c) in this want's family weight: its rate and n, or None when the family has none.
    endorsement_rate: Optional[float] = None
    endorsement_n: int = 0
    # Why (c) is out of this family's weight ("holdback", "corr_with_engagement:0.86", ...), or "".
    endorsement_excluded: str = ""


def trace_warnings(term: PersonalityTerm, *, rank: float, rank_without_care: float, net: Optional[float],
                   net_without_care: Optional[float]) -> list[str]:
    """The decision-warning lines that record the term (want_ledger keeps them)."""
    out = [f"{TRACE_PREFIX}arm={term.arm}"]
    if term.arm == ARM_HOLDBACK:
        # The care the market priced: 0, whatever the term carried.
        out.append(f"{TRACE_PREFIX}care=0:reason={REASON_HOLDBACK}")
    elif term.care > 0.0:
        cos = f"{term.cosine:.4f}" if term.cosine is not None else "na"
        out.append(f"{TRACE_PREFIX}care={term.care:.4f}:tendency={term.tendency_id}:cos={cos}")
    else:
        out.append(f"{TRACE_PREFIX}care=0:reason={term.reason or REASON_NO_MATCH}")
    if term.endorsement_excluded:
        out.append(f"{TRACE_PREFIX}endorsement=excluded:{term.endorsement_excluded}")
    elif term.endorsement_rate is not None:
        out.append(f"{TRACE_PREFIX}endorsement={term.endorsement_rate:.4f}:n={term.endorsement_n}")
    else:
        out.append(f"{TRACE_PREFIX}endorsement=none")
    out.append(f"{TRACE_PREFIX}rank_without_care={rank_without_care:.6f}")
    if net_without_care is not None:
        out.append(f"{TRACE_PREFIX}net_without_care={net_without_care:.6f}")
    return out


def _trace_value(warnings: Sequence[str], key: str) -> Optional[str]:
    p = f"{TRACE_PREFIX}{key}="
    for w in warnings:
        if w.startswith(p):
            return w[len(p):]
    return None


def net_without_care(warnings: Sequence[str], net: Optional[float]) -> Optional[float]:
    """The decision's net without the care term (``care_promoted_keys``' "without" score). A decision
    with no personality trace (flag off, or the market could not price it) returns ``net`` itself."""
    v = _trace_value(warnings, "net_without_care")
    if v is None:
        return net
    try:
        return float(v)
    except ValueError:
        return net


def trace_arm(warnings: Sequence[str]) -> Optional[str]:
    return _trace_value(warnings, "arm")


def trace_care(warnings: Sequence[str]) -> Optional[float]:
    v = _trace_value(warnings, "care")
    if v is None:
        return None
    try:
        return float(v.split(":", 1)[0])
    except ValueError:
        return None


__all__ = [
    "ARM_CARE",
    "ARM_HOLDBACK",
    "BETA",
    "CAP_DAYS",
    "Cap",
    "CareResult",
    "HOLDBACK_FRACTION",
    "OBJECT_TEXT_MAX",
    "PersonalityTerm",
    "SHOWN_SHARE_CAP",
    "Tendency",
    "WEIGHT_HALF",
    "apply_caps",
    "best_tendency",
    "care_for",
    "care_multiplier",
    "care_promoted_keys",
    "daily_shown_shares",
    "holdback_arm",
    "net_without_care",
    "object_text",
    "runaway_caps",
    "strength",
    "trace_arm",
    "trace_care",
    "trace_warnings",
    "unit",
]
