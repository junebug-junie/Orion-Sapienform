"""Spend a finite allowance on the actions worth running.

THE PROBLEM THIS HAD TO SOLVE
-----------------------------
Every action Orion has measures approximately zero. Live, with a proper
control arm: prune -0.0073 +/- 0.0342, containers -0.0405 +/- 0.0581,
inspect +0.0386 +/- 0.0508 -- every one inside its own error bar.

So the obvious allocator is unbuildable. Rank by value-per-cost and you rank
noise, confidently, which is the exact failure this whole arc has been
deleting. Set an absolute bar on value and it refuses everything, which stops
Orion doing anything at all.

THE RESOLUTION: THERE ARE TWO TERMS, AND ONLY ONE WAS BEING USED
----------------------------------------------------------------
Expected free energy is pragmatic value (does this move the world toward a
preferred state) PLUS epistemic value (does this reduce uncertainty). Value-
per-cost ranking uses only the first. But an action whose effect is UNKNOWN is
worth running precisely because it is unknown, and an action run 7,583 times
with a stable outcome is worth nothing however large its effect once was.

For the Normal-Normal model these posteriors use, the expected information
gain from one more observation has an exact closed form:

    E[KL(posterior || prior)] = 0.5 * ln(1 + sigma^2 / tau^2)     [nats]

with sigma^2 the current posterior variance and tau^2 the observation
variance. Derived, then verified against 40,000-sample Monte Carlo at four
magnitudes of sigma^2 (agreement to 3 decimals -- see
tests/test_motor_allocator.py, which pins both).

Note what it does NOT depend on: the observation. Expected information is a
property of how uncertain you currently are, full stop. A cold cell
(sigma^2 = 0.25) is worth 0.99 nats; a well-measured one (0.001) is worth
0.012. Eighty times less, automatically, with no anti-repetition rule written
anywhere. Redundancy stops paying on its own -- which was the whole reason for
choosing Bayesian surprise as the currency back at the start of this arc.

NO INVENTED EXCHANGE RATE
-------------------------
The temptation is to add pragmatic and epistemic value into one number, which
needs a conversion between signal-units and nats. Hand-typing that conversion
would recreate `risk_score` -- five constants in a YAML file -- one layer up,
and this arc exists because that number made everything unrankable.

So they are kept in different roles:

  * epistemic value is the SCORE, in nats per motor-second. One unit,
    no conversion.
  * pragmatic value is a GATE. An action whose measured effect is
    CONFIDENTLY in the direction it claims to prevent is refused outright,
    however informative it would be.

An action that cannot lose is not competing; the gate is what lets one lose.

THE BAR IS ABSOLUTE, NOT RELATIVE
---------------------------------
`min_nats_per_sec` is a floor, not a rank cut. A relative ranking always
crowns a winner and can never say "none of these were worth doing" -- and
given the numbers above, that sentence is one the system has to be able to
say. Percentages sum to 100% no matter how worthless the set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from orion.autonomy.prediction import DEFAULT_OBSERVATION_VARIANCE

RefusalReason = Literal[
    "below_information_floor",
    "confidently_harmful",
    "allowance_exhausted",
    "no_cost_estimate",
]

# How many standard deviations of the measured contrast must sit on the wrong
# side of zero before an action is called confidently harmful. 2.0 is
# deliberately conservative: with every live contrast currently inside its own
# error bar, a looser gate would start refusing actions on noise, which is the
# failure this module exists to avoid. It should refuse almost nothing today
# and start biting as evidence accumulates.
HARM_CONFIDENCE_SIGMAS = 2.0


def expected_information_gain_nats(
    posterior_variance: float,
    observation_variance: float = DEFAULT_OBSERVATION_VARIANCE,
) -> float:
    """Expected nats from running this action once more.

    0.5 * ln(1 + sigma^2/tau^2). Exact for the Normal-Normal update in
    orion.autonomy.prediction, and independent of what the observation turns
    out to be.
    """
    if posterior_variance < 0 or not math.isfinite(posterior_variance):
        raise ValueError(f"posterior_variance must be finite and >= 0, got {posterior_variance!r}")
    if observation_variance <= 0 or not math.isfinite(observation_variance):
        raise ValueError(
            f"observation_variance must be finite and > 0, got {observation_variance!r}"
        )
    return 0.5 * math.log(1.0 + (posterior_variance / observation_variance))


@dataclass(frozen=True)
class Candidate:
    """One action asking for a slice of the allowance."""

    dispatch_id: str
    dispatch_kind: str
    target_id: str

    # Current uncertainty about what this action does. A cold cell is
    # maximally informative; a well-measured one is nearly worthless.
    posterior_variance: float

    # Real measured cost, from this action's own history. None means we have
    # never timed it -- see the `no_cost_estimate` refusal.
    cost_sec: float | None

    # Measured effect and its spread, for the harm gate only. None when there
    # is no control coverage, which is NOT evidence of safety and does not
    # gate anything.
    contrast: float | None = None
    contrast_sd: float | None = None
    # "increase" / "decrease" / "no_change", or None if undeclared.
    claimed_direction: str | None = None

    @property
    def expected_nats(self) -> float:
        return expected_information_gain_nats(self.posterior_variance)

    @property
    def nats_per_sec(self) -> float | None:
        if self.cost_sec is None or self.cost_sec <= 0:
            return None
        return self.expected_nats / self.cost_sec

    @property
    def confidently_harmful(self) -> bool:
        """Is the measured effect confidently opposite to what it claims?

        Requires a contrast, a spread, and a directional claim. Missing any of
        them means unknown, and unknown is not harmless -- it simply cannot be
        gated on. `no_change` claims cannot be harmful in this sense: they
        assert nothing about direction.
        """
        if self.contrast is None or self.contrast_sd is None:
            return False
        if self.claimed_direction not in ("increase", "decrease"):
            return False
        margin = HARM_CONFIDENCE_SIGMAS * max(self.contrast_sd, 0.0)
        if self.claimed_direction == "decrease":
            return (self.contrast - margin) > 0.0
        return (self.contrast + margin) < 0.0


@dataclass(frozen=True)
class Allocation:
    admitted: tuple[Candidate, ...]
    refused: tuple[tuple[Candidate, RefusalReason], ...]
    spent_sec: float
    allowance_sec: float

    @property
    def admitted_nats(self) -> float:
        return sum(c.expected_nats for c in self.admitted)

    def refusals_by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, reason in self.refused:
            counts[reason] = counts.get(reason, 0) + 1
        return counts


def allocate(
    candidates: list[Candidate],
    *,
    allowance_sec: float,
    min_nats_per_sec: float,
) -> Allocation:
    """Spend the allowance on the most informative actions that clear the bar.

    Order of operations matters and is deliberate:

    1. The harm gate first. A confidently harmful action is refused however
       informative it would be -- learning is not a reason to do damage.
    2. Then the absolute floor. Below it, the action is not worth its seconds
       even if the allowance is untouched. This is what makes "none of these
       were worth doing" expressible.
    3. Then greedy by nats-per-second until the allowance runs out.

    Greedy, not optimal. This is the fractional-knapsack ordering, which is
    exactly optimal when items are divisible and near-optimal here; a real
    0/1 knapsack solve would be false precision over cost estimates that carry
    their own error bars.
    """
    admitted: list[Candidate] = []
    refused: list[tuple[Candidate, RefusalReason]] = []

    scored: list[Candidate] = []
    for candidate in candidates:
        if candidate.confidently_harmful:
            refused.append((candidate, "confidently_harmful"))
            continue
        rate = candidate.nats_per_sec
        if rate is None:
            # Never timed. NOT admitted for free: an unmeasured cost is the
            # one that could be enormous, and admitting it would let the most
            # expensive actions bypass the budget entirely.
            refused.append((candidate, "no_cost_estimate"))
            continue
        if rate < min_nats_per_sec:
            refused.append((candidate, "below_information_floor"))
            continue
        scored.append(candidate)

    # Ties broken by dispatch_id so the same input always yields the same
    # allocation -- an allocator that reshuffles equal candidates makes its
    # own logs unreproducible.
    scored.sort(key=lambda c: (-(c.nats_per_sec or 0.0), c.dispatch_id))

    spent = 0.0
    for candidate in scored:
        cost = candidate.cost_sec or 0.0
        if spent + cost > allowance_sec:
            refused.append((candidate, "allowance_exhausted"))
            continue
        admitted.append(candidate)
        spent += cost

    return Allocation(
        admitted=tuple(admitted),
        refused=tuple(refused),
        spent_sec=spent,
        allowance_sec=allowance_sec,
    )
