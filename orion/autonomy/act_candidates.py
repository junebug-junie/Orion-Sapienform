"""Rung-2 acts as motor-allocator candidates (thinking architecture rung 3, "compete across doers").

Design: docs/superpowers/specs/2026-10-14-orion-thinking-architecture-design.md, rung 3. Pure: no I/O.

A dispatch route that really produces a rung-2 act declares it (``act:`` on the route in
config/execution_dispatch/execution_dispatch_policy.v1.yaml). The allocator then scores that
candidate with the act's OWN belief instead of calling it ``unmeasurable``:

- information: the act cell ``(act:<verb>, actor, signal_id, bin 0)`` in
  ``substrate_action_effect_posterior`` -- the EffectPosterior rung 2 grades every closed act into
  (orion/grammar/act_ledger.py). Its tau^2 is the binary act learner's,
  ``ACT_BINARY_OBSERVATION_VARIANCE``; a cell with no observation is the act learner's cold prior
  and is flagged ``cold_start`` (the allocator's existing, once-per-action exemption).
- cost: the route's own measured motor-seconds (p50 ``latency_ms`` of its past dispatch results,
  which for render_scene is settled to the run's real work time). No typed fallback: an act-backed
  candidate with no measured cost is refused ``no_cost_estimate``.

Nothing here touches the floor, the allowance, the harm gate or any doer's own gates: the
allocator still only admits or refuses, and an admitted act still runs through the doer's full
existing path (for render_scene: policy, express scope, visual-baseline validation, durable
admission, the thermal gate and single-flight in orion-thought).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from orion.autonomy.allocator import Allocation, Candidate, expected_information_gain_nats
from orion.autonomy.prediction import (
    ACT_BINARY_OBSERVATION_VARIANCE,
    EffectPosterior,
    act_dispatch_kind,
    cold_act_posterior,
)

# Posterior cell key used by substrate_action_effect_posterior: (kind, target, signal, bin).
CellKey = tuple[str, str, str, int]

# Every act signal is binary and every rung-2 producer declares "increase" (did it happen by the
# horizon). Carried so the harm gate has a direction; it never fires without a contrast.
ACT_CLAIMED_DIRECTION = "increase"


@dataclass(frozen=True)
class ActRef:
    """The rung-2 act a dispatch route produces: whose belief the allocator should read."""

    verb: str
    actor: str
    signal_id: str
    horizon_sec: float | None = None
    # orion/execution_dispatch/policy.py ActRouteRefV1.object_source: "dispatch" (the dispatch mints
    # the act's object, e.g. a painting run) or "queue" (it picks an existing one, e.g. a reading seed).
    object_source: str = "dispatch"
    # ActRouteRefV1.cost_source / cold_cost_ceiling_sec: "act_duration" scores the act on its own
    # measured open->close time; the ceiling is the doer's enforced maximum, charged while no close
    # is on record (orion/execution_dispatch/policy.py).
    cost_source: str | None = None
    cold_cost_ceiling_sec: float | None = None

    @property
    def cell(self) -> CellKey:
        return (act_dispatch_kind(self.verb), self.actor, self.signal_id, 0)

    def label(self) -> str:
        return f"{self.verb}:{self.actor}:{self.signal_id}"


def act_posterior(posteriors: Mapping[CellKey, EffectPosterior], ref: ActRef) -> tuple[EffectPosterior, bool]:
    """(belief, cold). Cold when the act has never been graded: the act learner's own prior."""
    post = posteriors.get(ref.cell)
    if post is None or post.n <= 0:
        return cold_act_posterior(), True
    return post, False


def act_allocator_candidate(
    *,
    dispatch_id: str,
    dispatch_kind: str,
    target_id: str,
    ref: ActRef,
    posteriors: Mapping[CellKey, EffectPosterior],
    cost_sec: float | None,
    visual_baseline=None,
) -> Candidate:
    post, cold = act_posterior(posteriors, ref)
    cost = cost_sec if cost_sec is not None and math.isfinite(cost_sec) and cost_sec > 0 else None
    return Candidate(
        dispatch_id=dispatch_id,
        dispatch_kind=dispatch_kind,
        target_id=target_id,
        posterior_variance=post.variance,
        cost_sec=cost,
        claimed_direction=ACT_CLAIMED_DIRECTION,
        cold_start=cold,
        visual_baseline=visual_baseline,
        observation_variance=ACT_BINARY_OBSERVATION_VARIANCE,
    )


def act_expectation_reason(ref: ActRef, posteriors: Mapping[CellKey, EffectPosterior],
                           cost_sec: float | None) -> str:
    """One frame-visible line with the numbers the allocator scored this act on (the precommitted
    expectation: predicted rate, its spread, how many closes it rests on, nats, cost)."""
    post, cold = act_posterior(posteriors, ref)
    nats = expected_information_gain_nats(post.variance, ACT_BINARY_OBSERVATION_VARIANCE)
    cost = "none" if cost_sec is None else f"{cost_sec:.1f}"
    rate = "none" if not cost_sec else f"{nats / cost_sec:.6f}"
    return (f"act_expectation:{ref.label()}:predicted={post.mean:.3f}:var={post.variance:.4f}:"
            f"n={post.n}:cold={str(cold).lower()}:nats={nats:.4f}:cost_sec={cost}:nats_per_sec={rate}")


def refuse_in_flight(allocation: Allocation, in_flight: list[Candidate]) -> Allocation:
    """Record act-backed candidates held out of allocate() because their act is still in flight."""
    return Allocation(
        admitted=allocation.admitted,
        refused=(*allocation.refused, *((c, "act_in_flight") for c in in_flight)),
        spent_sec=allocation.spent_sec,
        allowance_sec=allocation.allowance_sec,
    )


def refuse_held(allocation: Allocation, held: list[tuple[Candidate, str]]) -> Allocation:
    """Record act-backed candidates refused before allocate() with their own named reason (the
    route's kill switch, or the doer's last refusal still holding)."""
    return Allocation(
        admitted=allocation.admitted,
        refused=(*allocation.refused, *held),
        spent_sec=allocation.spent_sec,
        allowance_sec=allocation.allowance_sec,
    )


# --- stage 3 option B: admit an act proven to help (Juniper 2026-10-21) ---------------------------
#
# The information path above retires an act once it is well measured, so an act that WORKS is
# refused for being known. This second, separate path admits a measured act whose own belief says it
# helps. It never touches the information floor or its constants; it only reconsiders candidates the
# information path refused `below_information_floor` (measured, costed, not harmful).
#
# "Proven to help", every number from existing code or data, none picked here:
# - enough observations: n >= PROVEN_MIN_N = 20, the per-arm floor the attend-to-act design already
#   fixes for fitting any action's value (2026-09-29 spec, D3: n_per_arm >= max(20, ...));
# - a credible lower bound: LB = mean - HARM_CONFIDENCE_SIGMAS * sd, the allocator's own 2-sigma
#   confidence used by the harm gate, mirrored (stage 3 b in the attend-to-act design);
# - above a reference rate: the act learner's own cold prior mean (ACT_BINARY_PRIOR_MEAN, 0.5,
#   "succeeds more often than the learner assumed before any evidence"), raised to the act's
#   randomized control-arm rate (Jeffreys mean (k+0.5)/(n+1)) once that arm has >= PROVEN_MIN_N
#   resolved rows and reads higher. Re-checked every tick. This is a posterior-vs-reference check,
#   not a randomized contrast (see orion/autonomy/act_episodes.py).
# Value, in the allocator's currency: KL(Bernoulli(LB) || Bernoulli(reference)) nats -- how far the
# act moves its outcome from what happens without it, taken at the conservative LB -- per measured
# motor-second. Admitted greedily by that rate inside the remaining allowance. No floor on this path:
# the information floor exists to retire acts we have learned nothing more from, which is exactly
# what a proven act is; the allowance, holdback, in-flight rule and every doer gate still bound it.

PROVEN_MIN_N = 20
PROVEN_REASON = "admitted:proven_effect"


def reference_rate(n_control: int, successes: int) -> tuple[float, str]:
    from orion.autonomy.prediction import ACT_BINARY_PRIOR_MEAN

    # The bar is never LOWER than the act prior: a control arm may raise it (revoking a claim when
    # not acting does as well as acting), never cut it. Review 2026-10-21: with a structural-zero
    # control (artifact_produced) the Jeffreys rate is ~0.024, which would have made the "check"
    # admit acts 3x more easily than with no arm at all.
    if n_control >= PROVEN_MIN_N:
        control = (successes + 0.5) / (n_control + 1.0)
        if control > ACT_BINARY_PRIOR_MEAN:
            return control, "control_arm"
        return ACT_BINARY_PRIOR_MEAN, "act_prior_above_control"
    return ACT_BINARY_PRIOR_MEAN, "act_prior"


def episode_posterior(n: int, successes: int) -> EffectPosterior:
    """The act's rate from its own resolved episodes (Jeffreys mean, Beta variance): the same
    estimator reference_rate applies to the control arm, so both arms are read alike."""
    mean = (successes + 0.5) / (n + 1.0)
    return EffectPosterior(mean=mean, variance=mean * (1.0 - mean) / (n + 2.0), n=int(n))


def kl_bernoulli(p: float, q: float) -> float:
    eps = 1e-9
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


@dataclass(frozen=True)
class ProvenVerdict:
    proven: bool
    lower_bound: float
    reference: float
    reference_source: str
    value_nats: float
    reason: str


def proven_effect(post: EffectPosterior, *, n_control: int, control_successes: int) -> ProvenVerdict:
    from orion.autonomy.allocator import HARM_CONFIDENCE_SIGMAS

    ref, src = reference_rate(n_control, control_successes)
    lb = post.mean - HARM_CONFIDENCE_SIGMAS * math.sqrt(max(post.variance, 0.0))
    if post.n < PROVEN_MIN_N:
        return ProvenVerdict(False, lb, ref, src, 0.0, f"too_few:{post.n}<{PROVEN_MIN_N}")
    if lb <= ref:
        return ProvenVerdict(False, lb, ref, src, 0.0, f"not_above_reference:lb={lb:.3f}<={ref:.3f}")
    return ProvenVerdict(True, lb, ref, src, kl_bernoulli(min(lb, 1.0), ref), "proven")


def admit_proven(allocation: Allocation, verdicts: Mapping[str, ProvenVerdict]) -> tuple[Allocation, dict[str, str]]:
    """Second pass: re-admit `below_information_floor` refusals whose act is proven to help, best
    value-per-second first, inside the allowance the information path left. Returns the new
    allocation and dispatch_id -> admission reason for every admitted act."""
    pool = [(c, verdicts[c.dispatch_id]) for c, why in allocation.refused
            if why == "below_information_floor" and c.dispatch_id in verdicts
            and verdicts[c.dispatch_id].proven and c.cost_sec]
    pool.sort(key=lambda cv: (-(cv[1].value_nats / cv[0].cost_sec), cv[0].dispatch_id))
    spent = allocation.spent_sec
    admitted = list(allocation.admitted)
    taken: dict[str, str] = {}
    for cand, v in pool:
        if spent + cand.cost_sec > allocation.allowance_sec:
            continue
        admitted.append(cand)
        spent += cand.cost_sec
        taken[cand.dispatch_id] = (
            f"{PROVEN_REASON}:lb={v.lower_bound:.3f}:ref={v.reference:.3f}:{v.reference_source}:"
            f"nats={v.value_nats:.4f}:nats_per_sec={v.value_nats / cand.cost_sec:.6f}")
    refused = tuple((c, why) for c, why in allocation.refused if c.dispatch_id not in taken)
    return Allocation(admitted=tuple(admitted), refused=refused, spent_sec=spent,
                      allowance_sec=allocation.allowance_sec), taken
