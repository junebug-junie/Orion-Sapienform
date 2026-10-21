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
