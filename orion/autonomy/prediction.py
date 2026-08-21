"""Bayesian surprise over action outcome predictions.

WHY THIS EXISTS
---------------
Orion's autonomy loop dispatches ~5,400 real actions a day drawn from 15
distinct targets, and until this module nothing scored whether any of them
did anything. The `surprise` field already written onto every
`action_outcomes` row is a global bus-traffic reading
(`latest_bus_synaptic_prediction_error()`), fetched once per tick and
stamped identically onto every candidate in that tick -- pruning Docker
images and reading `channels.yaml` get the same number, because the number
describes the message bus and not the action. It cannot separate a useful
action from a tic.

This module supplies the missing quantity: for a declared (action, signal)
pair, how much did observing the real outcome move our belief about what
that action does.

THEORY ANCHOR (not vibes)
-------------------------
Bayesian surprise (Itti & Baldi, "Bayesian surprise attracts human
attention", Vision Research 49(10), 2009) defines the information an
observation carries as the KL divergence between the posterior and the
prior it updated:

    S = KL(P(M | D) || P(M))                      [nats]

This is exactly the epistemic-value term of expected free energy in active
inference (Friston et al.), which is why it is the right common scale here:
an action that achieves a preferred state and an action that reduces
uncertainty both reduce to nats, so they can compete without an invented
conversion rate.

The practical property that makes this worth wiring in: for a repeated
action whose effect is already well estimated, the posterior stops moving
and S -> 0. An action Orion has taken 7,583 times with a stable outcome
earns approximately nothing, automatically, with no hand-written
"don't repeat yourself" rule. Redundancy stops paying on its own.

MODEL
-----
Normal-Normal conjugate update with known observation variance. The latent
quantity is `mu`, the mean effect of one (dispatch_kind, target_id,
signal_id) action on that signal, measured as the signal's delta across the
action's field window.

    prior      mu ~ N(mu0, var0)
    likelihood x | mu ~ N(mu, obs_var)
    posterior  mu ~ N(mu1, var1)

    precision1 = 1/var0 + 1/obs_var
    var1       = 1 / precision1
    mu1        = var1 * (mu0/var0 + x/obs_var)

and the closed-form KL between two univariate normals:

    KL(N(mu1,var1) || N(mu0,var0))
        = 0.5 * ( ln(var0/var1) + (var1 + (mu1-mu0)^2)/var0 - 1 )

Natural log, so the result is in nats.

CONSTANT PROVENANCE
-------------------
Both defaults below are derived from THIS repo's real data, not borrowed
from another domain (see CLAUDE.md's metric gate and the
"borrowed calibrated constants don't transfer" lesson). Measured over
68,715 `substrate_feedback_frames` pressure_delta observations across the
3 days ending 2026-08-21:

    channel                n       zero-ish    min       max      stddev
    execution_pressure     68715   24495       -0.392    0.392    0.175
    reasoning_pressure     68715   56189       -0.900    0.855    0.057
    reliability_pressure   68715   60428       -0.900    0.900    0.293
    resource_pressure      68715   13089       -0.842    0.788    0.211

Pooled per-observation spread is ~0.2, so DEFAULT_OBSERVATION_VARIANCE is
0.2^2. The prior is deliberately wider than any single observation
(sd 0.5 on a signal whose delta is bounded by +/-1.0) so a cold-start
posterior is dominated by real data within a handful of samples rather
than by the prior. Both are first-cut and disclosed; re-derive from live
`substrate_action_outcomes` rows once phase 1 has run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_PRIOR_MEAN = 0.0
DEFAULT_PRIOR_VARIANCE = 0.25
DEFAULT_OBSERVATION_VARIANCE = 0.04

# Guards against a degenerate posterior. A variance driven to (or below)
# zero makes the KL above diverge to +inf, which would let one pathological
# (action, signal) pair dominate every budget comparison forever. 1e-6 is a
# posterior sd of 1e-3 -- three orders of magnitude finer than the finest
# real signal movement measured above, so it can never bind on real data
# and only ever fires on arithmetic that has already gone wrong.
MIN_VARIANCE = 1e-6


@dataclass(frozen=True)
class EffectPosterior:
    """Belief about the mean effect of one action on one signal."""

    mean: float
    variance: float
    n: int

    @classmethod
    def cold(cls) -> "EffectPosterior":
        return cls(mean=DEFAULT_PRIOR_MEAN, variance=DEFAULT_PRIOR_VARIANCE, n=0)

    @property
    def sd(self) -> float:
        return math.sqrt(max(self.variance, 0.0))


def update_posterior(
    prior: EffectPosterior,
    observation: float,
    *,
    observation_variance: float = DEFAULT_OBSERVATION_VARIANCE,
) -> EffectPosterior:
    """One Normal-Normal conjugate update. Pure; no I/O."""
    if not math.isfinite(observation):
        raise ValueError(f"observation must be finite, got {observation!r}")
    if not math.isfinite(observation_variance) or observation_variance <= 0.0:
        raise ValueError(
            f"observation_variance must be finite and > 0, got {observation_variance!r}"
        )

    var0 = max(prior.variance, MIN_VARIANCE)
    precision = (1.0 / var0) + (1.0 / observation_variance)
    var1 = max(1.0 / precision, MIN_VARIANCE)
    mu1 = var1 * ((prior.mean / var0) + (observation / observation_variance))
    return EffectPosterior(mean=mu1, variance=var1, n=prior.n + 1)


def bayesian_surprise_nats(prior: EffectPosterior, posterior: EffectPosterior) -> float:
    """KL(posterior || prior) for two univariate normals, in nats.

    Zero exactly when the update moved nothing. Never negative (KL is
    non-negative by construction); the max() only absorbs float error near
    zero, it is not papering over a sign bug.
    """
    var0 = max(prior.variance, MIN_VARIANCE)
    var1 = max(posterior.variance, MIN_VARIANCE)
    mean_shift = posterior.mean - prior.mean
    kl = 0.5 * (math.log(var0 / var1) + (var1 + mean_shift * mean_shift) / var0 - 1.0)
    return max(kl, 0.0)


def score_observation(
    prior: EffectPosterior,
    observation: float,
    *,
    observation_variance: float = DEFAULT_OBSERVATION_VARIANCE,
) -> tuple[EffectPosterior, float, float]:
    """Convenience: update and score in one call.

    Returns (posterior, surprise_nats, prediction_error) where
    prediction_error is the raw residual `observation - prior.mean` -- the
    plain, un-transformed "how wrong was I", kept alongside the nats because
    a residual is directly auditable against the stored signal values and
    the nats are not.
    """
    posterior = update_posterior(
        prior, observation, observation_variance=observation_variance
    )
    return posterior, bayesian_surprise_nats(prior, posterior), observation - prior.mean
