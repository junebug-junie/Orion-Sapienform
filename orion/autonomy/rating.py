"""Score a human rating of an Orion-produced artifact.

WHY THIS IS NOT THE OTHER LEDGER
--------------------------------
`orion/autonomy/contrast.py` measures what an action did to a field pressure:
`after - before`, matched against an untreated control arm, because pressures
mean-revert and actions fire when they are high, so the raw delta is mostly
regression to the mean.

**A rating has no "before" and does not mean-revert.** There is no confound to
subtract; the value IS the rating. So this module reuses the posterior and the
Bayesian surprise -- the same common scale, which is the entire reason for
having one -- and reuses neither the baseline bins nor the arms. Forcing a
rating through the pressure-shaped hole because the hole exists would be the
mirror image of the defect that ledger was built to remove.

WHY IT MATTERS THAT THIS IS NOT A PRESSURE
------------------------------------------
Every outcome Orion could previously claim was one of six field pressures
(`PredictableSignal`), all of them derived from its own CPU/disk/GPU
telemetry. The action, the outcome and the grader all lived inside Orion --
homework it marks itself, which is a large part of why every measured action
scores approximately zero. A rating is decided by someone else. It is the
first outcome in this system that Orion cannot produce.

MAGNITUDE IS NOT INVENTED
-------------------------
A rating scores +1 or -1. The `categories` on the feedback event are recorded
and NOT scored, deliberately: five thumbs-down categories is not five times
worse than one, and turning a count of labels into a magnitude would be
exactly the defect this whole arc exists to delete (`risk_score` is five
hand-typed constants in a YAML file, 67% of them identical, and it is the
reason nothing could be ranked). The categories say *why*, and why is for
reading, not for arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from orion.autonomy.prediction import EffectPosterior, bayesian_surprise_nats, update_posterior

RatingValue = Literal["up", "down"]

# A rating is +/-1. Nothing between, because nothing produces anything between.
RATING_SCALARS: dict[str, float] = {"up": 1.0, "down": -1.0}

# UNCALIBRATED, AND SAID SO. The pressure priors in orion/autonomy/prediction.py
# were fitted to 68,715 real observations. There are 2 human ratings in this
# database, total, over three weeks -- so there is nothing to fit and these are
# max-entropy choices, not measurements:
#
#   prior variance 1.0  -- for a variable on [-1, +1] whose observations are
#       +/-1, the largest possible variance is 1.0 (a fair coin on the two
#       endpoints). Starting there asserts no belief at all about whether an
#       action produces good artifacts.
#   observation variance 1.0 -- same bound, same reasoning. It makes one
#       explicit human rating carry the same weight as the entire prior, so
#       the first rating moves the mean to +/-0.5 and the fourth to +/-0.8.
#       That is intended: a person taking the trouble to rate something is a
#       strong signal, and pretending otherwise would need ratings nobody is
#       going to give.
#
# Recalibrate from the real distribution once there are enough ratings to have
# one, and do not carry these numbers to any other domain -- borrowed
# calibrated constants do not transfer, and these are not even calibrated.
RATING_PRIOR_VARIANCE = 1.0
RATING_OBSERVATION_VARIANCE = 1.0

# (dispatch_kind, target_id). No signal component: the rating IS the signal.
# No baseline bin and no arm, for the reasons in the module docstring.
RatingKey = tuple[str, str]


def rating_scalar(feedback_value: str) -> float:
    if feedback_value not in RATING_SCALARS:
        raise ValueError(
            f"unscoreable rating {feedback_value!r}; expected one of "
            f"{sorted(RATING_SCALARS)}"
        )
    return RATING_SCALARS[feedback_value]


def cold_rating_prior() -> EffectPosterior:
    return EffectPosterior(mean=0.0, variance=RATING_PRIOR_VARIANCE, n=0)


@dataclass(frozen=True)
class ScoredRating:
    """One human rating, scored against what we believed before it."""

    artifact_ref: str
    dispatch_id: str
    dispatch_kind: str
    target_id: str

    feedback_value: RatingValue
    rating: float
    categories: tuple[str, ...]
    free_text: str | None

    predicted_rating: float
    prediction_error: float
    surprise_nats: float

    posterior_mean: float
    posterior_variance: float
    posterior_n: int

    rated_at: datetime


def score_rating(
    *,
    artifact_ref: str,
    dispatch_id: str,
    dispatch_kind: str,
    target_id: str,
    feedback_value: str,
    categories: list[str] | tuple[str, ...] | None,
    free_text: str | None,
    rated_at: datetime,
    prior: EffectPosterior | None = None,
) -> ScoredRating:
    """Fold one rating into the belief about what this action produces.

    Pure. No I/O. The caller supplies the prior and persists the result.
    """
    observed = rating_scalar(feedback_value)
    before = prior or cold_rating_prior()
    posterior = update_posterior(
        before, observed, observation_variance=RATING_OBSERVATION_VARIANCE
    )
    return ScoredRating(
        artifact_ref=artifact_ref,
        dispatch_id=dispatch_id,
        dispatch_kind=dispatch_kind,
        target_id=target_id,
        feedback_value=feedback_value,  # type: ignore[arg-type]
        rating=observed,
        categories=tuple(categories or ()),
        free_text=free_text,
        # What we expected before seeing it. Unlike the pressure ledger --
        # where the claim is stamped on the candidate at dispatch time and the
        # residual must be measured against THAT, not against the belief at
        # scoring time -- nothing is claimed here in advance, so the prior IS
        # the prediction and the two cannot drift apart.
        predicted_rating=before.mean,
        prediction_error=observed - before.mean,
        surprise_nats=bayesian_surprise_nats(before, posterior),
        posterior_mean=posterior.mean,
        posterior_variance=posterior.variance,
        posterior_n=posterior.n,
        rated_at=rated_at,
    )


def rating_key(dispatch_kind: str, target_id: str) -> RatingKey:
    return (dispatch_kind, target_id)
