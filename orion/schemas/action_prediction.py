"""Predictions attached to autonomous actions, and their scored outcomes.

An action that does not say what it expects to change cannot be wrong, so it
cannot be learned from, so it repeats forever. These two models are the
contract that makes an Orion dispatch falsifiable:

  ExpectedEffectV1     -- written BEFORE the action runs, onto the dispatch
                          candidate. A claim.
  ActionOutcomeRecordV1 -- written AFTER the action's field window closes.
                          The claim, scored.

See orion/autonomy/prediction.py for the theory and the arithmetic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# The signals an action is allowed to make a claim about. These are exactly
# the field pressure channels the feedback runtime already snapshots before
# and after every dispatch (config/feedback/feedback_policy.v1.yaml's
# `pressure_channels`, plus the two derived scalars carried on FieldStateV1
# itself). Deliberately closed: a claim about a signal nothing measures is
# not a claim, and a free-text signal_id would let one back in.
PredictableSignal = Literal[
    "execution_pressure",
    "resource_pressure",
    "reasoning_pressure",
    "reliability_pressure",
    "deviation_pressure",
    "sustained_load_pressure",
]

EffectDirection = Literal["increase", "decrease", "no_change"]


class ExpectedEffectV1(BaseModel):
    """What an action claims it will do, recorded before it does it.

    `predicted_delta` is NOT hand-authored. It is the current posterior mean
    for this (dispatch_kind, target_id, signal_id) triple -- i.e. what the
    action has actually done to this signal so far. A template author
    declares only `signal_id` and `direction`: which signal this action
    touches, and which way they believe it moves. That is the part a human
    genuinely knows and code cannot infer. Magnitude is measured, never
    typed -- typing it would repeat the exact defect this patch exists to
    fix (risk_score is five hand-written constants in a YAML file, 67% of
    them identical, and it is the reason nothing can be ranked).

    `direction` being declarable as "no_change" is load-bearing, not a
    placeholder. A read-only inspect that claims it moves nothing is making
    a real, falsifiable prediction; confirming it repeatedly drives its
    Bayesian surprise to ~0, which is the evidence that the action is a tic.
    """

    model_config = ConfigDict(extra="forbid")

    signal_id: PredictableSignal
    direction: EffectDirection

    predicted_delta: float
    predictor_variance: float = Field(ge=0.0)
    predictor_n: int = Field(ge=0)

    # True when this prediction came from a cold prior rather than from any
    # real observed history. Kept explicit so an analysis can separate
    # "we predicted 0.0 because we have learned it does nothing" from
    # "we predicted 0.0 because we have never seen it run" -- identical
    # floats, opposite meanings.
    cold_start: bool = False


class ActionOutcomeRecordV1(BaseModel):
    """One scored prediction. The unit of the action-outcome ledger."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["action.outcome.record.v1"] = "action.outcome.record.v1"

    dispatch_id: str
    dispatch_frame_id: str
    feedback_frame_id: str

    dispatch_kind: str
    target_id: str
    signal_id: PredictableSignal
    direction: EffectDirection

    observed_at: datetime

    baseline: float
    observed_after: float
    observed_delta: float

    predicted_delta: float
    prediction_error: float

    surprise_nats: float = Field(ge=0.0)

    posterior_mean: float
    posterior_variance: float = Field(ge=0.0)
    posterior_n: int = Field(ge=0)

    # Attribution honesty. The field delta is measured frame-wide, so when
    # several candidates in the same tick claim the same signal, none of
    # them individually caused the whole delta. This records how many did,
    # so an analysis can split sole-attribution samples (co_predictors == 0,
    # clean) from shared ones rather than the ambiguity being invisible in
    # the stored row. Phase 1 updates the posterior on both and reports the
    # split; it does not silently drop shared samples.
    co_predictors: int = Field(ge=0, default=0)

    # Real measured cost of the action, when the cortex result carried one.
    # None means the result did not report it -- never coerced to 0.0, which
    # would read as "free".
    latency_ms: float | None = None
