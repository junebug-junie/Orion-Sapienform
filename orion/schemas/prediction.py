"""orion-prediction bus payloads: the latest served forecast per series, and its monitoring alerts.

Spec: docs/superpowers/specs/2026-10-15-orion-prediction-service-design.md.

``orion-prediction`` publishes one ``PredictionForecastV1`` on ``orion:prediction:forecast`` each
time a new grain bucket completes for a target series. Each horizon is served by the model that
passed the backtest gate there, else by the better of the two baselines (persistence or
same-time-yesterday); ``model`` lists the per-horizon assignment, and ``served_baseline=true`` means
every horizon is a baseline. Quantiles are split-conformal from the served models' backtest
residual paths, not the models' internal intervals.

``exceedance.probability`` is the share of backtest residual *paths* that, added to this forecast,
reach the threshold at any step inside the window. It is about the series' own aggregate (for the
cabinet: 5-minute MEAN temperature), so it understates the chance that a single raw reading
crosses the line, which is what the 34 C reflex acts on. ``None`` means too few complete residual paths
to say -- never read it as 0.

Monitoring alerts are not a bus event (nothing would consume one): they are in orion-prediction's
/health and /series/{id}/live, and logged as ``prediction_alerts_changed`` when the set changes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

PREDICTION_FORECAST_CHANNEL = "orion:prediction:forecast"
PREDICTION_FORECAST_KIND = "prediction.forecast.v1"


class PredictionStepV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_ts: AwareDatetime
    horizon_min: int = Field(..., ge=1)
    point: float
    # Quantile level as a string key ("0.1") -> value. Monotone non-decreasing in level.
    quantiles: dict[str, float]


class PredictionExceedanceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    threshold: float
    window_min: int = Field(..., ge=1)
    probability: float | None = Field(None, ge=0.0, le=1.0)
    n_paths: int = Field(..., ge=0)


class PredictionForecastV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    series_id: str
    model: str
    served_baseline: bool
    run_id: str
    issued_at: AwareDatetime
    origin_ts: AwareDatetime  # start of the last complete grain bucket
    grain_min: int = Field(..., ge=1)
    claim_kind: Literal["forecast"] = "forecast"
    steps: list[PredictionStepV1]
    exceedance: list[PredictionExceedanceV1] = Field(default_factory=list)



# --- Action bets (docs/superpowers/specs/2026-10-23-embodied-experience-learning-design.md) -------
#
# HTTP contracts only (orion-prediction :8132, /action-bets/*); no bus channel in v1. A body asks for
# a quote on its options, places a bet on the one it chose (the service computes ``p`` itself at
# placing, so no caller can supply a probability after the fact) and settles it. ``skill`` and
# ``context`` use the game-free vocabulary in orion/embodiment/skills.py; the service treats them
# as opaque keys. Settled bets are graded with app/grading.py like every other probability.

ACTION_BET_MAX_OPTIONS = 20
ActionBetResult = Literal["success", "failure", "void"]
ActionBetLevel = Literal["place", "situation", "skill", "prior"]


class ActionBetOptionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    option_id: str = Field(..., min_length=1, max_length=64)
    skill: str = Field(..., min_length=1, max_length=32)
    context: dict[str, str] = Field(default_factory=dict, max_length=8)
    place: str | None = Field(None, max_length=128)


class ActionBetQuoteRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str = Field(..., min_length=1, max_length=64)   # 'retro:zelda'
    body: str = Field(..., min_length=1, max_length=32)     # 'retro'
    options: list[ActionBetOptionV1] = Field(..., min_length=1, max_length=ACTION_BET_MAX_OPTIONS)


class ActionBetCountsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: ActionBetLevel
    successes: int = Field(..., ge=0)
    failures: int = Field(..., ge=0)


class ActionBetQuoteV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    option_id: str
    model: str                       # 'action_beta_backoff', or the baseline when the gate serves it
    p: float = Field(..., ge=0.0, le=1.0)
    p_baseline: float = Field(..., ge=0.0, le=1.0)
    interval80: tuple[float, float]
    eig_nats: float = Field(..., ge=0.0)
    level: ActionBetLevel            # deepest level with at least one settled bet
    counts: list[ActionBetCountsV1]  # skill first, then situation, then place (those that apply)


class ActionBetPlaceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bet_id: str = Field(..., min_length=1, max_length=200)
    domain: str = Field(..., min_length=1, max_length=64)
    body: str = Field(..., min_length=1, max_length=32)
    skill: str = Field(..., min_length=1, max_length=32)
    context: dict[str, str] = Field(default_factory=dict, max_length=8)
    place: str | None = Field(None, max_length=128)
    action: dict = Field(default_factory=dict)       # the body's own goal, for inspection only
    chosen_rank: int | None = Field(None, ge=1, le=ACTION_BET_MAX_OPTIONS)
    session_id: str | None = Field(None, max_length=128)


class ActionBetSettleV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: ActionBetResult
    outcome: str = Field(..., min_length=1, max_length=64)   # the body's own words
    detail: dict = Field(default_factory=dict)


class ActionBetV1(BaseModel):
    """One stored bet, as returned by place, settle and the list endpoints."""
    model_config = ConfigDict(extra="forbid")

    bet_id: str
    domain: str
    body: str
    skill: str
    context: dict[str, str]
    place: str | None = None
    action: dict
    model: str
    p: float = Field(..., ge=0.0, le=1.0)
    p_baseline: float = Field(..., ge=0.0, le=1.0)
    level: ActionBetLevel
    counts: list[ActionBetCountsV1]
    chosen_rank: int | None = None
    session_id: str | None = None
    placed_at: AwareDatetime
    settled_at: AwareDatetime | None = None
    result: ActionBetResult | None = None
    outcome: str | None = None
    detail: dict = Field(default_factory=dict)
