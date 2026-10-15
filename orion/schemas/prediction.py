"""orion-prediction bus payloads: the latest served forecast per series, and its monitoring alerts.

Spec: docs/superpowers/specs/2026-10-15-orion-prediction-service-design.md.

``orion-prediction`` publishes one ``PredictionForecastV1`` on ``orion:prediction:forecast`` each
time a new grain bucket completes for a target series. Each horizon is served by the model that
passed the backtest gate there, else by the better of the two baselines (persistence or
same-time-yesterday); ``model`` lists the per-horizon assignment, and ``served_baseline=true`` means
every horizon is a baseline. Quantiles are split-conformal from the served models' backtest
residual paths, not the models' internal intervals.

``exceedance.probability`` is the share of backtest residual *paths* that, added to this forecast,
reach the threshold at any step inside the window. ``None`` means too few complete residual paths
to say -- never read it as 0.

Monitoring alerts are not a bus event (nothing would consume one): they are in orion-prediction's
/health and /series/{id}/live, and logged as ``prediction_alerts_changed`` when the set changes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PREDICTION_FORECAST_CHANNEL = "orion:prediction:forecast"
PREDICTION_FORECAST_KIND = "prediction.forecast.v1"


class PredictionStepV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_ts: datetime
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
    issued_at: datetime
    origin_ts: datetime  # start of the last observed grain bucket
    grain_min: int = Field(..., ge=1)
    claim_kind: Literal["forecast"] = "forecast"
    steps: list[PredictionStepV1]
    exceedance: list[PredictionExceedanceV1] = Field(default_factory=list)

