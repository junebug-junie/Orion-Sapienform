"""orion-prediction settings. Behaviour flags default ON (they are the rollback lever, not the rollout gate)."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # env_ignore_empty: an empty ORION_SITUATION_WEATHER_LAT= (the template default) means "unset",
    # not a float that fails to parse and stops the service at boot.
    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True, env_ignore_empty=True)

    service_name: str = Field("orion-prediction", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    orion_bus_url: str = Field(..., alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")
    postgres_uri: str = Field(..., alias="POSTGRES_URI")

    series_path: str = Field("/app/config/prediction/series.v1.yaml", alias="PREDICTION_SERIES_PATH")

    # --- kill switches ---
    enabled: bool = Field(True, alias="PREDICTION_ENABLED")                  # run the forecast loop at all
    publish_enabled: bool = Field(True, alias="PREDICTION_PUBLISH_ENABLED")  # publish forecasts/alerts on the bus

    tick_sec: float = Field(30.0, gt=0, alias="PREDICTION_TICK_SEC")
    # A bucket counts as complete this long after it ends (late readings still land).
    settle_sec: float = Field(60.0, ge=0, alias="PREDICTION_SETTLE_SEC")
    select_interval_sec: float = Field(21600.0, gt=0, alias="PREDICTION_SELECT_INTERVAL_SEC")
    backtest_windows: int = Field(40, ge=12, le=200, alias="PREDICTION_BACKTEST_WINDOWS")
    backtest_step_min: int = Field(180, ge=5, alias="PREDICTION_BACKTEST_STEP_MIN")

    # Forecast every contender model in the shadow and grade it live; the gate uses the live verdicts.
    shadow_enabled: bool = Field(True, alias="PREDICTION_SHADOW_ENABLED")

    # World-model shadow for action bets: the exported TransitionModel weights (written by
    # services/orion-world-model/scripts/train_transition.py --publish). Scored, stored, graded; never served.
    world_model_shadow_enabled: bool = Field(True, alias="PREDICTION_WORLD_MODEL_SHADOW_ENABLED")
    world_model_weights: str = Field("/models/transition.npz", alias="PREDICTION_WORLD_MODEL_WEIGHTS")

    # --- weather covariates (shared Open-Meteo client, orion/weather/openmeteo.py) ---
    weather_enabled: bool = Field(True, alias="PREDICTION_WEATHER_ENABLED")
    weather_refresh_sec: float = Field(3600.0, ge=300, alias="PREDICTION_WEATHER_REFRESH_SEC")
    # Same keys and values as the situation layer's location (orion-hub's .env); empty = weather off.
    weather_lat: float | None = Field(None, alias="ORION_SITUATION_WEATHER_LAT")
    weather_lon: float | None = Field(None, alias="ORION_SITUATION_WEATHER_LON")

    # --- switchback experiment runner (app/experiment_runner.py, POST /experiments/run) ---
    # Performs vision.focus:<stream_id> experiments: holds orion:vision:experiment:<stream_id> = on
    # (bus Redis, ORION_BUS_URL) during ON blocks. false = POST /experiments/run answers 503; the
    # design/log/analyze endpoints still work for hand-run experiments. Never starts on boot.
    experiment_runner_enabled: bool = Field(True, alias="PREDICTION_EXPERIMENT_RUNNER_ENABLED")
    # The frame router's /healthz (per-stream dispatch tier counts = the manipulation check).
    vision_router_url: str = Field("http://orion-vision-frame-router:8010", alias="PREDICTION_VISION_ROUTER_URL")
    # TTL on the override key, refreshed every tick: a dead runner leaves focus on for at most this long.
    experiment_override_ttl_sec: int = Field(90, ge=30, le=600, alias="PREDICTION_EXPERIMENT_OVERRIDE_TTL_SEC")
    experiment_tick_sec: float = Field(30.0, ge=5, le=60, alias="PREDICTION_EXPERIMENT_TICK_SEC")
    # Re-run the experiment guard (a forecast build) this often during a run, and before every ON block.
    experiment_guard_every_sec: float = Field(300.0, ge=60, le=1800, alias="PREDICTION_EXPERIMENT_GUARD_EVERY_SEC")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
