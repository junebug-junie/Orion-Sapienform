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

    # --- weather covariates (shared Open-Meteo client, orion/weather/openmeteo.py) ---
    weather_enabled: bool = Field(True, alias="PREDICTION_WEATHER_ENABLED")
    weather_refresh_sec: float = Field(3600.0, ge=300, alias="PREDICTION_WEATHER_REFRESH_SEC")
    # Same keys and values as the situation layer's location (orion-hub's .env); empty = weather off.
    weather_lat: float | None = Field(None, alias="ORION_SITUATION_WEATHER_LAT")
    weather_lon: float | None = Field(None, alias="ORION_SITUATION_WEATHER_LON")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
