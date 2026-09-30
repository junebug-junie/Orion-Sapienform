"""orion-hardware-watch settings. Behaviour flags default OFF in code, ON in .env_example."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    service_name: str = Field("orion-hardware-watch", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    orion_bus_url: str = Field(..., alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")
    postgres_uri: str = Field(..., alias="POSTGRES_URI")
    notify_base_url: str = Field("http://orion-athena-notify:7140", alias="NOTIFY_BASE_URL")
    notify_api_token: str = Field("", alias="NOTIFY_API_TOKEN")

    # --- behaviour flags (kill switches) ---
    enabled: bool = Field(False, alias="HARDWARE_WATCH_ENABLED")               # evaluate rules at all
    shed_enabled: bool = Field(False, alias="HARDWARE_WATCH_SHED_ENABLED")     # ask the pool to shed
    urgent_enabled: bool = Field(False, alias="HARDWARE_WATCH_URGENT_ENABLED") # start urgent runs
    test_hook_enabled: bool = Field(False, alias="HARDWARE_WATCH_TEST_HOOK_ENABLED")  # POST /incidents/simulate

    tick_sec: float = Field(30.0, gt=0, alias="HARDWARE_WATCH_TICK_SEC")
    refresh_sec: float = Field(60.0, gt=0, alias="HARDWARE_WATCH_REFRESH_SEC")
    shed_valid_sec: float = Field(300.0, gt=0, alias="HARDWARE_WATCH_SHED_VALID_SEC")
    operator_snooze_sec: float = Field(3600.0, ge=0, alias="HARDWARE_WATCH_OPERATOR_SNOOZE_SEC")
    alert_max_attempts: int = Field(60, ge=1, alias="HARDWARE_WATCH_ALERT_MAX_ATTEMPTS")

    # --- AC rule (subject cabinet_ac) ---
    ac_role: str = Field("cabinet_cooling", alias="HARDWARE_WATCH_AC_ROLE")   # home_cooling_sample.role
    ac_low_w: float = Field(150.0, alias="HARDWARE_WATCH_AC_LOW_W")
    ac_low_sec: float = Field(180.0, alias="HARDWARE_WATCH_AC_LOW_SEC")
    ac_stale_sec: float = Field(300.0, alias="HARDWARE_WATCH_AC_STALE_SEC")
    ac_frozen_sec: float = Field(3600.0, alias="HARDWARE_WATCH_AC_FROZEN_SEC")
    ac_resolve_w: float = Field(500.0, alias="HARDWARE_WATCH_AC_RESOLVE_W")
    ac_resolve_sec: float = Field(600.0, alias="HARDWARE_WATCH_AC_RESOLVE_SEC")

    # --- shed (cabinet sensor on this node's biometrics summary) ---
    cabinet_node: str = Field("athena", alias="HARDWARE_WATCH_CABINET_NODE")
    shed_rise_c: float = Field(1.0, alias="HARDWARE_WATCH_SHED_RISE_C")
    shed_rise_window_sec: float = Field(900.0, alias="HARDWARE_WATCH_SHED_RISE_WINDOW_SEC")

    # --- heat ---
    heat_nodes: str = Field("athena,circe", alias="HARDWARE_WATCH_HEAT_NODES")
    gpu_nodes: str = Field("athena,circe", alias="HARDWARE_WATCH_GPU_NODES")
    heat_sustain_sec: float = Field(600.0, alias="HARDWARE_WATCH_HEAT_SUSTAIN_SEC")
    heat_baseline_days: float = Field(7.0, gt=0, alias="HARDWARE_WATCH_HEAT_BASELINE_DAYS")
    heat_baseline_refresh_sec: float = Field(3600.0, gt=0, alias="HARDWARE_WATCH_HEAT_BASELINE_REFRESH_SEC")
    cpu_min_history_sec: float = Field(86400.0, alias="HARDWARE_WATCH_CPU_MIN_HISTORY_SEC")
    gpu_min_history_sec: float = Field(259200.0, alias="HARDWARE_WATCH_GPU_MIN_HISTORY_SEC")
    gpu_ceiling_c: float = Field(85.0, alias="HARDWARE_WATCH_GPU_CEILING_C")
    gpu_ceiling_sustain_sec: float = Field(120.0, alias="HARDWARE_WATCH_GPU_CEILING_SUSTAIN_SEC")
    gpu_ceiling_rearm_c: float = Field(80.0, alias="HARDWARE_WATCH_GPU_CEILING_REARM_C")

    @model_validator(mode="after")
    def _shed_signal_outlives_refresh(self) -> "Settings":
        # The pool's copy of a shed request lapses at shed_valid_sec; the watcher must re-publish
        # (every refresh_sec, checked every tick_sec) well before that or shedding flaps.
        if not (self.tick_sec < self.shed_valid_sec and self.refresh_sec < self.shed_valid_sec):
            raise ValueError("HARDWARE_WATCH_TICK_SEC and HARDWARE_WATCH_REFRESH_SEC must both be < "
                             "HARDWARE_WATCH_SHED_VALID_SEC")
        return self

    @property
    def heat_node_list(self) -> list[str]:
        return _csv(self.heat_nodes)

    @property
    def gpu_node_list(self) -> list[str]:
        return _csv(self.gpu_nodes)


@lru_cache
def get_settings() -> Settings:
    return Settings()
