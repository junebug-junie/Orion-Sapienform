from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


def _parse_list(val: str | None) -> List[str]:
    if not val:
        return []
    return [v.strip() for v in val.split(",") if v.strip()]


class Settings(BaseSettings):
    service_name: str = Field("orion-equilibrium-service", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: Optional[str] = Field(None, alias="NODE_NAME")
    instance_id: Optional[str] = Field(None, alias="INSTANCE_ID")

    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_enforce_catalog: bool = Field(False, alias="ORION_BUS_ENFORCE_CATALOG")
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")
    health_channel: str = Field("orion:system:health", alias="ORION_HEALTH_CHANNEL")

    expected_services_raw: Optional[str] = Field(None, alias="EQUILIBRIUM_EXPECTED_SERVICES")
    expected_services_path: Optional[Path] = Field(None, alias="EQUILIBRIUM_EXPECTED_SERVICES_PATH")
    grace_multiplier: float = Field(3.0, alias="EQUILIBRIUM_GRACE_MULTIPLIER")
    publish_interval_sec: float = Field(15.0, alias="EQUILIBRIUM_PUBLISH_INTERVAL_SEC")
    windows_sec: List[int] = Field(default_factory=lambda: [60, 300, 3600], alias="EQUILIBRIUM_WINDOWS_SEC")
    collapse_mirror_interval_sec: float = Field(15.0, alias="EQUILIBRIUM_COLLAPSE_MIRROR_INTERVAL_SEC")

    redis_state_key: str = Field("equilibrium:state", alias="EQUILIBRIUM_STATE_KEY")
    channel_equilibrium_snapshot: str = Field("orion:equilibrium:snapshot", alias="CHANNEL_EQUILIBRIUM_SNAPSHOT")
    channel_spark_signal: str = Field("orion:spark:signal", alias="CHANNEL_SPARK_SIGNAL")

    # Metacognition
    metacog_enable: bool = Field(False, alias="EQUILIBRIUM_METACOG_ENABLE")
    metacog_baseline_interval_sec: float = Field(60.0, alias="EQUILIBRIUM_METACOG_BASELINE_INTERVAL_SEC")
    metacog_cooldown_sec: float = Field(30.0, alias="EQUILIBRIUM_METACOG_COOLDOWN_SEC")
    metacog_pad_pulse_threshold: float = Field(0.8, alias="EQUILIBRIUM_METACOG_PAD_PULSE_THRESHOLD")

    channel_metacog_trigger: str = Field("orion:equilibrium:metacog:trigger", alias="CHANNEL_EQUILIBRIUM_METACOG_TRIGGER")
    channel_cortex_orch_request: str = Field("orion:verb:request", alias="CHANNEL_CORTEX_ORCH_REQUEST")
    channel_collapse_mirror_user_event: str = Field("orion:collapse:intake", alias="CHANNEL_COLLAPSE_MIRROR_USER_EVENT")
    channel_pad_signal: str = Field("orion:pad:signal", alias="CHANNEL_PAD_SIGNAL")

    class Config:
        env_file = ".env"
        extra = "ignore"

    @model_validator(mode="after")
    def _coerce_windows(self) -> "Settings":
        if isinstance(self.windows_sec, str):
            try:
                parsed = json.loads(self.windows_sec)
                if isinstance(parsed, list):
                    self.windows_sec = [int(x) for x in parsed]
            except Exception:
                self.windows_sec = [int(x) for x in _parse_list(self.windows_sec)]
        return self

    def expected_services(self) -> List[str]:
        items: List[str] = []
        items.extend(_parse_list(self.expected_services_raw))

        if self.expected_services_path and self.expected_services_path.exists():
            try:
                import yaml

                raw = yaml.safe_load(self.expected_services_path.read_text()) or {}
                from_file = raw.get("services") if isinstance(raw, dict) else raw
                if isinstance(from_file, list):
                    items.extend([str(x) for x in from_file])
            except Exception:
                pass

        # ensure uniqueness while preserving order
        deduped: List[str] = []
        for it in items:
            if it not in deduped:
                deduped.append(it)
        return deduped


settings = Settings()
