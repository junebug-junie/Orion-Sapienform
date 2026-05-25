from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    SERVICE_NAME: str = Field(default="orion-bus")
    SERVICE_VERSION: str = Field(default="0.1.0")

    REDIS_URL: str = Field(default="redis://bus-core:6379/0")

    publish_orion_bus_grammar: bool = Field(False, alias="PUBLISH_ORION_BUS_GRAMMAR")
    grammar_event_channel: str = Field("orion:grammar:event", alias="GRAMMAR_EVENT_CHANNEL")

    bus_observer_enabled: bool = Field(True, alias="BUS_OBSERVER_ENABLED")
    bus_observer_poll_interval_sec: float = Field(10.0, alias="BUS_OBSERVER_POLL_INTERVAL_SEC")
    bus_observer_streams: str = Field(
        "orion:evt:gateway,orion:bus:out",
        alias="BUS_OBSERVER_STREAMS",
    )
    bus_stream_depth_warning: int = Field(25000, alias="BUS_STREAM_DEPTH_WARNING")
    bus_stream_depth_critical: int = Field(100000, alias="BUS_STREAM_DEPTH_CRITICAL")
    bus_observer_node_id: str = Field("athena", alias="BUS_OBSERVER_NODE_ID")

    channels_catalog_path: str = Field(
        "orion/bus/channels.yaml",
        alias="BUS_CHANNELS_CATALOG_PATH",
    )

    @property
    def observer_stream_list(self) -> list[str]:
        return [s.strip() for s in self.bus_observer_streams.split(",") if s.strip()]


settings = Settings()
