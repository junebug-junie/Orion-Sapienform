# services/orion-psu-proxy/app/settings.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyUrl, Field


class Settings(BaseSettings):
    # Identity
    project: str = Field("orion-athena", alias="PROJECT")
    node_name: str = Field("athena", alias="NODE_NAME")

    # service name you’re passing from compose
    service_name: str = Field("gpu-cluster-power", alias="SERVICE_NAME")

    # Storage
    storage_root: str = Field("/mnt/storage-lukewarm", alias="STORAGE_ROOT")
    telemetry_root: str = Field("/mnt/telemetry", alias="TELEMETRY_ROOT")

    # PSU board config
    psu_base_url: str = Field("http://192.168.0.221", alias="PSU_BASE_URL")
    psu_on_path: str = Field("/power", alias="PSU_ON_PATH")
    psu_off_path: str = Field("/power", alias="PSU_OFF_PATH")
    psu_cycle_path: str = Field("/power", alias="PSU_CYCLE_PATH")

    # Optional HTTP auth token for this proxy
    api_token: str | None = Field(None, alias="API_TOKEN")

    # Orion bus
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_url: AnyUrl | None = Field(None, alias="ORION_BUS_URL")

    # Bus channels
    bus_channel_psu_command: str = Field(
        "orion:power:psu:command", alias="BUS_CHANNEL_PSU_COMMAND"
    )
    bus_channel_psu_events: str = Field(
        "orion:power:psu:events", alias="BUS_CHANNEL_PSU_EVENTS"
    )

    model_config = SettingsConfigDict(
        extra="ignore",
    )


settings = Settings()
