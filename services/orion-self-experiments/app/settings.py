from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    service_name: str = Field("orion-self-experiments", alias="SERVICE_NAME")
    service_version: str = Field("0.2.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    experiments_store_path: str = Field(
        "/tmp/orion-self-experiments/experiments.sqlite3",
        alias="EXPERIMENTS_STORE_PATH",
    )
    experiments_allow_non_read_only: bool = Field(False, alias="EXPERIMENTS_ALLOW_NON_READ_ONLY")
    port: int = Field(7172, alias="PORT")

    orion_bus_url: str = Field("redis://127.0.0.1:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(False, alias="ORION_BUS_ENABLED")
    # Bus-native SystemHealthV1 heartbeat cadence (orion:system:health). See
    # docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")

    class Config:
        env_file = ".env"
        extra = "ignore"
        populate_by_name = True


settings = Settings()
