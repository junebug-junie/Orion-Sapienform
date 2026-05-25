from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-field-digester", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")

    postgres_uri: str = Field(..., alias="POSTGRES_URI")
    lattice_path: str = Field(
        "config/field/orion_field_topology.v1.yaml",
        alias="LATTICE_PATH",
    )
    receipt_poll_interval_sec: float = Field(2.0, alias="RECEIPT_POLL_INTERVAL_SEC")
    biometrics_field_decay_rate: float = Field(0.92, alias="BIOMETRICS_FIELD_DECAY_RATE")
    biometrics_field_diffusion_rate: float = Field(1.0, alias="BIOMETRICS_FIELD_DIFFUSION_RATE")
    log_level: str = Field("INFO", alias="LOG_LEVEL")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
