from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    service_name: str = Field("orion-mesh-guardian", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    project: str = Field("orion", alias="PROJECT")
    node_name: str = Field("unknown", alias="NODE_NAME")
    orion_repo_root: str = Field("/repo", alias="ORION_REPO_ROOT")
    orion_bus_url: str = Field("redis://bus-core:6379/0", alias="ORION_BUS_URL")
    notify_base_url: str = Field("http://orion-notify:7140", alias="NOTIFY_BASE_URL")
    notify_api_token: str | None = Field(None, alias="NOTIFY_API_TOKEN")
    roster_path: str = Field("/repo/config/mesh_remediation_roster.yaml", alias="MESH_GUARDIAN_ROSTER_PATH")
    enabled: bool = Field(True, alias="MESH_GUARDIAN_ENABLED")
    auto_remediate: bool = Field(False, alias="MESH_GUARDIAN_AUTO_REMEDIATE")
    remediation_cooldown_sec: int = Field(300, alias="MESH_GUARDIAN_REMEDIATION_COOLDOWN_SEC")
    max_attempts_per_hour: int = Field(3, alias="MESH_GUARDIAN_MAX_ATTEMPTS_PER_HOUR")
    probe_interval_sec: int = Field(15, alias="MESH_GUARDIAN_PROBE_INTERVAL_SEC")
    post_remediate_grace_sec: int = Field(60, alias="MESH_GUARDIAN_POST_REMEDIATE_GRACE_SEC")
    consecutive_probe_fails: int = Field(2, alias="MESH_GUARDIAN_CONSECUTIVE_PROBE_FAILS")
    equilibrium_grace_sec: int = Field(30, alias="MESH_GUARDIAN_EQUILIBRIUM_GRACE_SEC")
    channel_equilibrium_snapshot: str = Field("orion:equilibrium:snapshot", alias="CHANNEL_EQUILIBRIUM_SNAPSHOT")
    health_http_port: int = Field(7161, alias="PORT")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
