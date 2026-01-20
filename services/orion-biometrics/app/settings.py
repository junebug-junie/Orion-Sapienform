from pydantic_settings import BaseSettings, SettingsConfigDict
import json
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    SERVICE_NAME: str = Field(default="orion-biometrics")
    SERVICE_VERSION: str = Field(default="0.1.0")
    NODE_NAME: str = Field(default="unknown")

    # Bus
    ORION_BUS_URL: str = Field(default="redis://orion-redis:6379/0")
    ORION_BUS_ENABLED: bool = Field(default=True)
    ORION_BUS_ENFORCE_CATALOG: bool = Field(default=False)

    # Channels
    TELEMETRY_PUBLISH_CHANNEL: str = Field(default="orion:telemetry:biometrics")
    BIOMETRICS_SAMPLE_CHANNEL: str = Field(default="orion:biometrics:sample")
    BIOMETRICS_SUMMARY_CHANNEL: str = Field(default="orion:biometrics:summary")
    BIOMETRICS_INDUCTION_CHANNEL: str = Field(default="orion:biometrics:induction")
    BIOMETRICS_CLUSTER_CHANNEL: str = Field(default="orion:biometrics:cluster")
    SPARK_SIGNAL_CHANNEL: str = Field(default="orion:spark:signal")

    # Behavior
    TELEMETRY_INTERVAL: int = Field(default=30)
    LOG_LEVEL: str = Field(default="INFO")
    BIOMETRICS_MODE: str = Field(default="agent")
    CLUSTER_PUBLISH_INTERVAL: int = Field(default=15)
    CLUSTER_ROLE_WEIGHTS: str = Field(default=json.dumps({"atlas": 0.7, "athena": 0.3, "other": 0.5}))
    SPARK_SIGNAL_TTL_MS: int = Field(default=15000)

    THERMAL_MIN_C: float = Field(default=50.0)
    THERMAL_MAX_C: float = Field(default=85.0)
    DISK_BW_MBPS: float = Field(default=200.0)
    NET_BW_MBPS: float = Field(default=125.0)
    POWER_BAND_ALPHA: float = Field(default=0.1)

    TABLE_NAME: str = Field(default="biometrics_raw")

    # Chassis Defaults
    HEARTBEAT_INTERVAL_SEC: float = 10.0
    ORION_HEALTH_CHANNEL: str = "orion:system:health"
    ERROR_CHANNEL: str = "orion:system:error"
    SHUTDOWN_GRACE_SEC: float = 10.0

settings = Settings()

try:
    settings.role_weights = json.loads(settings.CLUSTER_ROLE_WEIGHTS)
except Exception:
    settings.role_weights = {"atlas": 0.7, "athena": 0.3, "other": 0.5}
