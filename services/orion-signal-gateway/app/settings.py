import json
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = "orion-signal-gateway"
    SERVICE_VERSION: str = "0.1.0"
    NODE_NAME: str = "unknown"

    ORION_BUS_URL: str = "redis://orion-redis:6379/0"
    ORION_BUS_ENABLED: bool = True

    LOG_LEVEL: str = "INFO"
    HEARTBEAT_INTERVAL_SEC: float = 10.0
    ORION_HEALTH_CHANNEL: str = "orion:system:health"
    ERROR_CHANNEL: str = "orion:system:error"
    SHUTDOWN_GRACE_SEC: float = 10.0

    # OpenTelemetry (spec §5): OTLP gRPC endpoint, e.g. "http://otel-collector:4317"
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = None
    OTEL_CONSOLE_EXPORT: bool = False

    SIGNALS_OUTPUT_CHANNEL: str = "orion:signals"
    SIGNALS_PASSTHROUGH_PATTERN: str = "orion:signals:*"
    SIGNAL_WINDOW_SEC: float = 30.0

    ORGAN_CHANNELS: List[str] = [
        "orion:biometrics:*",
        "orion:equilibrium:*",
        "orion:collapse:*",
        "orion:recall:*",
        "orion:autonomy:*",
        "orion:signals:*",
        "orion:spark:*",
        "orion:memory:*",
        "orion:social:*",
        "orion:vision:*",
        "orion:agent:*",
        "orion:planner:*",
        "orion:dream:*",
        "orion:state_journaler:*",
        "orion:topic:*",
        "orion:concept:*",
        "orion:cognition:*",
        "orion:stance:*",
        "orion:journal:*",
        "orion:power:*",
        "orion:security:*",
        "orion:world:*",
    ]

    @field_validator("ORGAN_CHANNELS", mode="before")
    @classmethod
    def _parse_organ_channels(cls, value: object) -> List[str]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        return value  # type: ignore[return-value]


settings = Settings()
