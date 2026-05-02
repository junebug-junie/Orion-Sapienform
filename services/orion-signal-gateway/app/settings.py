import json
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Span attributes use dim.{key}; only these keys are exported by default (phase-2 PII / cardinality).
_DEFAULT_OTEL_DIMENSION_ALLOWLIST: tuple[str, ...] = (
    "level",
    "trend",
    "volatility",
    "valence",
    "confidence",
    "arousal",
    "coherence",
    "novelty",
    "salience",
    "tension",
    "goal_pressure",
    "pressure_coherence",
    "pressure_continuity",
    "pressure_relational",
    "pressure_autonomy",
    "pressure_capability",
    "pressure_predictive",
)


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
    # JSON array of dimension keys allowed as span attributes dim.* (never summary/notes on spans).
    OTEL_DIMENSION_ALLOWLIST: List[str] = Field(default_factory=lambda: list(_DEFAULT_OTEL_DIMENSION_ALLOWLIST))

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

    @field_validator("OTEL_DIMENSION_ALLOWLIST", mode="before")
    @classmethod
    def _parse_otel_dimension_allowlist(cls, value: object) -> List[str]:
        if isinstance(value, list):
            return [str(x) for x in value]
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except json.JSONDecodeError:
                pass
        return value  # type: ignore[return-value]


settings = Settings()
