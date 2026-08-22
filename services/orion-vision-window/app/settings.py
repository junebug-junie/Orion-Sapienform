from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = "vision-window"
    SERVICE_VERSION: str = "0.1.0"
    NODE_NAME: str = "athena"
    LOG_LEVEL: str = "INFO"

    # Bus
    ORION_BUS_URL: str = "redis://localhost:6379/0"
    ORION_BUS_ENFORCE_CATALOG: bool = False
    # Bus-native SystemHealthV1 heartbeat cadence (orion:system:health). See
    # docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
    HEARTBEAT_INTERVAL_SEC: float = 10.0

    # Channels
    CHANNEL_WINDOW_INTAKE: str = "orion:vision:artifacts"
    CHANNEL_WINDOW_PUB: str = "orion:vision:windows"
    CHANNEL_WINDOW_REQUEST: str = "orion:exec:request:VisionWindowService"

    # Rolling window (legacy WINDOW_SIZE_SEC kept as max wall clock span for a batch)
    WINDOW_SIZE_SEC: float = 30.0
    FLUSH_INTERVAL_MS: int = 5_000
    MAX_ARTIFACTS_PER_WINDOW: int = 64
    MAX_WINDOW_AGE_MS: int = 60_000
    STALE_AFTER_MS: int = 120_000

    # Scene belief habituation (per-stream; ephemeral like live_state)
    WINDOW_BELIEF_ENABLED: bool = True
    WINDOW_BELIEF_VOTE_N: int = 3
    WINDOW_BELIEF_ENTER_VOTES: int = 3
    WINDOW_BELIEF_EXIT_VOTES: int = 0

    # Embodied presence -- see app/presence.py. Layered on believed_labels
    # (already vote-smoothed by scene belief above), not on raw detections.
    WINDOW_PRESENCE_ENABLED: bool = True
    # How long "not seen just now" still reads as "recent" rather than
    # "absent" -- covers a bathroom break without flapping the state.
    WINDOW_PRESENCE_GRACE_SEC: float = 120.0
    WINDOW_PRESENCE_WRITE_MIN_INTERVAL_SEC: float = 5.0
    POSTGRES_URI: str = ""

    # Per-window scene census -> orion-sql-writer -> vision_scene_inventory.
    # Written on every window because the council only emits an event on a
    # label-SET change, so counts and departures are invisible in the event
    # stream. See main.py::_publish_scene_inventory.
    WINDOW_SCENE_INVENTORY_ENABLED: bool = True
    CHANNEL_SCENE_INVENTORY_PUB: str = "orion:vision:inventory:sql-write"

    # HTTP
    HTTP_HOST: str = "0.0.0.0"
    HTTP_PORT: int = 8000

    # Bounded recovery (§4.3) — same Redis URL as bus by default; dedicated URL optional
    VISION_WINDOW_RECOVERY_ENABLED: bool = True
    VISION_WINDOW_RECOVERY_REDIS_URL: str = ""
    VISION_WINDOW_RECOVERY_TTL_SEC: int = 3_600
    VISION_WINDOW_RECOVERY_MAX_N: int = 50
    VISION_WINDOW_HTTP_MAX_LIMIT: int = 50
    VISION_WINDOW_READY_REQUIRES_RECOVERY: bool = False
