from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-embodiment", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")

    bus_url: str = Field(default="redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")

    enabled: bool = Field(False, alias="ORION_EMBODIMENT_ENABLED")
    fcc_env_path: str = Field("/root/.fcc/.env", alias="EMBODIMENT_FCC_ENV_PATH")
    # Optional override for the Convex URL loaded from ~/.fcc/.env. Needed when the
    # bridge container cannot reach the host-oriented URL (e.g. set to
    # http://host.docker.internal:3210). Empty = use the ~/.fcc/.env value as-is.
    aitown_convex_url: str = Field("", alias="EMBODIMENT_AITOWN_CONVEX_URL")
    deliberate_hold_sec: float = Field(8.0, alias="EMBODIMENT_DELIBERATE_HOLD_SEC")
    wander_radius: float = Field(3.0, alias="EMBODIMENT_WANDER_RADIUS")
    locations_json: str = Field("{}", alias="EMBODIMENT_LOCATIONS_JSON")
    idle_heartbeat_sec: float = Field(0.0, alias="EMBODIMENT_IDLE_HEARTBEAT_SEC")
    # Keep the AI Town engine awake so queued inputs (moveTo) are processed. An
    # `inactive` world silently drops all inputs. Off by default (keeping the town
    # alive also runs the other town agents' LLM loops).
    world_heartbeat_enabled: bool = Field(False, alias="EMBODIMENT_WORLD_HEARTBEAT_ENABLED")
    orion_sprite: str = Field("f1", alias="EMBODIMENT_ORION_SPRITE")
    self_state_url: str = Field("http://orion-self-state-runtime:8123", alias="EMBODIMENT_SELF_STATE_URL")
    perception_interval_sec: float = Field(0.0, alias="EMBODIMENT_PERCEPTION_INTERVAL_SEC")
    social_cooldown_sec: float = Field(120.0, alias="EMBODIMENT_SOCIAL_COOLDOWN_SEC")
    # Debounce competing move actuations (approach/wander/go_to) when multiple
    # producers are enabled at once. 0 = off (no debounce).
    move_cooldown_sec: float = Field(0.0, alias="EMBODIMENT_MOVE_COOLDOWN_SEC")

    speech_enabled: bool = Field(False, alias="EMBODIMENT_SPEECH_ENABLED")
    speech_lane: str = Field("quick", alias="EMBODIMENT_SPEECH_LANE")
    speech_verb: str = Field("chat_quick", alias="EMBODIMENT_SPEECH_VERB")
    speech_timeout_sec: float = Field(30.0, alias="EMBODIMENT_SPEECH_TIMEOUT_SEC")
    cortex_request_channel: str = Field("orion:cortex:exec:request", alias="EMBODIMENT_CORTEX_REQUEST_CHANNEL")
    cortex_result_prefix: str = Field("orion:exec:result", alias="EMBODIMENT_CORTEX_RESULT_PREFIX")
    memory_enabled: bool = Field(False, alias="EMBODIMENT_MEMORY_ENABLED")

    channel_intent: str = Field("orion:embodiment:intent", alias="EMBODIMENT_CHANNEL_INTENT")
    channel_outcome: str = Field("orion:embodiment:outcome", alias="EMBODIMENT_CHANNEL_OUTCOME")
    channel_perception: str = Field("orion:embodiment:perception", alias="EMBODIMENT_CHANNEL_PERCEPTION")

    port: int = Field(8130, alias="EMBODIMENT_PORT")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
