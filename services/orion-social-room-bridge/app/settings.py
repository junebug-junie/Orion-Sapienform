from __future__ import annotations

from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from orion.schemas.social_autonomy import SocialAutonomyMode


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = Field("orion-social-room-bridge", alias="APP_NAME")
    service_name: str = Field("orion-social-room-bridge", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("unknown", alias="NODE_NAME")
    port: int = Field(8764, alias="PORT")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    orion_bus_url: str = Field("redis://orion-redis:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_enforce_catalog: bool = Field(False, alias="ORION_BUS_ENFORCE_CATALOG")

    social_bridge_enabled: bool = Field(True, alias="SOCIAL_BRIDGE_ENABLED")
    social_bridge_dry_run: bool = Field(False, alias="SOCIAL_BRIDGE_DRY_RUN")
    social_bridge_platform: str = Field("callsyne", alias="SOCIAL_BRIDGE_PLATFORM")
    social_bridge_use_recall: bool = Field(True, alias="SOCIAL_BRIDGE_USE_RECALL")
    social_bridge_room_allowlist: List[str] | str = Field(default_factory=list, alias="SOCIAL_BRIDGE_ROOM_ALLOWLIST")
    social_bridge_self_participant_ids: List[str] | str = Field(default_factory=list, alias="SOCIAL_BRIDGE_SELF_PARTICIPANT_IDS")
    social_bridge_self_name: str = Field("Oríon", alias="SOCIAL_BRIDGE_SELF_NAME")
    social_bridge_autonomy_mode: SocialAutonomyMode = Field("responsive", alias="SOCIAL_BRIDGE_AUTONOMY_MODE")
    social_bridge_only_when_addressed: bool = Field(False, alias="SOCIAL_BRIDGE_ONLY_WHEN_ADDRESSED")
    social_bridge_cooldown_sec: float = Field(0.0, alias="SOCIAL_BRIDGE_COOLDOWN_SEC")
    social_bridge_max_consecutive_orion_turns: int = Field(2, alias="SOCIAL_BRIDGE_MAX_CONSECUTIVE_ORION_TURNS")
    social_bridge_min_novelty_score: float = Field(0.22, alias="SOCIAL_BRIDGE_MIN_NOVELTY_SCORE")
    social_bridge_light_initiative_min_continuity: float = Field(
        0.45,
        alias="SOCIAL_BRIDGE_LIGHT_INITIATIVE_MIN_CONTINUITY",
    )
    social_bridge_dedupe_ttl_sec: int = Field(3600, alias="SOCIAL_BRIDGE_DEDUPE_TTL_SEC")
    social_bridge_session_namespace: str = Field("callsyne-room", alias="SOCIAL_BRIDGE_SESSION_NAMESPACE")
    social_bridge_hub_mode: str = Field("brain", alias="SOCIAL_BRIDGE_HUB_MODE")
    social_bridge_hub_verb: str = Field("", alias="SOCIAL_BRIDGE_HUB_VERB")
    social_bridge_return_2xx_on_delivery_failure: bool = Field(
        True,
        alias="SOCIAL_BRIDGE_RETURN_2XX_ON_DELIVERY_FAILURE",
    )
    social_bridge_callsyne_poll_enabled: bool = Field(False, alias="SOCIAL_BRIDGE_CALLSYNE_POLL_ENABLED")
    social_bridge_callsyne_poll_room_id: str = Field("world", alias="SOCIAL_BRIDGE_CALLSYNE_POLL_ROOM_ID")
    social_bridge_callsyne_poll_interval_sec: float = Field(2.0, alias="SOCIAL_BRIDGE_CALLSYNE_POLL_INTERVAL_SEC")
    social_bridge_callsyne_poll_since_message_id: str = Field("", alias="SOCIAL_BRIDGE_CALLSYNE_POLL_SINCE_MESSAGE_ID")
    social_bridge_callsyne_poll_skip_self: bool = Field(True, alias="SOCIAL_BRIDGE_CALLSYNE_POLL_SKIP_SELF")
    social_bridge_callsyne_poll_limit: int = Field(20, alias="SOCIAL_BRIDGE_CALLSYNE_POLL_LIMIT")
    social_bridge_callsyne_poll_path: str = Field("/api/bridge/messages", alias="SOCIAL_BRIDGE_CALLSYNE_POLL_PATH")

    hub_base_url: str = Field("http://orion-hub:8080", alias="HUB_BASE_URL")
    hub_chat_path: str = Field("/api/chat", alias="HUB_CHAT_PATH")
    hub_timeout_sec: float = Field(120.0, alias="HUB_TIMEOUT_SEC")

    callsyne_base_url: str = Field("https://api.callsyne.com", alias="CALLSYNE_BASE_URL")
    callsyne_api_token: str = Field("", alias="CALLSYNE_API_TOKEN")
    callsyne_timeout_sec: float = Field(30.0, alias="CALLSYNE_TIMEOUT_SEC")
    callsyne_post_path_template: str = Field(
        "/api/bridge/messages",
        alias="CALLSYNE_POST_PATH_TEMPLATE",
    )
    callsyne_webhook_token: str = Field("", alias="CALLSYNE_WEBHOOK_TOKEN")

    social_memory_base_url: str = Field("http://orion-social-memory:8765", alias="SOCIAL_MEMORY_BASE_URL")
    social_memory_timeout_sec: float = Field(3.0, alias="SOCIAL_MEMORY_TIMEOUT_SEC")

    room_intake_channel: str = Field("orion:bridge:social:room:intake", alias="ROOM_INTAKE_CHANNEL")
    room_delivery_channel: str = Field("orion:bridge:social:room:delivery", alias="ROOM_DELIVERY_CHANNEL")
    room_skipped_channel: str = Field("orion:bridge:social:room:skipped", alias="ROOM_SKIPPED_CHANNEL")
    room_participant_channel: str = Field("orion:bridge:social:participant", alias="ROOM_PARTICIPANT_CHANNEL")
    room_repair_signal_channel: str = Field("orion:social:repair:signal", alias="ROOM_REPAIR_SIGNAL_CHANNEL")
    room_repair_decision_channel: str = Field("orion:social:repair:decision", alias="ROOM_REPAIR_DECISION_CHANNEL")
    room_epistemic_signal_channel: str = Field("orion:social:epistemic:signal", alias="ROOM_EPISTEMIC_SIGNAL_CHANNEL")
    room_epistemic_decision_channel: str = Field("orion:social:epistemic:decision", alias="ROOM_EPISTEMIC_DECISION_CHANNEL")
    room_turn_policy_channel: str = Field("orion:social:turn-policy", alias="ROOM_TURN_POLICY_CHANNEL")

    @field_validator("social_bridge_room_allowlist", "social_bridge_self_participant_ids", mode="before")
    @classmethod
    def _split_csv(cls, value: List[str] | str | None) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [part.strip() for part in str(value).split(",") if part.strip()]

    @field_validator("social_bridge_autonomy_mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: SocialAutonomyMode | str | None) -> SocialAutonomyMode:
        normalized = str(value or "responsive").strip().lower().replace("-", "_")
        if normalized not in {"addressed_only", "responsive", "light_initiative"}:
            raise ValueError(f"Unsupported SOCIAL_BRIDGE_AUTONOMY_MODE: {value}")
        return normalized  # type: ignore[return-value]

    @field_validator("social_bridge_min_novelty_score", "social_bridge_light_initiative_min_continuity")
    @classmethod
    def _clamp_ratio(cls, value: float) -> float:
        number = float(value)
        if number < 0.0 or number > 1.0:
            raise ValueError("policy thresholds must be between 0.0 and 1.0")
        return number


settings = Settings()
