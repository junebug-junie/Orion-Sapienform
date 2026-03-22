from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = Field("orion-social-memory", alias="APP_NAME")
    service_name: str = Field("orion-social-memory", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("unknown", alias="NODE_NAME")
    port: int = Field(8765, alias="PORT")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    orion_bus_url: str = Field("redis://orion-redis:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_enforce_catalog: bool = Field(False, alias="ORION_BUS_ENFORCE_CATALOG")

    database_url: str = Field(
        "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney",
        alias="DATABASE_URL",
    )
    social_memory_input_channel: str = Field("orion:chat:social:stored", alias="SOCIAL_MEMORY_INPUT_CHANNEL")
    social_memory_participant_channel: str = Field(
        "orion:social:participant:continuity",
        alias="SOCIAL_MEMORY_PARTICIPANT_CHANNEL",
    )
    social_memory_room_channel: str = Field(
        "orion:social:room:continuity",
        alias="SOCIAL_MEMORY_ROOM_CHANNEL",
    )
    social_memory_stance_channel: str = Field(
        "orion:social:stance:snapshot",
        alias="SOCIAL_MEMORY_STANCE_CHANNEL",
    )
    social_memory_update_channel: str = Field(
        "orion:social:relational:update",
        alias="SOCIAL_MEMORY_UPDATE_CHANNEL",
    )
    social_memory_open_thread_channel: str = Field(
        "orion:social:open-thread",
        alias="SOCIAL_MEMORY_OPEN_THREAD_CHANNEL",
    )
    social_memory_peer_style_channel: str = Field(
        "orion:social:peer-style",
        alias="SOCIAL_MEMORY_PEER_STYLE_CHANNEL",
    )
    social_memory_room_ritual_channel: str = Field(
        "orion:social:room-ritual",
        alias="SOCIAL_MEMORY_ROOM_RITUAL_CHANNEL",
    )
    social_memory_commitment_channel: str = Field(
        "orion:social:commitment",
        alias="SOCIAL_MEMORY_COMMITMENT_CHANNEL",
    )
    social_memory_commitment_resolution_channel: str = Field(
        "orion:social:commitment:resolution",
        alias="SOCIAL_MEMORY_COMMITMENT_RESOLUTION_CHANNEL",
    )
    social_memory_claim_channel: str = Field(
        "orion:social:claim",
        alias="SOCIAL_MEMORY_CLAIM_CHANNEL",
    )
    social_memory_claim_revision_channel: str = Field(
        "orion:social:claim:revision",
        alias="SOCIAL_MEMORY_CLAIM_REVISION_CHANNEL",
    )
    social_memory_claim_stance_channel: str = Field(
        "orion:social:claim:stance",
        alias="SOCIAL_MEMORY_CLAIM_STANCE_CHANNEL",
    )
    social_memory_claim_attribution_channel: str = Field(
        "orion:social:claim:attribution",
        alias="SOCIAL_MEMORY_CLAIM_ATTRIBUTION_CHANNEL",
    )
    social_memory_claim_consensus_channel: str = Field(
        "orion:social:claim:consensus",
        alias="SOCIAL_MEMORY_CLAIM_CONSENSUS_CHANNEL",
    )
    social_memory_claim_divergence_channel: str = Field(
        "orion:social:claim:divergence",
        alias="SOCIAL_MEMORY_CLAIM_DIVERGENCE_CHANNEL",
    )
    social_memory_bridge_summary_channel: str = Field(
        "orion:social:bridge-summary",
        alias="SOCIAL_MEMORY_BRIDGE_SUMMARY_CHANNEL",
    )
    social_memory_clarifying_question_channel: str = Field(
        "orion:social:clarifying-question",
        alias="SOCIAL_MEMORY_CLARIFYING_QUESTION_CHANNEL",
    )
    social_memory_deliberation_decision_channel: str = Field(
        "orion:social:deliberation:decision",
        alias="SOCIAL_MEMORY_DELIBERATION_DECISION_CHANNEL",
    )
    social_memory_turn_handoff_channel: str = Field(
        "orion:social:turn-handoff",
        alias="SOCIAL_MEMORY_TURN_HANDOFF_CHANNEL",
    )
    social_memory_closure_signal_channel: str = Field(
        "orion:social:closure-signal",
        alias="SOCIAL_MEMORY_CLOSURE_SIGNAL_CHANNEL",
    )
    social_memory_floor_decision_channel: str = Field(
        "orion:social:floor:decision",
        alias="SOCIAL_MEMORY_FLOOR_DECISION_CHANNEL",
    )
    social_memory_evidence_max: int = Field(5, alias="SOCIAL_MEMORY_EVIDENCE_MAX")
    social_memory_topic_max: int = Field(6, alias="SOCIAL_MEMORY_TOPIC_MAX")
    social_memory_room_participant_max: int = Field(8, alias="SOCIAL_MEMORY_ROOM_PARTICIPANT_MAX")
    social_memory_open_thread_ttl_hours: int = Field(6, alias="SOCIAL_MEMORY_OPEN_THREAD_TTL_HOURS")
    social_memory_commitment_ttl_minutes: int = Field(90, alias="SOCIAL_MEMORY_COMMITMENT_TTL_MINUTES")
    social_memory_commitment_max_open: int = Field(3, alias="SOCIAL_MEMORY_COMMITMENT_MAX_OPEN")
    social_memory_style_adaptation_enabled: bool = Field(True, alias="SOCIAL_MEMORY_STYLE_ADAPTATION_ENABLED")
    social_memory_style_confidence_floor: float = Field(0.35, alias="SOCIAL_MEMORY_STYLE_CONFIDENCE_FLOOR")


settings = Settings()
