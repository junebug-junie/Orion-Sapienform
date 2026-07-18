from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """
    🏷️ Orion Meta-Tags Service
    Handles NLP tagging and metadata enrichment for Collapse Mirror events.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # === Core Identity ===
    SERVICE_NAME: str = Field(default="meta-tags", env="SERVICE_NAME")
    SERVICE_VERSION: str = Field(default="0.2.0", env="SERVICE_VERSION")
    NODE_NAME: str = Field(default="unknown")
    PORT: int = Field(default=8201, env="PORT")

    # === Orion Bus Configuration ===
    ORION_BUS_URL: str = Field(default="redis://orion-janus-bus-core:6379/0", env="ORION_BUS_URL")
    ORION_BUS_ENABLED: bool = Field(default=True, env="ORION_BUS_ENABLED")
    ORION_BUS_ENFORCE_CATALOG: bool = Field(default=False, env="ORION_BUS_ENFORCE_CATALOG")

    # === Channel Routing ===
    CHANNEL_EVENTS_TRIAGE: str = Field(default="orion:collapse:triage", env="CHANNEL_EVENTS_TRIAGE")
    CHANNEL_EVENTS_TAGGED: str = Field(default="orion:tags:enriched", env="CHANNEL_EVENTS_TAGGED")
    CHANNEL_EVENTS_CHAT_TURN: str = Field(default="orion:chat:history:turn", env="CHANNEL_EVENTS_CHAT_TURN")
    CHANNEL_EVENTS_TAGGED_CHAT: str = Field(default="orion:tags:chat:enriched", env="CHANNEL_EVENTS_TAGGED_CHAT")

    # === NLP Model ===
    SPA_MODEL: str = Field(default="en_core_web_trf", env="SPA_MODEL")

    # === Recall Falkor writer (Phase 2, dark by default) ===
    # docs/superpowers/plans/2026-07-18-recall-tag-entity-falkor-writer-plan.md
    # Cypher-native, additive write of chat-turn tag/entity enrichment into
    # FalkorDB, alongside (not replacing) orion-rdf-writer's existing Fuseki
    # write of the same tags.enriched event. Off by default.
    RECALL_FALKOR_TAG_ENTITY_ENABLED: bool = Field(default=False, env="RECALL_FALKOR_TAG_ENTITY_ENABLED")
    FALKORDB_URI: str = Field(default="redis://orion-athena-falkordb:6379", env="FALKORDB_URI")
    FALKORDB_RECALL_GRAPH: str = Field(default="orion_recall", env="FALKORDB_RECALL_GRAPH")

    # === Runtime ===
    STARTUP_DELAY: int = Field(default=5, env="STARTUP_DELAY")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    # Chassis Defaults
    HEARTBEAT_INTERVAL_SEC: float = 10.0
    ORION_HEALTH_CHANNEL: str = "orion:system:health"
    ERROR_CHANNEL: str = "orion:system:error"
    SHUTDOWN_GRACE_SEC: float = 10.0

settings = Settings()
