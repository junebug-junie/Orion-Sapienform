from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = "sql-writer"
    SERVICE_VERSION: str = "0.3.0"
    PORT: int = 8220

    # --- Bus ---
    ORION_BUS_ENABLED: bool = True
    ORION_BUS_URL: str = "redis://orion-janus-bus-core:6379/0"

    CHANNEL_TAGS_RAW: str = "orion:tags"
    CHANNEL_TAGS_ENRICHED: str = "orion:tags:enriched"

    CHANNEL_COLLAPSE_TRIAGE: str = "orion:collapse:triage"
    CHANNEL_COLLAPSE_PUBLISH: str = "orion:collapse:sql-write"

    # --- DB ---
    POSTGRES_URI: str = "postgresql://postgres:postgres@orion-janus-sql-db:5432/conjourney"
    POSTGRES_TABLE: str = "collapse_enrichment"

    # --- Channel → Table map ---
    BUS_TABLE_MAP: str = (
       "orion:collapse:triage:collapse_mirror",
       "orion:tags:raw:collapse_tags_raw",
       "orion:tags:enriched:collapse_enrichment",
       "orion:chat:history:log:chat_history",
       "orion:rag:document:add:rag_documents"
    )

    POLL_TIMEOUT: float = 1.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def get_table_for_channel(self, channel: str) -> str:
        """Parses the BUS_TABLE_MAP to find the target table for a given channel."""
        mapping = {}
        for pair in (self.BUS_TABLE_MAP or "").split(","):
            if ":" in pair:
                parts = pair.split(":")
                if len(parts) >= 2:
                    # Handle channel names that might contain colons
                    channel_name = ":".join(parts[:-1])
                    table_name = parts[-1]
                    mapping[channel_name] = table_name
        return mapping.get(channel, self.POSTGRES_TABLE)

    def get_all_subscribe_channels(self) -> list[str]:
        """Returns a list of all channels this service should subscribe to."""
        return [
            self.CHANNEL_TAGS_RAW,
            self.CHANNEL_TAGS_ENRICHED,
            self.CHANNEL_COLLAPSE_TRIAGE,
            self.CHANNEL_COLLAPSE_PUBLISH
        ]


settings = Settings()
