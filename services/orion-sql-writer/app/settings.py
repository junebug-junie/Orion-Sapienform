from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Service identity
    SERVICE_NAME: str = "orion-sql-writer"
    SERVICE_VERSION: str = "0.3.0"
    PORT: int = 8220

    # Bus
    ORION_BUS_ENABLED: bool = True
    ORION_BUS_URL: str = "redis://orion-janus-bus-core:6379/0"
    # Comma-separated list
    SUBSCRIBE_CHANNELS: str = "orion.tags.enriched,collapse.intake"

    # DB
    POSTGRES_URI: str = "postgresql://postgres:postgres@orion-janus-sql-db:5432/conjourney"
    # Default/fallback table if a channel isn't mapped below
    POSTGRES_TABLE: str = "collapse_enrichment"

    # Map channels → tables (comma-separated pairs: "<channel>:<table>")
    # You can extend this without code changes (e.g., collapse.events.raw:collapse_events)
    BUS_TABLE_MAP: str = (
        "orion.tags.enriched:collapse_enrichment,"
        "orion.tags:collapse_enrichment,"
        "collapse.intake:collapse_mirror"
    )

    # Misc
    POLL_TIMEOUT: float = 1.0

    class Config:
        # The container's CWD is /app; compose passes envs in. This also lets you run locally.
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    # Helpers
    def get_table_for_channel(self, channel: str) -> str:
        mapping = {}
        for pair in (self.BUS_TABLE_MAP or "").split(","):
            if ":" in pair:
                ch, tbl = pair.split(":", 1)
                mapping[ch.strip()] = tbl.strip()
        return mapping.get(channel, self.POSTGRES_TABLE)


settings = Settings()
