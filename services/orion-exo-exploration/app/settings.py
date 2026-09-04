from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    service_name: str = Field("orion-exo-exploration", validation_alias=AliasChoices("SERVICE_NAME"))
    service_version: str = Field("0.1.0", validation_alias=AliasChoices("SERVICE_VERSION"))
    node_name: str = Field("unknown", validation_alias=AliasChoices("NODE_NAME", "HOSTNAME"))
    log_level: str = Field("INFO", validation_alias=AliasChoices("LOG_LEVEL"))
    port: int = Field(8622, validation_alias=AliasChoices("PORT"))

    exo_exploration_pg_dsn: str = Field(
        "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney",
        validation_alias=AliasChoices("EXO_EXPLORATION_PG_DSN", "POSTGRES_URI", "POSTGRES_DSN"),
    )

    # --- Crawl behavior ---------------------------------------------------
    # Seconds between the start of one full crawl (all categories) and the
    # next. Default once/day, matching the design doc.
    exo_exploration_crawl_interval_seconds: int = Field(
        86400, validation_alias=AliasChoices("EXO_EXPLORATION_CRAWL_INTERVAL_SECONDS")
    )
    # How long the retention sweep loop sleeps between passes. Deliberately
    # much shorter than the crawl interval -- a listing that fell out of
    # retention should not sit around for up to a day before it is swept.
    exo_exploration_retention_sweep_interval_seconds: int = Field(
        3600, validation_alias=AliasChoices("EXO_EXPLORATION_RETENTION_SWEEP_INTERVAL_SECONDS")
    )
    exo_exploration_retention_days: int = Field(
        14, validation_alias=AliasChoices("EXO_EXPLORATION_RETENTION_DAYS")
    )
    # Whether the daemon loops (crawl + retention sweep) run at all. Off by
    # default in a plain `pytest` import; docker-compose turns it on.
    exo_exploration_daemon_enabled: bool = Field(
        True, validation_alias=AliasChoices("EXO_EXPLORATION_DAEMON_ENABLED")
    )
    # Run one crawl immediately at process start rather than waiting a full
    # interval first -- see orion-hub's SUBSTRATE_TOPIC_FOUNDRY_SCHEDULER_RUN_AT_STARTUP
    # regression (a loop that sleeps its full interval before its first tick
    # needs 24 unbroken hours of uptime to ever fire once).
    exo_exploration_crawl_run_at_startup: bool = Field(
        True, validation_alias=AliasChoices("EXO_EXPLORATION_CRAWL_RUN_AT_STARTUP")
    )

    exo_exploration_categories: str = Field(
        (
            "https://classifieds.ksl.com/search/cat/Electronics,"
            "https://classifieds.ksl.com/search/cat/Computers,"
            "https://classifieds.ksl.com/search/cat/FREE"
        ),
        validation_alias=AliasChoices("EXO_EXPLORATION_CATEGORIES"),
    )
    exo_exploration_user_agent: str = Field(
        "orion-exo-exploration/0.1 (+operator-contact-tbd)",
        validation_alias=AliasChoices("EXO_EXPLORATION_USER_AGENT"),
    )
    exo_exploration_fetch_timeout_seconds: int = Field(
        20, validation_alias=AliasChoices("EXO_EXPLORATION_FETCH_TIMEOUT_SECONDS")
    )
    # Delay between consecutive HTTP requests to classifieds.ksl.com -- both
    # category pages and per-listing detail-page fetches. Polite-crawl floor,
    # not a performance knob.
    exo_exploration_request_delay_seconds: float = Field(
        1.5, validation_alias=AliasChoices("EXO_EXPLORATION_REQUEST_DELAY_SECONDS")
    )
    # Upper bound on how many category-page candidates get a detail-page
    # fetch per crawl run, across all categories combined. A candidate only
    # earns a detail fetch after it already passes the keyword/price filter
    # on its title -- see app/crawl/daemon.py -- so this caps worst case
    # request volume against KSL even on a category with an unusually high
    # tech-density hit rate.
    exo_exploration_max_detail_fetches_per_run: int = Field(
        60, validation_alias=AliasChoices("EXO_EXPLORATION_MAX_DETAIL_FETCHES_PER_RUN")
    )


settings = Settings()
