from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Service identity ──────────────────────────────────────────────
    SERVICE_NAME: str = Field(default="self-study-enrichment")
    SERVICE_VERSION: str = Field(default="0.1.0")
    SELF_STUDY_ENRICHMENT_NODE_NAME: str = Field(default_factory=lambda: os.uname().nodename)

    # ── Orion Bus config ──────────────────────────────────────────────
    ORION_BUS_ENABLED: bool = Field(default=True)
    ORION_BUS_ENFORCE_CATALOG: bool = Field(default=False)
    ORION_BUS_URL: str = Field(default="redis://100.92.216.81:6379/0")
    HEARTBEAT_INTERVAL_SEC: float = Field(default=10.0)

    CHANNEL_SELF_STUDY_ENRICHMENT_REQUESTED: str = Field(default="orion:self_study:enrichment:requested")

    # ── Repo access (evidence bundle assembly) ─────────────────────────
    # Read-only mount of this repo's own real working tree -- same pattern
    # as orion-cocreation-signals' COCREATION_SIGNALS_REPO_PATH, for the
    # same reason: needs a live .git checkout + graphify-out/, not a stale
    # image-baked copy.
    SELF_STUDY_ENRICHMENT_REPO_PATH: str = Field(default="/repo")

    # ── Claude subprocess credential (2026-08-21: switched from a
    # file-mount of the host's `.credentials.json` to a `claude setup-token`
    # long-lived OAuth token, same fix and same rationale as
    # orion-room-companion's identical switch on 2026-08-18 -- see that
    # service's README "FIXED 2026-08-18" section for the incident this
    # retires: the file-mount credential silently went stale on Claude
    # Code's ~7.5h internal refresh cycle, since nothing re-wrote the
    # bind-mounted file inside the container. Still authenticates against
    # the operator's own Claude subscription, NOT a separate
    # ANTHROPIC_API_KEY pay-per-token billing path -- see README.md's
    # "Credential isolation" section. `CLAUDE_CONFIG_DIR` is still set (see
    # main.py) so the one-shot `claude -p` subprocess has a config dir to
    # write into, but it no longer needs to be a persistent/mounted
    # directory -- there is nothing left in it that must survive between
    # calls, since the OAuth token is supplied fresh from Settings every
    # time, not read from a file Claude Code expects to find there.
    # SecretStr so the value never appears in a repr/log by accident; the
    # validator strips whitespace a manual paste of `claude setup-token`'s
    # output commonly introduces.) ──────────────────────────────────────
    SELF_STUDY_ENRICHMENT_CLAUDE_OAUTH_TOKEN: Optional[SecretStr] = Field(default=None)

    @field_validator("SELF_STUDY_ENRICHMENT_CLAUDE_OAUTH_TOKEN", mode="before")
    @classmethod
    def _strip_pasted_token(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    SELF_STUDY_ENRICHMENT_CLAUDE_CONFIG_DIR: str = Field(default="/root/.claude")
    SELF_STUDY_ENRICHMENT_CLAUDE_BIN: str = Field(default="claude")
    SELF_STUDY_ENRICHMENT_MODEL: str = Field(default="claude-sonnet-5")
    SELF_STUDY_ENRICHMENT_EFFORT: str = Field(default="medium")
    SELF_STUDY_ENRICHMENT_TIMEOUT_SEC: float = Field(default=120.0)
    SELF_STUDY_ENRICHMENT_SETTING_SOURCES_ENV_KEY: str = Field(default="SELF_STUDY_ENRICHMENT_SETTING_SOURCES")
    SELF_STUDY_ENRICHMENT_SETTING_SOURCES: str = Field(default="user,local")

    # ── Cache (content-hash-keyed, gitignored -- mirrors graphify-out/cache/semantic/
    # in *shape*, but deliberately lives on a separate writable volume, not
    # inside the read-only repo mount: SELF_STUDY_ENRICHMENT_REPO_PATH is
    # mounted `:ro` in docker-compose.yml, same as orion-cocreation-signals'
    # repo mount, and this service needs to write cache entries + rate-limit
    # state) ──────────────────────────────────────────────────────────────
    SELF_STUDY_ENRICHMENT_CACHE_DIR: str = Field(default="/data/cache/self_study_enrichment")

    # ── Safety backstop: max real enrichment runs per day ──────────────
    SELF_STUDY_ENRICHMENT_MAX_PER_DAY: int = Field(default=8)
    SELF_STUDY_ENRICHMENT_RATE_LIMIT_STATE_PATH: str = Field(
        default="/data/state/self_study_enrichment_service_rate_limit.json"
    )

    # ── Graphify evidence source ────────────────────────────────────────
    SELF_STUDY_ENRICHMENT_GRAPH_JSON_PATH: str = Field(default="/repo/graphify-out/graph.json")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
