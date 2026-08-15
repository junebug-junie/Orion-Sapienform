from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Service identity ──────────────────────────────────────────────
    SERVICE_NAME: str = Field(default="room-companion")
    SERVICE_VERSION: str = Field(default="0.1.0")
    ROOM_COMPANION_NODE_NAME: str = Field(default_factory=lambda: os.uname().nodename)

    # ── Orion Bus config ──────────────────────────────────────────────
    ORION_BUS_ENABLED: bool = Field(default=True)
    ORION_BUS_ENFORCE_CATALOG: bool = Field(default=False)
    ORION_BUS_URL: str = Field(default="redis://100.92.216.81:6379/0")
    HEARTBEAT_INTERVAL_SEC: float = Field(default=10.0)

    CHANNEL_ROOM_CLAUDE_REQUEST: str = Field(default="orion:room:claude:request")
    CHANNEL_ROOM_CLAUDE_UTTERANCE: str = Field(default="orion:room:claude:utterance")

    # ── Claude subprocess ─────────────────────────────────────────────
    # Authenticates as the operator's already-logged-in Claude Code session,
    # exactly as orion-self-study-enrichment does: CLAUDE_CONFIG_DIR points at
    # a container-local dir holding nothing but a read-only bind mount of the
    # host's real `.credentials.json`. NEVER an ANTHROPIC_API_KEY -- that
    # would open a second, pay-per-token billing relationship instead of
    # reusing the subscription, and is guarded by a regression test.
    #
    # This service exists as a separate container precisely so this credential
    # is NOT in orion-hub, which runs as root with SSH keys, a gh token and
    # the docker socket all readable by Orion's own FCC turns. In v2 Orion
    # triggers these calls; a credential it can read is one it can use to
    # spawn Claude outside the meter.
    ROOM_COMPANION_CLAUDE_CONFIG_DIR: str = Field(default="/root/.claude")
    ROOM_COMPANION_CLAUDE_BIN: str = Field(default="claude")
    ROOM_COMPANION_MODEL: str = Field(default="claude-sonnet-5")
    ROOM_COMPANION_EFFORT: str = Field(default="medium")
    ROOM_COMPANION_TIMEOUT_SEC: float = Field(default=180.0)
    ROOM_COMPANION_SETTING_SOURCES_ENV_KEY: str = Field(default="ROOM_COMPANION_SETTING_SOURCES")
    ROOM_COMPANION_SETTING_SOURCES: str = Field(default="user")

    # `--tools ""` means the subprocess never reads a file, but Claude Code
    # still resolves CLAUDE.md/settings from its cwd. Point it at a neutral
    # directory so a room conversation cannot inherit this repo's development
    # contract as if it were room context.
    ROOM_COMPANION_WORKSPACE: str = Field(default="/data/workspace")

    # ── Room participant identity ─────────────────────────────────────
    ROOM_COMPANION_PARTICIPANT_ID: str = Field(default="claude")
    ROOM_COMPANION_PARTICIPANT_NAME: str = Field(default="Claude")

    # ── Session store ─────────────────────────────────────────────────
    # room_id -> claude session uuid. One room is one durable Claude session,
    # so Claude remembers the conversation without the transcript being
    # re-sent every turn. On a writable volume, not the (absent) repo mount.
    ROOM_COMPANION_SESSION_STATE_PATH: str = Field(
        default="/data/state/room_companion_sessions.json"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
