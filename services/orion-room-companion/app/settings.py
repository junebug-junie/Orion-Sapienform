from __future__ import annotations

import os
from functools import lru_cache

from typing import Optional

from pydantic import Field, SecretStr, field_validator
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
    # Authenticates as the operator's own Claude subscription. NEVER an
    # ANTHROPIC_API_KEY -- that would open a second, pay-per-token billing
    # relationship instead of reusing the subscription, and is guarded by a
    # regression test.
    #
    # This service exists as a separate container precisely so this credential
    # is NOT in orion-hub, which runs as root with SSH keys, a gh token and
    # the docker socket all readable by Orion's own FCC turns. In v2 Orion
    # triggers these calls; a credential it can read is one it can use to
    # spawn Claude outside the meter.
    ROOM_COMPANION_CLAUDE_CONFIG_DIR: str = Field(default="/root/.claude")

    # Long-lived (1-year) OAuth token from `claude setup-token`, generated
    # interactively and pasted straight into the operator's local .env --
    # never through this repo or through chat with an agent. Replaced the
    # `.credentials.json` file bind mount 2026-08-18: that file goes stale
    # every ~7.5h token refresh (rename -> new inode; a file bind stays
    # pinned to the old one, so the container keeps reading a server-revoked
    # token -- confirmed live 2026-08-14, third recurrence of this exact
    # failure across the fleet). A token has no inode to go stale, and unlike
    # the file (where a host-side `mv` does NOT cut the container off -- the
    # bind mount holds the old inode) this one has a real kill switch:
    # revoke it at https://claude.ai/settings/claude-code. `docker stop`
    # still works too and is faster.
    #
    # This is explicitly NOT passed through as the bare `CLAUDE_CODE_OAUTH_TOKEN`
    # env var anywhere in this container's own environment -- see
    # build_subprocess_env in main.py, which injects it under its real name
    # only into the `claude` subprocess's env, never this service's own
    # os.environ. `_ENV_DENY_PREFIXES` in main.py still strips any bare
    # CLAUDE_CODE_OAUTH_TOKEN that shows up in ambient os.environ (e.g. a
    # copy-paste mistake, an inherited shell var) -- only the value that
    # arrives through this specific Settings field, sourced from this
    # service's own .env, ever reaches the subprocess.
    #
    # SecretStr, not str: this is the first real secret this Settings class
    # has ever held (the old design mounted a file and never put credential
    # bytes into a Python object at all). Without it, any future
    # `logger.debug(settings)` or a debug endpoint's `model_dump()` would
    # render the live token in full.
    ROOM_COMPANION_CLAUDE_OAUTH_TOKEN: Optional[SecretStr] = Field(default=None)

    @field_validator("ROOM_COMPANION_CLAUDE_OAUTH_TOKEN", mode="before")
    @classmethod
    def _strip_pasted_token(cls, value: object) -> object:
        """Documented workflow is a manual terminal copy-paste; a trailing
        newline or space riding along would make the token silently wrong
        (truthy, present, and still rejected by Claude Code) and misread as a
        revoked credential rather than a paste artifact."""
        return value.strip() if isinstance(value, str) else value

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
