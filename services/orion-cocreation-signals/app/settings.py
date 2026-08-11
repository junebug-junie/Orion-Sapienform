from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Service identity ──────────────────────────────────────────────
    SERVICE_NAME: str = Field(default="cocreation-signals")
    SERVICE_VERSION: str = Field(default="0.1.0")
    COCREATION_SIGNALS_NODE_NAME: str = Field(default_factory=lambda: os.uname().nodename)

    # ── Orion Bus config ──────────────────────────────────────────────
    ORION_BUS_ENABLED: bool = Field(default=True)
    ORION_BUS_ENFORCE_CATALOG: bool = Field(default=False)
    ORION_BUS_URL: str = Field(default="redis://100.92.216.81:6379/0")
    HEARTBEAT_INTERVAL_SEC: float = Field(default=10.0)

    # ── Repo access (git_delta, graph_delta) ──────────────────────────
    # Read-only mount of this repo's own working tree -- see
    # docs/superpowers/specs/2026-07-30-codebase-mass-signal-design.md's
    # "Producer + consumer patch design" section for why a real .git checkout
    # is needed at runtime (not baked into the image at build time, which
    # would freeze git history at build time and defeat the point of tracking
    # live churn).
    COCREATION_SIGNALS_REPO_PATH: str = Field(default="/repo")

    # ── GitHub access (pr_lifecycle) ──────────────────────────────────
    # No minted GITHUB_TOKEN secret -- pr_lifecycle.py shells out to the `gh`
    # CLI (orion/structural_mass/pr_lifecycle.py), same as this repo's own
    # dev/CI environments already do. `gh` itself honors GH_TOKEN/GITHUB_TOKEN
    # env vars for auth with no `gh auth login` step required -- set one of
    # those directly in this service's .env, do not add a redundant
    # service-specific token field here.
    COCREATION_SIGNALS_GITHUB_OWNER: str = Field(default="junebug-junie")
    COCREATION_SIGNALS_GITHUB_REPO: str = Field(default="Orion-Sapienform")

    # ── Claude Code transcript access (affective_state) ────────────────
    # Read-only mount of Juniper's real local ~/.claude/projects tree --
    # deliberately the *whole* tree, not scoped to just this repo's own
    # sessions (Juniper's explicit call, 2026-08-11, overriding the narrower
    # scoping this producer's PR originally proposed as an open question --
    # see docs/superpowers/pr-reports/2026-08-11-juniper-affective-state-
    # signal-replay.md). MUST equal COCREATION_SIGNALS_CLAUDE_PROJECTS_HOST_PATH
    # exactly, not an arbitrary in-container path -- confirmed live 2026-08-11:
    # Claude Code stores cross-project subagent transcripts as absolute-path
    # symlinks back into ~/.claude/projects/..., which only resolve if the
    # mount lands at the identical path (see docker-compose.yml's own comment
    # on this mount). The default here only applies if the env var is fully
    # absent; .env_example sets it explicitly to match the real host path.
    COCREATION_SIGNALS_CLAUDE_PROJECTS_PATH: str = Field(default="/claude-projects")

    # ── Bus channels ───────────────────────────────────────────────────
    CHANNEL_CODEBASE_DELTA: str = Field(default="orion:substrate:codebase_delta")
    CHANNEL_JUNIPER_AFFECTIVE_STATE: str = Field(
        default="orion:substrate:juniper_affective_state"
    )

    # ── Producer enable flags (each independently toggleable -- a GitHub
    # API/rate-limit problem for pr_lifecycle must never block git_delta or
    # graph_delta) ──────────────────────────────────────────────────────
    COCREATION_SIGNALS_GIT_DELTA_ENABLED: bool = Field(default=True)
    COCREATION_SIGNALS_PR_LIFECYCLE_ENABLED: bool = Field(default=True)
    COCREATION_SIGNALS_GRAPH_DELTA_ENABLED: bool = Field(default=True)
    # Default OFF, unlike the three structural_mass producers above -- same
    # convention as every other new signal in this codebase (e.g.
    # SUBSTRATE_WRITE_PREDICTION_ERROR_NODES's pattern): this is a pure
    # shadow write with no consumer yet (orion/bus/channels.yaml), flip on
    # deliberately once the live stream itself has had a sanity pass, not
    # just the offline replay.
    COCREATION_SIGNALS_AFFECTIVE_STATE_ENABLED: bool = Field(default=False)

    # ── Producer intervals, one per real cadence (see spec's "Producer
    # scheduling" section) ──────────────────────────────────────────────
    COCREATION_SIGNALS_GIT_DELTA_POLL_INTERVAL_SEC: float = Field(default=60.0)
    COCREATION_SIGNALS_PR_LIFECYCLE_POLL_INTERVAL_SEC: float = Field(default=900.0)
    COCREATION_SIGNALS_GRAPH_DELTA_POLL_INTERVAL_SEC: float = Field(default=300.0)
    # 15min, same cadence as pr_lifecycle -- an affective-state read doesn't
    # need git_delta's 60s responsiveness, and scanning the full transcript
    # tree every tick (see affective_state.py's module docstring) is real
    # work worth spacing out.
    COCREATION_SIGNALS_AFFECTIVE_STATE_POLL_INTERVAL_SEC: float = Field(default=900.0)

    # Real, acknowledged gap (code review 2026-07-30): unlike git_delta/
    # graph_delta (diff-based, self-healing across a restart -- a missed
    # window just produces a bigger real diff next time), pr_lifecycle is
    # window-based -- a PR event entirely inside a downtime window longer
    # than this setting is genuinely lost, not recovered. Deliberately a
    # separate, more generous setting from the steady-state poll interval
    # (default 1h vs. the 15min poll cadence) rather than reusing it, so a
    # restart's recovery window doesn't silently shrink to whatever the
    # steady-state cadence happens to be. See pr_lifecycle_loop()'s own
    # docstring for the full reasoning.
    COCREATION_SIGNALS_PR_LIFECYCLE_COLD_START_LOOKBACK_SEC: float = Field(default=3600.0)

    # Same real, acknowledged restart-loss gap as pr_lifecycle's own setting
    # above (window-based, not diff-based) -- see affective_state_loop()'s
    # own docstring.
    COCREATION_SIGNALS_AFFECTIVE_STATE_COLD_START_LOOKBACK_SEC: float = Field(default=3600.0)

    # pr_lifecycle's own `gh pr list --limit` -- must be generous enough to
    # reach back past the oldest event in a poll window (see
    # orion/structural_mass/pr_lifecycle.py's own possibly_truncated docs).
    COCREATION_SIGNALS_PR_FETCH_LIMIT: int = Field(default=200)

    @field_validator(
        "COCREATION_SIGNALS_GIT_DELTA_POLL_INTERVAL_SEC",
        "COCREATION_SIGNALS_PR_LIFECYCLE_POLL_INTERVAL_SEC",
        "COCREATION_SIGNALS_GRAPH_DELTA_POLL_INTERVAL_SEC",
        "COCREATION_SIGNALS_PR_LIFECYCLE_COLD_START_LOOKBACK_SEC",
        "COCREATION_SIGNALS_AFFECTIVE_STATE_POLL_INTERVAL_SEC",
        "COCREATION_SIGNALS_AFFECTIVE_STATE_COLD_START_LOOKBACK_SEC",
    )
    @classmethod
    def _ensure_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Poll interval/lookback must be positive")
        return v

    @field_validator("COCREATION_SIGNALS_PR_FETCH_LIMIT")
    @classmethod
    def _ensure_fetch_limit_positive(cls, v: int) -> int:
        # Confirmed live during code review 2026-07-30: a limit of 0 would
        # make orion/structural_mass/pr_lifecycle.py's
        # `fetch_appears_capped = len(prs) >= fetch_limit` trivially true
        # forever (`len(prs) >= 0`), making `possibly_truncated` fire
        # spuriously on every real tick regardless of actual data -- fail
        # fast at startup instead of silently misreporting truncation later.
        if v <= 0:
            raise ValueError("PR fetch limit must be positive")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
