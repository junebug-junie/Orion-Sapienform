from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_claude_limit_specs(raw: str) -> tuple[tuple[str, float, float], ...]:
    """Parse `kind:hours:interval_sec` triples into validated specs.

    Shared by the validator and the property so the two can never disagree
    about what a given string means. Returns () for anything unusable -- the
    validator turns that into a boot failure, and the property can then trust
    its own output.

    The KIND is required, not inferred from the window size. `LimitObservation.
    state` reads the chronologically last event regardless of kind, so a
    published series that does not say which limit it is about is one a
    consumer cannot use without re-merging the two pools.
    """
    valid_kinds = ("session_limit", "weekly_limit")
    out: list[tuple[str, float, float]] = []
    seen: set[tuple[str, float]] = set()
    for chunk in (raw or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(":")]
        if len(parts) != 3:
            return ()
        kind, hours_raw, interval_raw = parts
        if kind not in valid_kinds:
            return ()
        try:
            hours = float(hours_raw)
            interval = float(interval_raw)
        except ValueError:
            return ()
        if hours <= 0 or interval <= 0:
            return ()
        key = (kind, hours)
        if key in seen:
            # A duplicate series would publish the same fact twice per tick and
            # double whatever a consumer counts. Refuse rather than dedupe
            # silently: it is a config typo, not an intent worth guessing at.
            return ()
        seen.add(key)
        out.append((kind, hours, interval))
    return tuple(out)


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
    CHANNEL_DEV_ECONOMICS_LEDGER: str = Field(default="orion:substrate:dev_economics_ledger")
    CHANNEL_CLAUDE_LIMIT: str = Field(default="orion:substrate:claude_limit")

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
    # Default OFF, same reasoning as affective_state above -- pure shadow
    # write, flip on deliberately once the live stream has had a sanity pass.
    COCREATION_SIGNALS_DEV_ECONOMICS_ENABLED: bool = Field(default=False)
    # Default OFF like its two transcript-scanning neighbours, but for a
    # different reason: this one is NOT a shadow write -- orion-hub consumes it
    # from day one (orion/bus/channels.yaml). Off by default because it reads
    # Juniper's transcript tree and the operator should turn that on knowingly,
    # not because the signal is unproven. rate_limit_events.py has been correct
    # since it shipped; what was missing was a consumer, not confidence.
    COCREATION_SIGNALS_CLAUDE_LIMIT_ENABLED: bool = Field(default=False)

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
    # Same cadence and reasoning as affective_state above -- both scan the
    # same real transcript tree, just extracting different signals from it.
    COCREATION_SIGNALS_DEV_ECONOMICS_POLL_INTERVAL_SEC: float = Field(default=900.0)
    # `kind:hours:interval_sec` triples, one per published series. The kind is
    # required rather than inferred: `LimitObservation.state` reads the last
    # event in the window regardless of kind, so a weekly limit still in force
    # reads `clear` the moment a spent session limit lands after it.
    #
    # The two intervals differ on purpose. Measured on the real tree, the 168h
    # scan reads 429 files / 462MB / 2.4s while the 5h scan reads 42 / 110MB.
    # The tight cadence exists because a session limit's stated reset time
    # expires; a weekly window's answer moves on a day scale, and re-reading
    # 460MB every five minutes for it would be ~165GB/day of pure waste.
    COCREATION_SIGNALS_CLAUDE_LIMIT_WINDOWS: str = Field(
        default="session_limit:5:300,weekly_limit:168:3600"
    )

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

    # ── doc-semantic-drift (docs/superpowers/specs/2026-07-30-doc-semantic-
    # drift-design.md) ──────────────────────────────────────────────────
    # Default OFF -- pure shadow write, no consumer yet (orion/bus/
    # channels.yaml), same convention as affective_state's own rollout.
    # Real replay (docs/superpowers/pr-reports/2026-08-11-doc-semantic-
    # drift-diff-scoped-embedding.md) confirmed the diff-scoped embedding
    # signal separates trivial from real doc edits, but on only 3 real
    # non-truncated samples -- flip on deliberately once the live stream
    # has had its own sanity pass, not just the offline replay.
    COCREATION_SIGNALS_DOC_SEMANTIC_DRIFT_ENABLED: bool = Field(default=False)
    # Doc edits are far less frequent than git commits in general -- same
    # cadence as graph_delta, no responsiveness need for git_delta's 60s.
    COCREATION_SIGNALS_DOC_SEMANTIC_DRIFT_POLL_INTERVAL_SEC: float = Field(default=300.0)
    CHANNEL_DOC_SEMANTIC_DRIFT: str = Field(default="orion:substrate:doc_semantic_drift")
    # Real, already-registered channel (orion/bus/channels.yaml) this
    # producer requests embeddings from -- orion-vector-host, the same
    # live, non-frontier model (BAAI/bge-large-en-v1.5) the calibration
    # replay used. No new channel needed; producer_services already
    # includes "*".
    CHANNEL_EMBEDDING_GENERATE: str = Field(default="orion:embedding:generate")
    # Scoped collection so these hunk-diff embeddings (real, already-
    # committed doc/code text -- not chat, not personal data) land in their
    # own vector-store collection, not commingled with chat/social memory.
    # Decided explicitly with Juniper 2026-08-11 rather than defaulted:
    # orion-vector-host's real embedding-request contract persists every
    # embedded text as a vector-store document unconditionally (confirmed
    # live, no opt-out exists) -- every other real caller of that contract
    # already accepts this, so this producer following the same
    # established pattern, scoped to its own collection, is consistent
    # with existing architecture rather than a new precedent.
    COCREATION_SIGNALS_DOC_SEMANTIC_DRIFT_EMBED_COLLECTION: str = Field(
        default="doc_semantic_drift"
    )
    COCREATION_SIGNALS_DOC_SEMANTIC_DRIFT_EMBED_TIMEOUT_SEC: float = Field(default=30.0)
    # Chunk window each hunk side is split into before embedding, so a long
    # doc diff gets chunked instead of silently clipped inside the model.
    # Replaces `..._TRUNCATION_CHAR_THRESHOLD`, which only *flagged* an
    # over-long hunk (and fired True on every real event, making it useless
    # as a signal).
    #
    # 1200, not the obvious 2048. Measured 2026-08-12 by running 400 real
    # chunks from this repo's own *.md history through the live
    # orion-vector-host container's BAAI/bge-large-en-v1.5 tokenizer:
    #
    #   at 2048 chars -> median 541 tokens, p90 608, max 722
    #                    66.5% of chunks EXCEEDED the model's 512 ceiling
    #   chars/token on near-full chunks -> median 3.59, min 2.81
    #
    # The 512 ceiling is hard and `embedder.embed()` passes truncation=True,
    # so those chunks were dropped inside the model with no error. The
    # earlier ~4 chars/token estimate was simply wrong for this corpus --
    # paths, code fences and identifiers tokenize far denser than prose.
    # 512 * 2.81 (the observed minimum ratio) = 1437, and 1200 leaves real
    # margin under it. Note scripts/analysis/measure_doc_semantic_drift.py
    # already rejected a chars-per-token proxy for this same corpus in
    # favor of a live token count; this remains a proxy, just a measured
    # one rather than a guessed one.
    #
    # gt=0 is load-bearing: `_chunk_text` returns the text unchunked for
    # max_chars <= 0, which would silently restore full-text clipping with
    # chunk_count=1 and no flag.
    COCREATION_SIGNALS_DOC_SEMANTIC_DRIFT_CHUNK_CHAR_SIZE: int = Field(default=1200, gt=0)

    # Real Redis key doc_semantic_drift_loop's baseline last_sha is persisted
    # to, so a redeploy resumes from where it left off instead of re-seeding
    # at whatever HEAD happens to be at boot -- fixed live 2026-08-12 after a
    # real doc (PR #1571, then again #1577) got silently swallowed by a
    # redeploy landing right after its merge. See
    # doc_semantic_drift_loop()'s own docstring for the full story.
    COCREATION_SIGNALS_DOC_SEMANTIC_DRIFT_STATE_KEY: str = Field(
        default="orion:cocreation_signals:state:doc_semantic_drift:last_sha"
    )

    @field_validator(
        "COCREATION_SIGNALS_GIT_DELTA_POLL_INTERVAL_SEC",
        "COCREATION_SIGNALS_PR_LIFECYCLE_POLL_INTERVAL_SEC",
        "COCREATION_SIGNALS_GRAPH_DELTA_POLL_INTERVAL_SEC",
        "COCREATION_SIGNALS_PR_LIFECYCLE_COLD_START_LOOKBACK_SEC",
        "COCREATION_SIGNALS_AFFECTIVE_STATE_POLL_INTERVAL_SEC",
        "COCREATION_SIGNALS_AFFECTIVE_STATE_COLD_START_LOOKBACK_SEC",
        "COCREATION_SIGNALS_DOC_SEMANTIC_DRIFT_POLL_INTERVAL_SEC",
        "COCREATION_SIGNALS_DOC_SEMANTIC_DRIFT_EMBED_TIMEOUT_SEC",
        "COCREATION_SIGNALS_DEV_ECONOMICS_POLL_INTERVAL_SEC",
    )
    @classmethod
    def _ensure_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Poll interval/lookback must be positive")
        return v

    @field_validator("COCREATION_SIGNALS_CLAUDE_LIMIT_WINDOWS")
    @classmethod
    def _ensure_windows_parse(cls, v: str) -> str:
        # Validated here rather than at the call site so a typo fails at boot
        # with the offending value named, instead of starting a producer that
        # publishes nothing and logs "no windows configured" forever.
        if not _parse_claude_limit_specs(v):
            raise ValueError(
                "COCREATION_SIGNALS_CLAUDE_LIMIT_WINDOWS must be comma-separated "
                "kind:hours:interval_sec triples, kind in (session_limit, weekly_limit), "
                f"hours and interval positive, no duplicate kind+hours; got {v!r}"
            )
        return v

    @property
    def claude_limit_specs(self) -> tuple[tuple[str, float, float], ...]:
        """Parsed, in configured order. Not sorted: the operator's order is the
        publish order, and there is no cross-series ordering guarantee to
        preserve now that each series runs on its own task and cadence."""
        return _parse_claude_limit_specs(self.COCREATION_SIGNALS_CLAUDE_LIMIT_WINDOWS)

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
