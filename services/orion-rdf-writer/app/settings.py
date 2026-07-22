from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List


class Settings(BaseSettings):
    # RDF + GraphDB (GRAPHDB_URL optional when RDF_STORE_BACKEND != graphdb)
    GRAPHDB_URL: str | None = Field(default=None, env="GRAPHDB_URL")
    GRAPHDB_REPO: str = Field(default="collapse", env="GRAPHDB_REPO")
    GRAPHDB_USER: str | None = Field(None, env="GRAPHDB_USER")
    GRAPHDB_PASS: str | None = Field(None, env="GRAPHDB_PASS")

    # Backend-neutral RDF store
    RDF_STORE_BACKEND: str = Field(default="fuseki", env="RDF_STORE_BACKEND")
    RDF_STORE_BASE_URL: str | None = Field(default=None, env="RDF_STORE_BASE_URL")
    RDF_STORE_DATASET: str = Field(default="orion", env="RDF_STORE_DATASET")
    RDF_STORE_QUERY_URL: str | None = Field(default=None, env="RDF_STORE_QUERY_URL")
    RDF_STORE_UPDATE_URL: str | None = Field(default=None, env="RDF_STORE_UPDATE_URL")
    RDF_STORE_GRAPH_STORE_URL: str | None = Field(default=None, env="RDF_STORE_GRAPH_STORE_URL")
    RDF_STORE_USER: str | None = Field(default=None, env="RDF_STORE_USER")
    RDF_STORE_PASS: str | None = Field(default=None, env="RDF_STORE_PASS")
    RDF_STORE_TIMEOUT_SEC: float = Field(default=10.0, env="RDF_STORE_TIMEOUT_SEC")
    RDF_STORE_NORMALIZE_GRAPHDB_CONTEXT: bool = Field(
        default=False,
        env="RDF_STORE_NORMALIZE_GRAPHDB_CONTEXT",
    )

    RDF_WRITE_ASYNC_ENABLED: bool = Field(default=True, env="RDF_WRITE_ASYNC_ENABLED")
    RDF_WRITE_QUEUE_MAXSIZE: int = Field(default=5000, env="RDF_WRITE_QUEUE_MAXSIZE")
    RDF_WRITE_WORKERS: int = Field(default=8, env="RDF_WRITE_WORKERS")
    RDF_WRITE_MAX_IN_FLIGHT: int = Field(default=32, env="RDF_WRITE_MAX_IN_FLIGHT")
    RDF_WRITE_HTTP_MAX_CONNECTIONS: int = Field(default=64, env="RDF_WRITE_HTTP_MAX_CONNECTIONS")
    RDF_WRITE_HTTP_MAX_KEEPALIVE: int = Field(default=32, env="RDF_WRITE_HTTP_MAX_KEEPALIVE")
    RDF_WRITE_RETRY_ATTEMPTS: int = Field(default=3, env="RDF_WRITE_RETRY_ATTEMPTS")
    RDF_WRITE_RETRY_BASE_DELAY_SEC: float = Field(default=0.25, env="RDF_WRITE_RETRY_BASE_DELAY_SEC")
    RDF_WRITE_RETRY_MAX_DELAY_SEC: float = Field(default=5.0, env="RDF_WRITE_RETRY_MAX_DELAY_SEC")
    RDF_WRITE_DEAD_LETTER_ENABLED: bool = Field(default=True, env="RDF_WRITE_DEAD_LETTER_ENABLED")
    RDF_WRITE_DEAD_LETTER_PATH: str = Field(
        default="/app/logs/orion-rdf-writer-deadletter.ndjson",
        env="RDF_WRITE_DEAD_LETTER_PATH",
    )

    # === ORION BUS (Shared Core) ===
    ORION_BUS_URL: str = Field(..., env="ORION_BUS_URL")
    ORION_BUS_ENABLED: bool = Field(default=True, env="ORION_BUS_ENABLED")
    ORION_BUS_ENFORCE_CATALOG: bool = Field(default=False, env="ORION_BUS_ENFORCE_CATALOG")

    # === LISTENER CHANNELS ===
    CHANNEL_RDF_ENQUEUE: str = Field(default="orion:rdf:enqueue", env="CHANNEL_RDF_ENQUEUE")
    CHANNEL_EVENTS_COLLAPSE: str = Field(default="orion:collapse:intake", env="CHANNEL_EVENTS_COLLAPSE")
    # orion:tags:chat:enriched (CHANNEL_EVENTS_TAGGED_CHAT) deliberately not
    # subscribed as of 2026-07-18: chat-turn tag/entity data now lands in
    # FalkorDB only (orion-meta-tags' own Phase 2 writer,
    # services/orion-meta-tags/README.md "Dual persistence pathway" ->
    # single pathway after this change). Fuseki's `chat_tagging` enrichment
    # copy of this same data was a second, redundant materialization of the
    # same entities/sentiment already durably written to Falkor -- not an
    # independent signal. orion-meta-tags no longer publishes to this
    # channel either (see its main.py). Do not re-add.
    # orion:tags:enriched (CHANNEL_EVENTS_TAGGED) deliberately not subscribed
    # HERE as of 2026-07-22 -- but the publish itself is NOT dead and must
    # NOT be killed at the source: orion-sql-writer is also subscribed to
    # this channel and materializes it into Postgres `collapse_enrichment`
    # (76 live rows, latest timestamp same-day as this comment), which
    # orion-recall (storage/sql_adapter.py) and orion-dream
    # (aggregators_sql.py) both genuinely query. Only THIS service's Fuseki
    # copy of the same data was redundant (Falkor write shipped additively
    # PR #1271, 68/68 historical rows backfilled PR #1273, one real live
    # event verified landing in Postgres/Fuseki/Falkor simultaneously) -- the
    # bus publish itself stays alive for sql-writer's sake. See
    # docs/superpowers/specs/2026-07-22-tags-enriched-fuseki-kill-spec.md
    # (including its "Correction found during implementation" section) for
    # the full reasoning. Do not re-add this service's subscription without
    # a real Falkor-gap reason, and do not remove orion-meta-tags' publish.
    CHANNEL_CORE_EVENTS: str = Field(default="orion:core:events", env="CHANNEL_CORE_EVENTS")
    # orion:rdf:worker (CHANNEL_WORKER_RDF) deliberately not subscribed as of
    # 2026-07-18: channels.yaml claims orion-cortex-exec as producer, but no
    # code anywhere actually publishes to it (the one hit outside this file
    # was self_study.py referencing the channel *name* in a self-knowledge
    # catalog string, not a bus.publish call). Zero live traffic possible.
    # Do not re-add without a real producer.
    # orion:cognition:trace / orion:metacog:trace deliberately not subscribed
    # as of 2026-07-22: live Fuseki traffic was ~750 writes/6h for these two
    # kinds, pure redundancy -- both already durably owned by Postgres via
    # orion-sql-writer (`cognition_traces`, 61k+ rows;
    # `orion_metacognitive_trace`). The one other Fuseki reader
    # (orion-graph-compression's episodic federator) is fail-open per-graph
    # and degrades correctly when a federated graph goes quiet. Same shape as
    # the chat.history kill below -- a real consumer already exists
    # elsewhere. (A prior attempt at this kill, PR #1155, was reviewed and
    # correct but the PR itself was closed without merging -- this
    # re-implements the same change.) Do not re-add without a real
    # Falkor/Postgres-gap reason.
    # orion:memory:drives:audit deliberately not subscribed: drive audits are
    # Postgres-only (`drive_audits` via orion-sql-writer) as of 2026-07-15.
    # orion:chat:history:turn / orion:chat:history:log deliberately not
    # subscribed (2026-07-17): both are Postgres-only via orion-sql-writer
    # (`chat_message`, `chat_history_log`) -- the RDF copy covered only
    # ~11-18% of real chat volume, live-checked. See split-design spec's open
    # questions for the still-unexplained coverage gap.
    # orion:memory:identity:snapshot / orion:memory:goals:proposed (identity
    # snapshots / goal proposals) deliberately not subscribed as of
    # 2026-07-18: live Fuseki query found zero graphs matching
    # autonomy/identity/goals ever recorded, despite a real producer
    # existing. identity_snapshots had a real, actively-pruned Postgres store
    # (orion-self-state-runtime's SelfStateRuntimeStore) at the time; that
    # service was deleted 2026-07-22 (SelfStateV1 burn), leaving the table
    # with zero producer -- flagged as a follow-up, not a reason to
    # reconsider this channel. Goal proposals are already consumed live by
    # orion-substrate-runtime's
    # goal_context_listener.py. Same shape as the drive-audit kill and the
    # cognition/metacog kill — a real consumer already exists elsewhere.
    # Do not re-add any of the above without a real producer/consumer.

    # === PUBLISH CHANNELS ===
    CHANNEL_RDF_CONFIRM: str = Field(default="orion:rdf:confirm", env="CHANNEL_RDF_CONFIRM")
    CHANNEL_RDF_ERROR: str = Field(default="orion:rdf:error", env="CHANNEL_RDF_ERROR")
    # orion:cortex:telemetry (CORTEX_LOG_CHANNEL) deliberately not subscribed
    # as of 2026-07-18: zero producers anywhere in the repo, not even
    # registered in channels.yaml. Dead subscription. Do not re-add without
    # a real producer.

    SERVICE_NAME: str = Field(default="orion-rdf-writer", env="SERVICE_NAME")
    SERVICE_VERSION: str = Field(default="0.2.0", env="SERVICE_VERSION")
    NODE_NAME: str = Field(default="unknown")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    RDF_SKIP_KINDS: str = Field(default="", env="RDF_SKIP_KINDS")
    RDF_SKIP_REJECTED: bool = Field(default=True, env="RDF_SKIP_REJECTED")
    RDF_DURABLE_ONLY: bool = Field(default=False, env="RDF_DURABLE_ONLY")

    RDF_RETENTION_ENABLED: bool = Field(default=False, env="RDF_RETENTION_ENABLED")
    RDF_RETENTION_DRY_RUN: bool = Field(default=False, env="RDF_RETENTION_DRY_RUN")
    RDF_RETENTION_INTERVAL_HOURS: int = Field(default=168, env="RDF_RETENTION_INTERVAL_HOURS")
    RDF_RETENTION_TIMEOUT_SEC: float = Field(default=3600.0, env="RDF_RETENTION_TIMEOUT_SEC")
    RDF_RETENTION_POLICIES: str | None = Field(default=None, env="RDF_RETENTION_POLICIES")

    # orion:world_pulse:graph:upsert (CHANNEL_WORLD_PULSE_GRAPH) deliberately
    # not subscribed as of 2026-07-18: WORLD_PULSE_GRAPH_ENABLED was false in
    # the live .env (fully inert), and even enabled it defaulted to dry-run.
    # Graph shape was 3 flat literal properties per digest item
    # (category/title/runId) with no edges to anything else -- no real graph
    # structure lost, and world-pulse's richer claim/event/entity emit
    # channels already reach Postgres via orion-sql-writer. Do not re-add;
    # if world-pulse ever needs graph storage, design it fresh against
    # Falkor instead of reviving this dormant SPARQL path. The WORLD_PULSE_*
    # settings fields themselves are removed too -- nothing else reads them.

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def get_all_subscribe_channels(self) -> List[str]:
        channels = [
            self.CHANNEL_RDF_ENQUEUE,
            "orion:rdf-collapse:enqueue",
            self.CHANNEL_EVENTS_COLLAPSE,
            "orion:chat:social:stored",
            self.CHANNEL_CORE_EVENTS,
        ]
        seen = set()
        ordered: List[str] = []
        for channel in channels:
            channel = (channel or "").strip()
            if not channel or channel in seen:
                continue
            seen.add(channel)
            ordered.append(channel)
        return ordered

    def get_skip_kinds(self) -> List[str]:
        return [k.strip() for k in (self.RDF_SKIP_KINDS or "").split(",") if k.strip()]


settings = Settings()
