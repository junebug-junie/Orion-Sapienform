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
    CHANNEL_EVENTS_TAGGED: str = Field(default="orion:tags:enriched", env="CHANNEL_EVENTS_TAGGED")
    # orion:tags:chat:enriched (CHANNEL_EVENTS_TAGGED_CHAT) deliberately not
    # subscribed as of 2026-07-18: chat-turn tag/entity data now lands in
    # FalkorDB only (orion-meta-tags' own Phase 2 writer,
    # services/orion-meta-tags/README.md "Dual persistence pathway" ->
    # single pathway after this change). Fuseki's `chat_tagging` enrichment
    # copy of this same data was a second, redundant materialization of the
    # same entities/sentiment already durably written to Falkor -- not an
    # independent signal. orion-meta-tags no longer publishes to this
    # channel either (see its main.py). Do not re-add.
    CHANNEL_CORE_EVENTS: str = Field(default="orion:core:events", env="CHANNEL_CORE_EVENTS")
    # orion:rdf:worker (CHANNEL_WORKER_RDF) deliberately not subscribed as of
    # 2026-07-18: channels.yaml claims orion-cortex-exec as producer, but no
    # code anywhere actually publishes to it (the one hit outside this file
    # was self_study.py referencing the channel *name* in a self-knowledge
    # catalog string, not a bus.publish call). Zero live traffic possible.
    # Do not re-add without a real producer.
    CHANNEL_COGNITION_TRACE_PUB: str = Field(default="orion:cognition:trace", env="CHANNEL_COGNITION_TRACE_PUB")
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
    # existing. identity_snapshots already has a real, actively-pruned
    # Postgres store (orion-self-state-runtime's SelfStateRuntimeStore), and
    # goal proposals are already consumed live by orion-substrate-runtime's
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
            self.CHANNEL_EVENTS_TAGGED,
            "orion:chat:social:stored",
            self.CHANNEL_CORE_EVENTS,
            self.CHANNEL_COGNITION_TRACE_PUB,
            "orion:metacog:trace",
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
