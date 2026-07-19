# services/orion-recall/app/settings.py
from __future__ import annotations

import os
from typing import Optional

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Recall service settings.

    Source-of-truth precedence:
      1) Environment variables (docker-compose passes these explicitly)
      2) Local .env file (only for local dev / direct uvicorn runs)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Service Metadata ──────────────────────────────────────────────
    SERVICE_NAME: str = Field(default="recall", validation_alias=AliasChoices("SERVICE_NAME"))
    SERVICE_VERSION: str = Field(default="0.1.0", validation_alias=AliasChoices("SERVICE_VERSION"))
    NODE_NAME: str = Field(default="unknown", validation_alias=AliasChoices("NODE_NAME", "HOSTNAME"))
    PORT: int = Field(default=8260, validation_alias=AliasChoices("PORT"))

    # ── Orion Bus ─────────────────────────────────────────────────────
    ORION_BUS_ENABLED: bool = Field(default=True, validation_alias=AliasChoices("ORION_BUS_ENABLED"))
    ORION_BUS_ENFORCE_CATALOG: bool = Field(default=False, validation_alias=AliasChoices("ORION_BUS_ENFORCE_CATALOG"))
    ORION_BUS_URL: str = Field(
        default="redis://127.0.0.1:6379/0",
        validation_alias=AliasChoices("ORION_BUS_URL", "REDIS_URL"),
    )

    # RPC intake + reply + telemetry
    RECALL_BUS_INTAKE: str = Field(
        default="orion:exec:request:RecallService",
        validation_alias=AliasChoices("RECALL_BUS_INTAKE", "CHANNEL_RECALL_REQUEST"),
    )
    RECALL_BUS_REPLY_DEFAULT: str = Field(
        default="orion:exec:result:RecallService",
        validation_alias=AliasChoices("RECALL_BUS_REPLY_DEFAULT", "CHANNEL_RECALL_DEFAULT_REPLY_PREFIX"),
    )
    RECALL_BUS_TELEMETRY: str = Field(
        default="orion:recall:telemetry",
        validation_alias=AliasChoices("RECALL_BUS_TELEMETRY"),
    )

    # ── Chassis / Runtime ─────────────────────────────────────────────
    HEARTBEAT_INTERVAL_SEC: float = Field(default=10.0, validation_alias=AliasChoices("HEARTBEAT_INTERVAL_SEC"))
    ORION_HEALTH_CHANNEL: str = Field(default="orion:system:health", validation_alias=AliasChoices("ORION_HEALTH_CHANNEL"))
    ERROR_CHANNEL: str = Field(default="orion:system:error", validation_alias=AliasChoices("ERROR_CHANNEL"))
    SHUTDOWN_GRACE_SEC: float = Field(default=10.0, validation_alias=AliasChoices("SHUTDOWN_GRACE_SEC"))
    # When true, Rabbit handler runs concurrent requests (reduces head-of-line blocking on slow RPCs).
    RECALL_RABBIT_CONCURRENT_HANDLERS: bool = Field(
        default=False,
        validation_alias=AliasChoices("RECALL_RABBIT_CONCURRENT_HANDLERS"),
    )

    # ── Default Recall Behavior ───────────────────────────────────────
    RECALL_DEFAULT_MAX_ITEMS: int = Field(default=16, validation_alias=AliasChoices("RECALL_DEFAULT_MAX_ITEMS"))
    RECALL_DEFAULT_TIME_WINDOW_DAYS: int = Field(
        default=30, validation_alias=AliasChoices("RECALL_DEFAULT_TIME_WINDOW_DAYS")
    )
    RECALL_DEFAULT_MODE: str = Field(default="hybrid", validation_alias=AliasChoices("RECALL_DEFAULT_MODE"))
    RECALL_DEFAULT_PROFILE: str = Field(default="reflect.v1", validation_alias=AliasChoices("RECALL_DEFAULT_PROFILE"))

    # ── Source Toggles ────────────────────────────────────────────────
    RECALL_ENABLE_SQL_CHAT: bool = Field(default=True, validation_alias=AliasChoices("RECALL_ENABLE_SQL_CHAT"))
    RECALL_ENABLE_SQL_MIRRORS: bool = Field(default=True, validation_alias=AliasChoices("RECALL_ENABLE_SQL_MIRRORS"))
    RECALL_ENABLE_VECTOR: bool = Field(default=False, validation_alias=AliasChoices("RECALL_ENABLE_VECTOR"))
    RECALL_ENABLE_RDF: bool = Field(default=False, validation_alias=AliasChoices("RECALL_ENABLE_RDF"))

    # ── Postgres / SQL ────────────────────────────────────────────────
    RECALL_PG_DSN: str = Field(
        default="postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney",
        validation_alias=AliasChoices("RECALL_PG_DSN", "POSTGRES_URI", "POSTGRES_DSN"),
    )

    # Chat history
    RECALL_SQL_CHAT_TABLE: str = Field(default="chat_history_log", validation_alias=AliasChoices("RECALL_SQL_CHAT_TABLE"))
    RECALL_SQL_CHAT_TEXT_COL: str = Field(default="prompt", validation_alias=AliasChoices("RECALL_SQL_CHAT_TEXT_COL"))
    RECALL_SQL_CHAT_RESPONSE_COL: str = Field(
        default="response", validation_alias=AliasChoices("RECALL_SQL_CHAT_RESPONSE_COL")
    )
    RECALL_SQL_CHAT_CREATED_AT_COL: str = Field(
        default="created_at", validation_alias=AliasChoices("RECALL_SQL_CHAT_CREATED_AT_COL")
    )
    RECALL_SQL_CHAT_ID_COL: str = Field(
        default="id", validation_alias=AliasChoices("RECALL_SQL_CHAT_ID_COL")
    )
    # RDF chat-turn recall carries no usable graph timestamp (turns are written without one),
    # so it ignored the per-profile time window and surfaced months-old turns into reflective
    # recall. When enabled, RDF chat-turn candidates are joined back to chat_history_log and
    # dropped when older than the profile's sql_since_minutes (giving them the same window the
    # SQL chat/timeline backends already honor).
    RECALL_RDF_CHAT_WINDOW_ENABLED: bool = Field(
        default=True, validation_alias=AliasChoices("RECALL_RDF_CHAT_WINDOW_ENABLED")
    )

    # Collapse mirror base + semantic fields
    RECALL_SQL_MIRROR_TABLE: str = Field(default="collapse_mirror", validation_alias=AliasChoices("RECALL_SQL_MIRROR_TABLE"))
    RECALL_SQL_MIRROR_SUMMARY_COL: str = Field(default="summary", validation_alias=AliasChoices("RECALL_SQL_MIRROR_SUMMARY_COL"))
    RECALL_SQL_MIRROR_TRIGGER_COL: str = Field(default="trigger", validation_alias=AliasChoices("RECALL_SQL_MIRROR_TRIGGER_COL"))
    RECALL_SQL_MIRROR_OBSERVER_COL: str = Field(default="observer", validation_alias=AliasChoices("RECALL_SQL_MIRROR_OBSERVER_COL"))
    RECALL_SQL_MIRROR_OBSERVER_STATE_COL: str = Field(
        default="observer_state", validation_alias=AliasChoices("RECALL_SQL_MIRROR_OBSERVER_STATE_COL")
    )
    RECALL_SQL_MIRROR_FIELD_RESONANCE_COL: str = Field(
        default="field_resonance", validation_alias=AliasChoices("RECALL_SQL_MIRROR_FIELD_RESONANCE_COL")
    )
    RECALL_SQL_MIRROR_INTENT_COL: str = Field(default="intent", validation_alias=AliasChoices("RECALL_SQL_MIRROR_INTENT_COL"))
    RECALL_SQL_MIRROR_TYPE_COL: str = Field(default="type", validation_alias=AliasChoices("RECALL_SQL_MIRROR_TYPE_COL"))
    RECALL_SQL_MIRROR_ENTITY_COL: str = Field(
        default="emergent_entity", validation_alias=AliasChoices("RECALL_SQL_MIRROR_ENTITY_COL")
    )
    RECALL_SQL_MIRROR_MANTRA_COL: str = Field(default="mantra", validation_alias=AliasChoices("RECALL_SQL_MIRROR_MANTRA_COL"))
    RECALL_SQL_MIRROR_CAUSAL_ECHO_COL: str = Field(
        default="causal_echo", validation_alias=AliasChoices("RECALL_SQL_MIRROR_CAUSAL_ECHO_COL")
    )
    RECALL_SQL_MIRROR_TS_COL: str = Field(default="timestamp", validation_alias=AliasChoices("RECALL_SQL_MIRROR_TS_COL"))

    # Collapse enrichment
    RECALL_SQL_ENRICH_TABLE: str = Field(default="collapse_enrichment", validation_alias=AliasChoices("RECALL_SQL_ENRICH_TABLE"))
    RECALL_SQL_ENRICH_COLLAPSE_ID_COL: str = Field(
        default="collapse_id", validation_alias=AliasChoices("RECALL_SQL_ENRICH_COLLAPSE_ID_COL")
    )
    RECALL_SQL_ENRICH_TAGS_COL: str = Field(default="tags", validation_alias=AliasChoices("RECALL_SQL_ENRICH_TAGS_COL"))
    RECALL_SQL_ENRICH_ENTITIES_COL: str = Field(
        default="entities", validation_alias=AliasChoices("RECALL_SQL_ENRICH_ENTITIES_COL")
    )
    RECALL_SQL_ENRICH_SALIENCE_COL: str = Field(
        default="salience", validation_alias=AliasChoices("RECALL_SQL_ENRICH_SALIENCE_COL")
    )
    RECALL_SQL_ENRICH_TS_COL: str = Field(default="ts", validation_alias=AliasChoices("RECALL_SQL_ENRICH_TS_COL"))

    # ── Global Vector / RDF knobs ─────────────────────────────────────
    VECTOR_DB_HOST: str = Field(default="orion-athena-vector-db", validation_alias=AliasChoices("VECTOR_DB_HOST"))
    VECTOR_DB_PORT: int = Field(default=8000, validation_alias=AliasChoices("VECTOR_DB_PORT"))
    VECTOR_DB_COLLECTION: str = Field(default="orion_main_store", validation_alias=AliasChoices("VECTOR_DB_COLLECTION"))

    GRAPHDB_URL: str = Field(default="", validation_alias=AliasChoices("GRAPHDB_URL"))
    GRAPHDB_REPO: str = Field(default="collapse", validation_alias=AliasChoices("GRAPHDB_REPO"))
    GRAPHDB_USER: str = Field(default="admin", validation_alias=AliasChoices("GRAPHDB_USER"))
    GRAPHDB_PASS: str = Field(default="admin", validation_alias=AliasChoices("GRAPHDB_PASS"))

    # ── Vector backend (recall-specific overrides) ────────────────────
    RECALL_VECTOR_BASE_URL: Optional[str] = Field(default=None, validation_alias=AliasChoices("RECALL_VECTOR_BASE_URL"))
    RECALL_VECTOR_COLLECTIONS: Optional[str] = Field(default=None, validation_alias=AliasChoices("RECALL_VECTOR_COLLECTIONS"))
    RECALL_VECTOR_EMBEDDING_URL: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("RECALL_VECTOR_EMBEDDING_URL"),
    )
    RECALL_VECTOR_TIMEOUT_SEC: float = Field(default=5.0, validation_alias=AliasChoices("RECALL_VECTOR_TIMEOUT_SEC"))
    RECALL_VECTOR_MAX_ITEMS: int = Field(default=24, validation_alias=AliasChoices("RECALL_VECTOR_MAX_ITEMS"))
    RECALL_EXCLUDE_REJECTED: bool = Field(default=True, validation_alias=AliasChoices("RECALL_EXCLUDE_REJECTED"))
    RECALL_DURABLE_ONLY: bool = Field(default=False, validation_alias=AliasChoices("RECALL_DURABLE_ONLY"))
    # Recall V2 may consume enriched journal provenance emitted via pageindex query results.
    RECALL_V2_PAGEINDEX_URL: str = Field(
        default="http://orion-athena-pageindex:8384",
        validation_alias=AliasChoices("RECALL_V2_PAGEINDEX_URL"),
    )
    # Emit REC_TAPE RECALL debug dumps for top-N selected items.
    RECALL_DEBUG_DUMP_TOP_N: int = Field(default=0, validation_alias=AliasChoices("RECALL_DEBUG_DUMP_TOP_N"))

    RECALL_MEMORY_GRAPH_SPARQL_ENABLED: bool = Field(
        default=False, validation_alias=AliasChoices("RECALL_MEMORY_GRAPH_SPARQL_ENABLED")
    )
    RECALL_MEMORY_GRAPH_NAMED_GRAPHS: str = Field(
        default="", validation_alias=AliasChoices("RECALL_MEMORY_GRAPH_NAMED_GRAPHS")
    )
    RECALL_MEMORY_GRAPH_SPARQL_TIMEOUT_SEC: float = Field(
        default=2.0, validation_alias=AliasChoices("RECALL_MEMORY_GRAPH_SPARQL_TIMEOUT_SEC")
    )

    # ── Memory cards (Postgres) ───────────────────────────────────────
    RECALL_ENABLE_CARDS: bool = Field(default=True, validation_alias=AliasChoices("RECALL_ENABLE_CARDS"))
    RECALL_CARDS_TIMEOUT_SEC: float = Field(default=8.0, validation_alias=AliasChoices("RECALL_CARDS_TIMEOUT_SEC"))
    RECALL_CARDS_MAX_NEIGHBORS: int = Field(default=6, validation_alias=AliasChoices("RECALL_CARDS_MAX_NEIGHBORS"))
    RECALL_CARDS_EMBEDDING_URL: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("RECALL_CARDS_EMBEDDING_URL"),
    )
    RECALL_CARDS_EMBED_TIMEOUT_SEC: float = Field(
        default=5.0,
        validation_alias=AliasChoices("RECALL_CARDS_EMBED_TIMEOUT_SEC"),
    )
    RECALL_CARDS_MIN_SIMILARITY: float = Field(
        default=0.32,
        validation_alias=AliasChoices("RECALL_CARDS_MIN_SIMILARITY"),
    )
    RECALL_CARDS_EMBED_CONCURRENCY: int = Field(
        default=4,
        validation_alias=AliasChoices("RECALL_CARDS_EMBED_CONCURRENCY"),
    )
    RECALL_INTENT_ROUTING_ENABLED: bool = Field(default=True, validation_alias=AliasChoices("RECALL_INTENT_ROUTING_ENABLED"))
    RECALL_VECTOR_EXCLUDE_COLLECTIONS: str = Field(
        default="",
        validation_alias=AliasChoices("RECALL_VECTOR_EXCLUDE_COLLECTIONS"),
    )
    RECALL_RENDER_BUDGET_INDICATOR: bool = Field(
        default=True,
        validation_alias=AliasChoices("RECALL_RENDER_BUDGET_INDICATOR"),
    )

    # ── Purpose-conditioned recall (PCR) ──────────────────────────────
    RECALL_PCR_ENABLED: bool = Field(default=True, validation_alias=AliasChoices("RECALL_PCR_ENABLED"))
    RECALL_SKIP_MAX_NOVELTY: float = Field(
        default=0.25, validation_alias=AliasChoices("RECALL_SKIP_MAX_NOVELTY")
    )
    RECALL_CONTINUITY_SQL_MINUTES: int = Field(
        default=120, validation_alias=AliasChoices("RECALL_CONTINUITY_SQL_MINUTES")
    )
    RECALL_CONTINUITY_RENDER_BUDGET: int = Field(
        default=96, validation_alias=AliasChoices("RECALL_CONTINUITY_RENDER_BUDGET")
    )
    RECALL_ACTIVE_PACKET_ENABLED: bool = Field(
        default=True, validation_alias=AliasChoices("RECALL_ACTIVE_PACKET_ENABLED")
    )
    # Turn-scoped concept-region collector (services/orion-recall/app/collectors/
    # concept_region.py) -- cheap, self-gating label-substring match against the
    # substrate concept graph (golden Orion/Juniper/relationship concepts +
    # organically-grown topic-foundry concepts, see orion/substrate/seed.py and
    # orion/substrate/adapters/topic_foundry.py). Reads via
    # app/substrate_store.py's lazily-initialized store handle (env-driven
    # backend selection: SUBSTRATE_STORE_BACKEND/FALKORDB_URI/
    # FALKORDB_SUBSTRATE_GRAPH below), never raises.
    RECALL_CONCEPT_REGION_ENABLED: bool = Field(
        default=True, validation_alias=AliasChoices("RECALL_CONCEPT_REGION_ENABLED")
    )
    RECALL_BELIEF_RENDER_BUDGET: int = Field(
        default=128, validation_alias=AliasChoices("RECALL_BELIEF_RENDER_BUDGET")
    )
    RECALL_GRAPHITI_IN_CHAT: bool = Field(
        default=False, validation_alias=AliasChoices("RECALL_GRAPHITI_IN_CHAT")
    )
    RECALL_GRAPHITI_ADAPTER_URL: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("RECALL_GRAPHITI_ADAPTER_URL")
    )
    RECALL_GRAPHITI_FALKORDB_URI: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("RECALL_GRAPHITI_FALKORDB_URI")
    )
    RECALL_GRAPHITI_TIMEOUT_SEC: float = Field(
        default=10.0, validation_alias=AliasChoices("RECALL_GRAPHITI_TIMEOUT_SEC")
    )
    # Phase 4 of the recall RDF->Falkor cutover (docs/superpowers/specs/
    # 2026-07-17-recall-rdf-writer-falkor-cutover-phase2-spec.md). When true,
    # storage/falkor_chat_adapter.py::fetch_falkor_chatturn_fragments SWAPS IN
    # for storage/rdf_adapter.py::fetch_rdf_chatturn_fragments in worker.py's
    # _query_backends -- not additive like RECALL_GRAPHITI_IN_CHAT (that flag
    # merges an extra rail; this one replaces a rail, since the whole point of
    # this migration arc is retiring RDF/Fuseki). Independent of
    # RECALL_ENABLE_RDF/_rdf_enabled() -- Falkor chatturn fetch does not
    # require "RDF" to be enabled as a concept. Only covers chatturn fragments
    # (prompt/response text, joined from Postgres since the Falkor ChatTurn
    # node is deliberately thin); fetch_rdf_graphtri_fragments (Claim-based)
    # stays on Fuseki -- that function's entire shape assumes Claim nodes,
    # which the Falkor writer replaced with HAS_TAG/MENTIONS_ENTITY edges, so
    # it needs its own redesign, not a rewrite. Code-level default stays
    # False (a safe fallback for any environment that hasn't set this key
    # at all -- False just means "keep using RDF for chatturn," not a
    # data-loss state, unlike RECALL_FALKOR_TAG_ENTITY_ENABLED's write-side
    # flag). The real operator default is True as of 2026-07-18 --
    # .env_example ships true; see its comment and README.md's "Known gap"
    # note for the historical-backfill caveat.
    RECALL_FALKOR_IN_CHAT: bool = Field(
        default=False, validation_alias=AliasChoices("RECALL_FALKOR_IN_CHAT")
    )
    # Second Falkor swap flag, deliberately separate from RECALL_FALKOR_IN_CHAT
    # -- graphtri (Claim-based) recall is a distinct code path serving
    # different profiles (graphtri.v1, deep.graph.v1, and brain.recall.v1's
    # expansion chain) with a genuinely different, newer, less-proven redesign
    # than the chatturn swap, so it gets its own independent dark-by-default
    # rollout rather than riding on RECALL_FALKOR_IN_CHAT's already-live state.
    # When true, storage/falkor_graphtri_adapter.py's
    # fetch_falkor_graphtri_fragments/fetch_falkor_graphtri_anchors SWAP IN for
    # storage/rdf_adapter.py's fetch_rdf_graphtri_fragments/fetch_graphtri_anchors
    # at both call sites in worker.py (_query_backends's graphtri branch, and
    # process_recall's graphtri-profile branch via _build_anchor_set) --
    # swap, not additive, same reasoning as RECALL_FALKOR_IN_CHAT.
    #
    # Filtering divergence from the RDF version, named not silent: the old
    # SPARQL filtered at the TURN level (keyword CONTAINS on raw prompt/
    # response text), which Falkor's thin ChatTurn node can't do without a
    # Postgres join. This filters at the ENTITY level instead (keyword match
    # against Entity.name directly in Cypher) -- no Postgres join needed, and
    # arguably more semantically correct for a graph meant to represent
    # entity relevance in the first place, not a proxy for turn-text search.
    #
    # This is a lower-risk redesign than it looks: a live audit (Phase 0
    # spec's "Ground truth" section) found Claim.predicate only ever took 2
    # fixed values in production (hasTag, mentionsEntity -- never open-
    # vocabulary), confidence/salience were always 0.0/0.0 (dead constants,
    # confirmed never real signal), and no downstream code (fusion.py,
    # render.py) ever parsed the predicate/object structure -- the whole
    # "Claim: ..." string was always carried as opaque text. Since
    # :Tag/HAS_TAG is also empty by design (see RECALL_FALKOR_IN_CHAT's
    # sibling write-side comment), MENTIONS_ENTITY already covers the only
    # part of the old Claim shape that was ever real.
    RECALL_FALKOR_GRAPHTRI_IN_CHAT: bool = Field(
        default=False, validation_alias=AliasChoices("RECALL_FALKOR_GRAPHTRI_IN_CHAT")
    )
    # Phase 2 of the entity-graph-reasoning arc (docs/superpowers/specs/
    # 2026-07-19-recall-entity-graph-reasoning-arc.md): fuse_candidates
    # (fusion.py) additively boosts a candidate's composite score when its
    # source turn's MENTIONS_ENTITY set overlaps with entities Jaccard-
    # related to the query's own extracted entities (app/storage/
    # falkor_entity_relatedness.py::fetch_related_entities). Ships dark --
    # the chosen integration shape (fusion-weight boost, over query-
    # expansion or a new backend) directly couples relatedness scoring into
    # an already-complex ranking function, so it needs live before/after
    # evidence before flipping on, same bar as every other Falkor swap-in
    # in this service. Scoped to falkor_chat candidates only in this first
    # cut: that's the only source whose id/uri reliably IS the real Falkor
    # turn_id (confirmed by reading falkor_chat_adapter.py) -- sql_chat's id
    # is a correlation_id/synthetic fallback that doesn't always match.
    #
    # FIXED (was a known limitation): app/worker.py::_extract_entities had a
    # literal double-backslash regex bug (`\\s`/`\\.` inside an r-string
    # match a literal backslash, not whitespace/dot) that silently broke
    # multi-word entity spans ("New York" -> "New","York" separately) and
    # dropped all-caps acronyms ("NVIDIA" -> nothing). Fixed in its own
    # changeset (worker.py::_extract_entities' docstring, and
    # tests/test_extract_entities.py) after confirming, by reading every
    # caller, that the fix only improves the other two call sites too (more
    # precise SQL ILIKE patterns, better query-expansion signals) -- not
    # something this boost feature should have carried a caveat about
    # forever. Still no stopword filtering (sentence-initial capitalized
    # words like "Tell" still extract as false-positive "entities") -- a
    # separate, smaller, still-open precision gap, not a correctness bug.
    #
    # SECOND FIX, load-bearing: live testing across 6 real queries / 3
    # profiles showed the boost-only design above NEVER changed a single
    # ranking, because falkor_chat's own fetch (falkor_chat_adapter.py) is
    # deliberately recency-windowed with no query filter (Phase 4's own
    # design) -- an entity from an older turn never entered the candidate
    # pool for the boost to act on in the first place. _compute_entity_
    # relatedness_boost_map now ALSO fetches real ChatTurn ids that mention
    # the query's target entities directly (fetch_turns_mentioning_entities,
    # independent of recency), hydrates them via the same Postgres join
    # falkor_chat_adapter.py already uses, and injects them as new
    # candidates -- not just re-ranking whatever the recency fetch happened
    # to grab. This is what actually made the feature work: re-verified
    # live afterward, 4 of 6 real queries changed, and in every changed case
    # the new top result was visibly more relevant to the query's topic
    # (e.g. "Tesla gpu setup" surfaced a real GPU-status turn instead of an
    # unrelated recent one); both no-entity control queries stayed
    # unchanged, confirming no false-positive regression. Re-runnable via
    # scripts/live_verify_entity_relatedness_boost.py.
    #
    # Flipped live 2026-07-19 on the strength of that evidence. Known,
    # disclosed, unproven-live gap: injected candidates don't pass through
    # _suppress_self_hits (which runs before injection in process_recall)
    # -- low risk in practice since entity extraction from a turn is async/
    # post-persist, so the current turn's own entities are not normally in
    # Falkor yet when recall runs for that same turn, but not proven
    # impossible. Worth a follow-up test, not treated as blocking.
    RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED: bool = Field(
        default=False, validation_alias=AliasChoices("RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED")
    )
    RECALL_CRYSTALLIZATION_VECTOR_COLLECTION: str = Field(
        default="orion_memory_crystallizations",
        validation_alias=AliasChoices("RECALL_CRYSTALLIZATION_VECTOR_COLLECTION"),
    )

    # ── RDF / GraphDB (recall-specific) ───────────────────────────────
    RECALL_RDF_ENDPOINT_URL: Optional[str] = Field(default=None, validation_alias=AliasChoices("RECALL_RDF_ENDPOINT_URL"))
    RECALL_RDF_QUERY_URL: Optional[str] = Field(default=None, validation_alias=AliasChoices("RECALL_RDF_QUERY_URL"))
    RECALL_RDF_TIMEOUT_SEC: float = Field(default=5.0, validation_alias=AliasChoices("RECALL_RDF_TIMEOUT_SEC"))
    RECALL_RDF_USER: str = Field(default="admin", validation_alias=AliasChoices("RECALL_RDF_USER", "GRAPHDB_USER"))
    RECALL_RDF_PASS: str = Field(default="admin", validation_alias=AliasChoices("RECALL_RDF_PASS", "GRAPHDB_PASS"))
    RECALL_RDF_ENABLE_SUMMARIES: bool = Field(
        default=False, validation_alias=AliasChoices("RECALL_RDF_ENABLE_SUMMARIES")
    )

    # ── SQL timeline knobs ────────────────────────────────────────────
    RECALL_ENABLE_SQL_TIMELINE: bool = Field(default=True, validation_alias=AliasChoices("RECALL_ENABLE_SQL_TIMELINE"))
    RECALL_SQL_SINCE_MINUTES: int = Field(default=180, validation_alias=AliasChoices("RECALL_SQL_SINCE_MINUTES"))
    RECALL_SQL_TOP_K: int = Field(default=10, validation_alias=AliasChoices("RECALL_SQL_TOP_K"))
    # Default timeline source is chat history. When RECALL_SQL_TIMELINE_TABLE == RECALL_SQL_CHAT_TABLE,
    # sql_timeline uses RECALL_SQL_CHAT_* columns to build "User/Orion" turns and ignores TIMELINE_* columns.
    # When RECALL_SQL_TIMELINE_TABLE is another table (e.g. collapse_mirror), TIMELINE_* columns are used.
    RECALL_SQL_TIMELINE_TABLE: str = Field(default="chat_history_log", validation_alias=AliasChoices("RECALL_SQL_TIMELINE_TABLE"))
    RECALL_SQL_TIMELINE_TS_COL: str = Field(default="timestamp", validation_alias=AliasChoices("RECALL_SQL_TIMELINE_TS_COL"))
    RECALL_SQL_TIMELINE_TEXT_COL: str = Field(default="summary", validation_alias=AliasChoices("RECALL_SQL_TIMELINE_TEXT_COL"))
    RECALL_SQL_TIMELINE_SESSION_COL: str = Field(default="observer", validation_alias=AliasChoices("RECALL_SQL_TIMELINE_SESSION_COL"))
    RECALL_SQL_TIMELINE_NODE_COL: str = Field(default="observer_state", validation_alias=AliasChoices("RECALL_SQL_TIMELINE_NODE_COL"))
    RECALL_SQL_TIMELINE_TAGS_COL: str = Field(default="tags", validation_alias=AliasChoices("RECALL_SQL_TIMELINE_TAGS_COL"))
    # Juniper observer filter is intended for collapse_mirror timelines only.
    RECALL_SQL_TIMELINE_REQUIRE_JUNIPER_OBSERVER: Optional[bool] = Field(
        default=None, validation_alias=AliasChoices("RECALL_SQL_TIMELINE_REQUIRE_JUNIPER_OBSERVER")
    )

    # ── SQL message table (optional) ──────────────────────────────────
    RECALL_SQL_MESSAGE_TABLE: str = Field(default="", validation_alias=AliasChoices("RECALL_SQL_MESSAGE_TABLE"))
    RECALL_SQL_MESSAGE_ROLE_COL: str = Field(
        default="role", validation_alias=AliasChoices("RECALL_SQL_MESSAGE_ROLE_COL")
    )
    RECALL_SQL_MESSAGE_TEXT_COL: str = Field(
        default="text", validation_alias=AliasChoices("RECALL_SQL_MESSAGE_TEXT_COL")
    )
    RECALL_SQL_MESSAGE_CREATED_AT_COL: str = Field(
        default="created_at", validation_alias=AliasChoices("RECALL_SQL_MESSAGE_CREATED_AT_COL")
    )

    # ── Future tensor / ranker toggles ────────────────────────────────
    RECALL_TENSOR_RANKER_ENABLED: bool = Field(default=False, validation_alias=AliasChoices("RECALL_TENSOR_RANKER_ENABLED"))
    RECALL_TENSOR_RANKER_MODEL_PATH: str = Field(
        default="/mnt/storage-warm/orion/recall/tensor-ranker.pt",
        validation_alias=AliasChoices("RECALL_TENSOR_RANKER_MODEL_PATH"),
    )

    # ── Graph Compression backend ─────────────────────────────────────
    RECALL_COMPRESSION_ENABLED: bool = Field(
        default=False, validation_alias=AliasChoices("RECALL_COMPRESSION_ENABLED")
    )
    RECALL_COMPRESSION_PG_DSN: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("RECALL_COMPRESSION_PG_DSN")
    )
    RECALL_COMPRESSION_RDF_QUERY_URL: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("RECALL_COMPRESSION_RDF_QUERY_URL")
    )
    RECALL_COMPRESSION_RDF_USER: str = Field(
        default="admin", validation_alias=AliasChoices("RECALL_COMPRESSION_RDF_USER")
    )
    RECALL_COMPRESSION_RDF_PASS: str = Field(
        default="orion", validation_alias=AliasChoices("RECALL_COMPRESSION_RDF_PASS")
    )
    RECALL_COMPRESSION_TIMEOUT_SEC: float = Field(
        default=3.0, validation_alias=AliasChoices("RECALL_COMPRESSION_TIMEOUT_SEC")
    )

    # ── Helpers ───────────────────────────────────────────────────────
    @field_validator(
        "RECALL_VECTOR_BASE_URL",
        "RECALL_VECTOR_COLLECTIONS",
        "RECALL_VECTOR_EMBEDDING_URL",
        "RECALL_CARDS_EMBEDDING_URL",
        "RECALL_RDF_ENDPOINT_URL",
        "RECALL_RDF_QUERY_URL",
        "RECALL_COMPRESSION_PG_DSN",
        "RECALL_COMPRESSION_RDF_QUERY_URL",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, v):
        if v is None:
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @model_validator(mode="after")
    def _derive_endpoints(self):
        # If recall-specific base URL is not set, build from VECTOR_DB_HOST/PORT
        if not self.RECALL_VECTOR_BASE_URL:
            self.RECALL_VECTOR_BASE_URL = f"http://{self.VECTOR_DB_HOST}:{self.VECTOR_DB_PORT}"

        # If recall-specific collections are not set, fall back to the chat collection
        if not self.RECALL_VECTOR_COLLECTIONS:
            self.RECALL_VECTOR_COLLECTIONS = "orion_chat_turns,orion_chat"

        # Default Juniper filter only for collapse_mirror timelines.
        if self.RECALL_SQL_TIMELINE_REQUIRE_JUNIPER_OBSERVER is None:
            self.RECALL_SQL_TIMELINE_REQUIRE_JUNIPER_OBSERVER = (
                self.RECALL_SQL_TIMELINE_TABLE == "collapse_mirror"
            )

        if not self.RECALL_RDF_ENDPOINT_URL and self.RECALL_RDF_QUERY_URL:
            self.RECALL_RDF_ENDPOINT_URL = str(self.RECALL_RDF_QUERY_URL).strip()

        # RDF query endpoint: explicit recall URL, RECALL_RDF_QUERY_URL, RDF_STORE_QUERY_URL, or legacy GraphDB
        # only when GRAPH_BACKEND=graphdb (never implicit GraphDB from GRAPHDB_URL alone).
        if not self.RECALL_RDF_ENDPOINT_URL:
            q = (self.RECALL_RDF_QUERY_URL or "").strip()
            if not q:
                q = (os.getenv("RDF_STORE_QUERY_URL") or "").strip()
            if q:
                self.RECALL_RDF_ENDPOINT_URL = q
            elif (os.getenv("GRAPH_BACKEND") or "").strip().lower() == "graphdb" and (self.GRAPHDB_URL or "").strip():
                base = self.GRAPHDB_URL.rstrip("/")
                self.RECALL_RDF_ENDPOINT_URL = f"{base}/repositories/{self.GRAPHDB_REPO}"

        if not self.RECALL_CARDS_EMBEDDING_URL and self.RECALL_VECTOR_EMBEDDING_URL:
            self.RECALL_CARDS_EMBEDDING_URL = str(self.RECALL_VECTOR_EMBEDDING_URL).strip()

        return self


settings = Settings()
