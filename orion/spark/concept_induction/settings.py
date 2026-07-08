from __future__ import annotations

from functools import lru_cache
import json
from typing import List

from pydantic import Field, AliasChoices, field_validator
from pydantic_settings import BaseSettings

DEFAULT_CONCEPT_STORE_PATH = "/tmp/concept-induction-state.json"


class ConceptSettings(BaseSettings):
    """Environment-driven settings for Concept Induction."""

    # Identity
    service_name: str = Field("orion-spark-concept-induction", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")

    # Bus
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_enforce_catalog: bool = Field(False, alias="ORION_BUS_ENFORCE_CATALOG")

    # Channels
    intake_channels: List[str] = Field(
        default_factory=lambda: [
            "orion:chat:history:log",
            "orion:chat:history:turn",
            "orion:chat:social:turn",
            "orion:chat:social:stored",
            "orion:chat:gpt:turn",
            "orion:chat:gpt:message:log",
            "orion:collapse:sql-write",
            "orion:spark:telemetry",
            "orion:metacognition:tick",
            "orion:cognition:trace",
            "orion:substrate:self_state",
            "orion:feedback:frame",
            "orion:world_pulse:run:result",
        ],
        validation_alias=AliasChoices("BUS_INTAKE_CHANNELS", "CONCEPT_INTAKE_CHANNELS"),
    )
    concept_autonomous_trigger_enabled: bool = Field(True, alias="CONCEPT_AUTONOMOUS_TRIGGER_ENABLED")
    profile_channel: str = Field(
        "orion:spark:concepts:profile",
        alias="BUS_PROFILE_OUT",
    )
    delta_channel: str = Field(
        "orion:spark:concepts:delta",
        alias="BUS_DELTA_OUT",
    )
    forward_vector_channel: str = Field(
        "orion:vector:write",
        alias="BUS_VECTOR_OUT",
    )
    forward_rdf_channel: str = Field(
        "orion:rdf:enqueue",
        alias="BUS_RDF_OUT",
    )
    forward_sql_channel: str = Field(
        "orion:collapse:sql-write",
        alias="BUS_SQL_OUT",
    )
    drive_state_channel: str = Field(
        "orion:memory:drives:state",
        alias="BUS_DRIVE_STATE_OUT",
    )
    tension_event_channel: str = Field(
        "orion:memory:tension:event",
        alias="BUS_TENSION_EVENT_OUT",
    )
    drive_audit_channel: str = Field(
        "orion:memory:drives:audit",
        alias="BUS_DRIVE_AUDIT_OUT",
    )
    identity_snapshot_channel: str = Field(
        "orion:memory:identity:snapshot",
        alias="BUS_IDENTITY_SNAPSHOT_OUT",
    )
    goal_proposal_channel: str = Field(
        "orion:memory:goals:proposed",
        alias="BUS_GOAL_PROPOSAL_OUT",
    )
    turn_dossier_channel: str = Field(
        "orion:debug:turn:dossier",
        alias="BUS_TURN_DOSSIER_OUT",
    )
    action_outcome_channel: str = Field(
        "orion:autonomy:action:outcome",
        alias="BUS_ACTION_OUTCOME_OUT",
    )

    # Windowing
    window_max_events: int = Field(200, alias="CONCEPT_WINDOW_MAX_EVENTS")
    window_max_minutes: int = Field(360, alias="CONCEPT_WINDOW_MAX_MINUTES")
    concept_trigger_cooldown_sec: int = Field(300, alias="CONCEPT_TRIGGER_COOLDOWN_SEC")
    concept_trigger_dedupe_sec: int = Field(45, alias="CONCEPT_TRIGGER_DEDUPE_SEC")
    concept_trigger_recent_decisions: int = Field(200, alias="CONCEPT_TRIGGER_RECENT_DECISIONS")
    subjects: List[str] | str = Field(
        default_factory=lambda: ["orion", "juniper", "relationship"],
        alias="CONCEPT_SUBJECTS",
    )

    # Extraction / Embedding / Clustering
    spacy_model: str = Field("en_core_web_sm", alias="SPACY_MODEL")
    max_candidates: int = Field(50, alias="CONCEPT_MAX_CANDIDATES")
    embedding_base_url: str = Field(
        "http://orion-athena-vector-host:8320", alias="EMBEDDINGS_BASE_URL"
    )
    embedding_timeout_sec: float = Field(5.0, alias="EMBEDDINGS_TIMEOUT_SEC")
    cluster_cosine_threshold: float = Field(0.8, alias="CONCEPT_CLUSTER_THRESHOLD")

    # Optional Cortex-Orch override
    use_cortex_orch: bool = Field(False, alias="USE_CORTEX_ORCH")
    cortex_orch_verb: str = Field("concept_induction", alias="CORTEX_ORCH_VERB")
    cortex_request_channel: str = Field(
        "orion:cortex:request", alias="CORTEX_ORCH_REQUEST_CHANNEL"
    )
    cortex_result_prefix: str = Field(
        "orion:cortex:result", alias="CORTEX_ORCH_RESULT_PREFIX"
    )
    cortex_timeout_sec: float = Field(12.0, alias="CORTEX_TIMEOUT_SEC")

    # Storage / persistence
    store_path: str = Field(
        DEFAULT_CONCEPT_STORE_PATH, alias="CONCEPT_STORE_PATH"
    )
    drive_decay_tau_sec: float = Field(1800.0, alias="DRIVE_DECAY_TAU_SEC")
    drive_saturation_gain: float = Field(1.8, alias="DRIVE_SATURATION_GAIN")
    drive_activation_on: float = Field(0.62, alias="DRIVE_ACTIVATION_ON")
    drive_activation_off: float = Field(0.42, alias="DRIVE_ACTIVATION_OFF")
    # Homeostatic drives (spec 2026-07-07). Leaky math replaces the soft-saturate
    # fixed point (~0.731 pin); the source path mints deviation-triggered tensions
    # from the signal/failure/health bus. Both default true (Juniper-authorized).
    drive_leaky_math_enabled: bool = Field(True, alias="ORION_DRIVE_LEAKY_MATH_ENABLED")
    homeostatic_drives_enabled: bool = Field(True, alias="ORION_HOMEOSTATIC_DRIVES_ENABLED")
    deviation_ewma_alpha: float = Field(0.1, alias="DEVIATION_EWMA_ALPHA")
    deviation_z_threshold: float = Field(1.5, alias="DEVIATION_Z_THRESHOLD")
    deviation_sigma_floor: float = Field(0.02, alias="DEVIATION_SIGMA_FLOOR")
    signal_tension_impulse_k: float = Field(0.25, alias="SIGNAL_TENSION_IMPULSE_K")
    signal_tension_cap_per_window: int = Field(3, alias="SIGNAL_TENSION_CAP_PER_WINDOW")
    signal_tension_window_sec: int = Field(60, alias="SIGNAL_TENSION_WINDOW_SEC")
    # Homeostatic consumer subscription: SPECIFIC organ/failure channels only —
    # never the orion:signals:* wildcard, so the 55/s scene_state flood is
    # excluded at the subscription. These route to a drive-only tick that does
    # NOT trigger concept induction.
    homeostatic_signal_channels: List[str] = Field(
        default_factory=lambda: [
            "orion:signals:biometrics",
            "orion:signals:spark",
            "orion:signals:equilibrium",
        ],
        validation_alias=AliasChoices("HOMEOSTATIC_SIGNAL_CHANNELS"),
    )
    homeostatic_failure_channels: List[str] = Field(
        default_factory=lambda: [
            "orion:system:error",
            "orion:rdf:error",
            "orion:vision:edge:error",
        ],
        validation_alias=AliasChoices("HOMEOSTATIC_FAILURE_CHANNELS"),
    )
    homeostatic_failure_severity: float = Field(0.8, alias="HOMEOSTATIC_FAILURE_SEVERITY")
    # Endogenous drive origination (spec 2026-07-07, Step 1). DEFAULT-OFF: this
    # changes what causes Orion to *want* things (cognition-loop change) and is
    # gated on measurement 0(a). Proposal mode — enable only after sign-off.
    endogenous_origination_enabled: bool = Field(False, alias="ORION_ENDOGENOUS_ORIGINATION_ENABLED")
    origination_window: int = Field(8, alias="ORIGINATION_WINDOW")
    origination_threshold: float = Field(0.55, alias="ORIGINATION_THRESHOLD")
    origination_cooldown_sec: float = Field(900.0, alias="ORIGINATION_COOLDOWN_SEC")
    endogenous_mag_cap: float = Field(0.5, alias="ENDOGENOUS_MAG_CAP")
    origination_w_drift: float = Field(0.4, alias="ORIGINATION_W_DRIFT")
    origination_w_dwell: float = Field(0.35, alias="ORIGINATION_W_DWELL")
    origination_w_agency: float = Field(0.25, alias="ORIGINATION_W_AGENCY")
    origination_exogenous_floor: int = Field(0, alias="ORIGINATION_EXOGENOUS_FLOOR")
    goal_proposal_cooldown_minutes: int = Field(180, alias="GOAL_PROPOSAL_COOLDOWN_MINUTES")
    goal_generation_mode: str = Field("evidence_rules", alias="GOAL_GENERATION_MODE")
    goal_drive_origin_source: str = Field("tick_attribution", alias="GOAL_DRIVE_ORIGIN_SOURCE")
    substrate_autonomy_metabolism_enabled: bool = Field(
        False,
        alias="ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED",
    )
    autonomy_episode_journal_enabled: bool = Field(
        False,
        alias="ORION_AUTONOMY_EPISODE_JOURNAL_ENABLED",
    )
    # Journal compose is an LLM RPC to cortex-exec (~16s observed). The generic
    # cortex_timeout_sec (12s) is too tight and silently drops the episode journal,
    # so give the compose its own generous budget.
    autonomy_episode_journal_timeout_sec: float = Field(
        120.0,
        alias="ORION_AUTONOMY_EPISODE_JOURNAL_TIMEOUT_SEC",
    )
    # Durable world-pulse run-result consumption (Redis Stream + consumer group).
    # When enabled, orion:world_pulse:run:result is consumed from the durable stream
    # instead of pub/sub so a busy/restarting worker never loses the episode trigger.
    wp_run_result_stream_enabled: bool = Field(
        False,
        alias="WP_RUN_RESULT_STREAM_ENABLED",
    )
    wp_run_result_stream_key: str = Field(
        "orion:stream:world_pulse:run:result",
        alias="WP_RUN_RESULT_STREAM_KEY",
    )
    wp_run_result_stream_group: str = Field(
        "cg:concept-induction",
        alias="WP_RUN_RESULT_STREAM_GROUP",
    )
    wp_run_result_dlq_key: str = Field(
        "orion:stream:world_pulse:run:result:dlq",
        alias="WP_RUN_RESULT_DLQ_KEY",
    )
    wp_run_result_block_ms: int = Field(5000, alias="WP_RUN_RESULT_BLOCK_MS")
    wp_run_result_max_attempts: int = Field(5, alias="WP_RUN_RESULT_MAX_ATTEMPTS")
    wp_run_result_autoclaim_idle_ms: int = Field(
        120000,
        alias="WP_RUN_RESULT_AUTOCLAIM_IDLE_MS",
    )
    episode_fetch_backend: str = Field("auto", alias="ORION_EPISODE_FETCH_BACKEND")
    orion_fcc_env_path: str = Field("~/.fcc/.env", alias="ORION_FCC_ENV_PATH")
    firecrawl_api_key: str = Field("", alias="FIRECRAWL_API_KEY")
    journal_write_channel: str = Field("orion:journal:write", alias="JOURNAL_WRITE_CHANNEL")
    journal_session_id: str = Field("orion", alias="JOURNAL_SESSION_ID")
    journal_user_id: str = Field("juniper", alias="JOURNAL_USER_ID")
    journal_author: str = Field("orion", alias="JOURNAL_AUTHOR")

    # Repository backend / Graph read model (Phase 2)
    concept_profile_repository_backend: str = Field(
        "local",
        alias="CONCEPT_PROFILE_REPOSITORY_BACKEND",
    )
    concept_profile_backend_concept_induction_pass: str = Field(
        "",
        alias="CONCEPT_PROFILE_BACKEND_CONCEPT_INDUCTION_PASS",
    )
    concept_profile_graph_cutover_fallback_policy: str = Field(
        "fail_open_local",
        alias="CONCEPT_PROFILE_GRAPH_CUTOVER_FALLBACK_POLICY",
    )
    concept_profile_graphdb_endpoint: str = Field(
        "",
        validation_alias=AliasChoices(
            "CONCEPT_PROFILE_GRAPHDB_ENDPOINT",
            "RECALL_RDF_ENDPOINT_URL",
            "RECALL_RDF_QUERY_URL",
            "RDF_STORE_QUERY_URL",
        ),
    )
    concept_profile_graphdb_url: str = Field(
        "",
        validation_alias=AliasChoices("CONCEPT_PROFILE_GRAPHDB_URL", "GRAPHDB_URL"),
    )
    concept_profile_graphdb_repo: str = Field(
        "collapse",
        validation_alias=AliasChoices("CONCEPT_PROFILE_GRAPHDB_REPO", "GRAPHDB_REPO"),
    )
    concept_profile_graphdb_user: str = Field(
        "",
        validation_alias=AliasChoices("CONCEPT_PROFILE_GRAPHDB_USER", "GRAPHDB_USER"),
    )
    concept_profile_graphdb_pass: str = Field(
        "",
        validation_alias=AliasChoices("CONCEPT_PROFILE_GRAPHDB_PASS", "GRAPHDB_PASS"),
    )
    concept_profile_graph_timeout_sec: float = Field(6.0, alias="CONCEPT_PROFILE_GRAPH_TIMEOUT_SEC")
    concept_profile_graph_uri: str = Field(
        "http://conjourney.net/graph/spark/concept-profile",
        alias="CONCEPT_PROFILE_GRAPH_URI",
    )
    concept_profile_parity_min_comparisons: int = Field(
        50,
        alias="CONCEPT_PROFILE_PARITY_MIN_COMPARISONS",
    )
    concept_profile_parity_max_mismatch_rate: float = Field(
        0.05,
        alias="CONCEPT_PROFILE_PARITY_MAX_MISMATCH_RATE",
    )
    concept_profile_parity_max_unavailable_rate: float = Field(
        0.02,
        alias="CONCEPT_PROFILE_PARITY_MAX_UNAVAILABLE_RATE",
    )
    concept_profile_parity_critical_mismatch_classes: str = Field(
        "profile_missing_on_graph,profile_missing_on_local,query_error",
        alias="CONCEPT_PROFILE_PARITY_CRITICAL_MISMATCH_CLASSES",
    )
    concept_profile_parity_summary_interval: int = Field(
        25,
        alias="CONCEPT_PROFILE_PARITY_SUMMARY_INTERVAL",
    )

    # Heartbeat
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")

    # Logging
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        populate_by_name = True


    @field_validator("subjects", mode="before")
    @classmethod
    def _parse_subjects(cls, v):
        if v is None:
            return None
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            raw = v.strip()
            if not raw:
                return None
            if raw.startswith("["):
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    pass
            return [item.strip() for item in raw.split(",") if item.strip()]
        return v

    @field_validator("concept_profile_repository_backend", mode="before")
    @classmethod
    def _parse_repository_backend(cls, v):
        raw = str(v or "local").strip().lower()
        if raw not in {"local", "graph", "shadow"}:
            return "local"
        return raw

    @field_validator("concept_profile_backend_concept_induction_pass", mode="before")
    @classmethod
    def _parse_concept_induction_backend_override(cls, v):
        raw = str(v or "").strip().lower()
        if not raw:
            return ""
        if raw not in {"local", "graph", "shadow"}:
            return ""
        return raw

    @field_validator("concept_profile_graph_cutover_fallback_policy", mode="before")
    @classmethod
    def _parse_cutover_fallback_policy(cls, v):
        raw = str(v or "fail_open_local").strip().lower()
        if raw not in {"fail_open_local", "fail_closed"}:
            return "fail_open_local"
        return raw

    @field_validator("concept_profile_graphdb_endpoint", mode="after")
    @classmethod
    def _resolve_graph_endpoint(cls, v, info):
        if v:
            return v.rstrip("/")
        base = str(info.data.get("concept_profile_graphdb_url") or "").strip()
        repo = str(info.data.get("concept_profile_graphdb_repo") or "").strip()
        if not base:
            return ""
        if not repo:
            repo = "collapse"
        return f"{base.rstrip('/')}/repositories/{repo}"



@lru_cache(maxsize=1)
def get_settings() -> ConceptSettings:
    return ConceptSettings()


settings = get_settings()
