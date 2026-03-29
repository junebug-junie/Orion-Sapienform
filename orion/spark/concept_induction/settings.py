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
            "orion:chat:social:stored",
            "orion:collapse:sql-write",
            "orion:spark:telemetry",
            "orion:metacognition:tick",
            "orion:cognition:trace",
        ],
        validation_alias=AliasChoices("BUS_INTAKE_CHANNELS", "CONCEPT_INTAKE_CHANNELS"),
    )
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
    goal_proposal_cooldown_minutes: int = Field(180, alias="GOAL_PROPOSAL_COOLDOWN_MINUTES")

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
        validation_alias=AliasChoices("CONCEPT_PROFILE_GRAPHDB_ENDPOINT", "RECALL_RDF_ENDPOINT_URL"),
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
