from __future__ import annotations

from functools import lru_cache
import json
from typing import List

from pydantic import Field, AliasChoices, field_validator
from pydantic_settings import BaseSettings


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
            "orion:collapse:mirror",
            "orion:memory:episode",
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
        "orion:rdf:write",
        alias="BUS_RDF_OUT",
    )
    forward_sql_channel: str = Field(
        "orion:collapse:sql-write",
        alias="BUS_SQL_OUT",
    )

    # Windowing
    window_max_events: int = Field(200, alias="CONCEPT_WINDOW_MAX_EVENTS")
    window_max_minutes: int = Field(360, alias="CONCEPT_WINDOW_MAX_MINUTES")
    subjects: List[str] | str = Field(
        default_factory=lambda: ["orion", "juniper", "relationship"],
        alias="CONCEPT_SUBJECTS",
    )

    # Extraction / Embedding / Clustering
    spacy_model: str = Field("en_core_web_sm", alias="SPACY_MODEL")
    max_candidates: int = Field(50, alias="CONCEPT_MAX_CANDIDATES")
    embedding_base_url: str = Field(
        "http://orion-embeddings-host:8000", alias="EMBEDDINGS_BASE_URL"
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
        "/tmp/concept-induction-state.json", alias="CONCEPT_STORE_PATH"
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



@lru_cache(maxsize=1)
def get_settings() -> ConceptSettings:
    return ConceptSettings()


settings = get_settings()
