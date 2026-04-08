from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, List

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings

from orion.spark.concept_induction.settings import DEFAULT_CONCEPT_STORE_PATH


class OrchConceptProfileSettings(BaseSettings):
    """Concept-profile repository settings used by Orch runtime.

    This adapter intentionally models only the settings required at the
    concept-profile repository seam.
    """

    store_path: str = Field(DEFAULT_CONCEPT_STORE_PATH, alias="CONCEPT_STORE_PATH")
    subjects: List[str] | str = Field(
        default_factory=lambda: ["orion", "juniper", "relationship"],
        alias="CONCEPT_SUBJECTS",
    )

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

    concept_profile_parity_min_comparisons: int = Field(50, alias="CONCEPT_PROFILE_PARITY_MIN_COMPARISONS")
    concept_profile_parity_max_mismatch_rate: float = Field(0.05, alias="CONCEPT_PROFILE_PARITY_MAX_MISMATCH_RATE")
    concept_profile_parity_max_unavailable_rate: float = Field(
        0.02,
        alias="CONCEPT_PROFILE_PARITY_MAX_UNAVAILABLE_RATE",
    )
    concept_profile_parity_critical_mismatch_classes: str = Field(
        "profile_missing_on_graph,profile_missing_on_local,query_error",
        alias="CONCEPT_PROFILE_PARITY_CRITICAL_MISMATCH_CLASSES",
    )
    concept_profile_parity_summary_interval: int = Field(25, alias="CONCEPT_PROFILE_PARITY_SUMMARY_INTERVAL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @field_validator("subjects", mode="before")
    @classmethod
    def _parse_subjects(cls, value: Any) -> list[str]:
        if value is None:
            return ["orion", "juniper", "relationship"]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return ["orion", "juniper", "relationship"]
            if raw.startswith("["):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
                except json.JSONDecodeError:
                    pass
            return [item.strip() for item in raw.split(",") if item.strip()]
        return ["orion", "juniper", "relationship"]

    @field_validator("concept_profile_repository_backend", mode="before")
    @classmethod
    def _parse_repository_backend(cls, value: Any) -> str:
        raw = str(value or "local").strip().lower()
        if raw not in {"local", "graph", "shadow"}:
            return "local"
        return raw

    @field_validator("concept_profile_backend_concept_induction_pass", mode="before")
    @classmethod
    def _parse_concept_induction_backend_override(cls, value: Any) -> str:
        raw = str(value or "").strip().lower()
        if not raw:
            return ""
        if raw not in {"local", "graph", "shadow"}:
            return ""
        return raw

    @field_validator("concept_profile_graph_cutover_fallback_policy", mode="before")
    @classmethod
    def _parse_cutover_fallback_policy(cls, value: Any) -> str:
        raw = str(value or "fail_open_local").strip().lower()
        if raw not in {"fail_open_local", "fail_closed"}:
            return "fail_open_local"
        return raw

    @field_validator("concept_profile_graphdb_endpoint", mode="after")
    @classmethod
    def _resolve_graph_endpoint(cls, value: str, info) -> str:
        if value:
            return value.rstrip("/")
        base = str(info.data.get("concept_profile_graphdb_url") or "").strip()
        repo = str(info.data.get("concept_profile_graphdb_repo") or "").strip()
        if not base:
            return ""
        if not repo:
            repo = "collapse"
        return f"{base.rstrip('/')}/repositories/{repo}"


@lru_cache(maxsize=1)
def get_orch_concept_profile_settings() -> OrchConceptProfileSettings:
    return OrchConceptProfileSettings()


def build_orch_concept_profile_settings(orch_settings: Any | None = None) -> OrchConceptProfileSettings:
    """Return concept-profile repository config for Orch runtime.

    Environment is the source of truth. Optional orch_settings values are used
    only when they intentionally expose seam-owned concept-profile fields.
    """

    cfg = get_orch_concept_profile_settings()
    updates: dict[str, Any] = {}
    if orch_settings is not None:
        for field in OrchConceptProfileSettings.model_fields:
            if hasattr(orch_settings, field):
                updates[field] = getattr(orch_settings, field)
    resolved = cfg.model_copy(update=updates) if updates else cfg
    if not resolved.concept_profile_graphdb_endpoint and resolved.concept_profile_graphdb_url:
        repo = (resolved.concept_profile_graphdb_repo or "collapse").strip() or "collapse"
        endpoint = f"{resolved.concept_profile_graphdb_url.rstrip('/')}/repositories/{repo}"
        resolved = resolved.model_copy(update={"concept_profile_graphdb_endpoint": endpoint})
    return resolved
