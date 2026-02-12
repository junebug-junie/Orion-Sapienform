from __future__ import annotations

from typing import Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    service_name: str = Field("orion-topic-foundry", validation_alias=AliasChoices("SERVICE_NAME"))
    service_version: str = Field("0.1.0", validation_alias=AliasChoices("SERVICE_VERSION"))
    node_name: str = Field("unknown", validation_alias=AliasChoices("NODE_NAME", "HOSTNAME"))
    log_level: str = Field("INFO", validation_alias=AliasChoices("LOG_LEVEL"))
    port: int = Field(8615, validation_alias=AliasChoices("PORT"))

    topic_foundry_pg_dsn: str = Field(
        "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney",
        validation_alias=AliasChoices("TOPIC_FOUNDRY_PG_DSN", "POSTGRES_URI", "POSTGRES_DSN"),
    )
    topic_foundry_embedding_url: str = Field(
        "http://orion-vector-host:8320/embedding",
        validation_alias=AliasChoices("TOPIC_FOUNDRY_EMBEDDING_URL"),
    )

    topic_foundry_topic_engine: str = Field(
        "bertopic",
        validation_alias=AliasChoices("TOPIC_FOUNDRY_TOPIC_ENGINE"),
    )
    topic_foundry_embedding_backend: str = Field(
        "vector_host",
        validation_alias=AliasChoices("TOPIC_FOUNDRY_EMBEDDING_BACKEND"),
    )
    topic_foundry_embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        validation_alias=AliasChoices("TOPIC_FOUNDRY_EMBEDDING_MODEL"),
    )
    topic_foundry_vector_host_url: str = Field(
        "http://orion-vector-host:8320/embedding",
        validation_alias=AliasChoices("TOPIC_FOUNDRY_VECTOR_HOST_URL"),
    )
    topic_foundry_embedding_batch_size: int = Field(32, validation_alias=AliasChoices("TOPIC_FOUNDRY_EMBEDDING_BATCH_SIZE"))
    topic_foundry_embedding_max_retries: int = Field(2, validation_alias=AliasChoices("TOPIC_FOUNDRY_EMBEDDING_MAX_RETRIES"))
    topic_foundry_embedding_retry_delay_secs: float = Field(0.75, validation_alias=AliasChoices("TOPIC_FOUNDRY_EMBEDDING_RETRY_DELAY_SECS"))
    topic_foundry_reducer: str = Field("umap", validation_alias=AliasChoices("TOPIC_FOUNDRY_REDUCER"))
    topic_foundry_clusterer: str = Field("hdbscan", validation_alias=AliasChoices("TOPIC_FOUNDRY_CLUSTERER"))
    topic_foundry_representation: str = Field("ctfidf", validation_alias=AliasChoices("TOPIC_FOUNDRY_REPRESENTATION"))
    topic_foundry_random_state: int = Field(42, validation_alias=AliasChoices("TOPIC_FOUNDRY_RANDOM_STATE"))
    topic_foundry_umap_n_neighbors: int = Field(15, validation_alias=AliasChoices("TOPIC_FOUNDRY_UMAP_N_NEIGHBORS"))
    topic_foundry_umap_n_components: int = Field(5, validation_alias=AliasChoices("TOPIC_FOUNDRY_UMAP_N_COMPONENTS"))
    topic_foundry_umap_min_dist: float = Field(0.0, validation_alias=AliasChoices("TOPIC_FOUNDRY_UMAP_MIN_DIST"))
    topic_foundry_umap_metric: str = Field("cosine", validation_alias=AliasChoices("TOPIC_FOUNDRY_UMAP_METRIC"))
    topic_foundry_hdbscan_min_cluster_size: int = Field(15, validation_alias=AliasChoices("TOPIC_FOUNDRY_HDBSCAN_MIN_CLUSTER_SIZE"))
    topic_foundry_hdbscan_min_samples: int = Field(5, validation_alias=AliasChoices("TOPIC_FOUNDRY_HDBSCAN_MIN_SAMPLES"))
    topic_foundry_hdbscan_metric: str = Field("euclidean", validation_alias=AliasChoices("TOPIC_FOUNDRY_HDBSCAN_METRIC"))
    topic_foundry_hdbscan_cluster_selection_method: str = Field("eom", validation_alias=AliasChoices("TOPIC_FOUNDRY_HDBSCAN_CLUSTER_SELECTION_METHOD"))
    topic_foundry_vectorizer_ngram_min: int = Field(1, validation_alias=AliasChoices("TOPIC_FOUNDRY_VECTORIZER_NGRAM_MIN"))
    topic_foundry_vectorizer_ngram_max: int = Field(1, validation_alias=AliasChoices("TOPIC_FOUNDRY_VECTORIZER_NGRAM_MAX"))
    topic_foundry_vectorizer_min_df: int = Field(1, validation_alias=AliasChoices("TOPIC_FOUNDRY_VECTORIZER_MIN_DF"))
    topic_foundry_vectorizer_max_df: float = Field(0.95, validation_alias=AliasChoices("TOPIC_FOUNDRY_VECTORIZER_MAX_DF"))
    topic_foundry_vectorizer_max_features: int = Field(5000, validation_alias=AliasChoices("TOPIC_FOUNDRY_VECTORIZER_MAX_FEATURES"))
    topic_foundry_vectorizer_stop_words: str = Field("english", validation_alias=AliasChoices("TOPIC_FOUNDRY_VECTORIZER_STOP_WORDS"))
    topic_foundry_stop_words_extra: str = Field("", validation_alias=AliasChoices("TOPIC_FOUNDRY_STOP_WORDS_EXTRA"))
    topic_foundry_top_n_words: int = Field(12, validation_alias=AliasChoices("TOPIC_FOUNDRY_TOP_N_WORDS"))
    topic_foundry_enable_class_based: bool = Field(True, validation_alias=AliasChoices("TOPIC_FOUNDRY_ENABLE_CLASS_BASED"))
    topic_foundry_enable_long_document: bool = Field(True, validation_alias=AliasChoices("TOPIC_FOUNDRY_ENABLE_LONG_DOCUMENT"))
    topic_foundry_enable_hierarchical: bool = Field(True, validation_alias=AliasChoices("TOPIC_FOUNDRY_ENABLE_HIERARCHICAL"))
    topic_foundry_enable_dynamic: bool = Field(True, validation_alias=AliasChoices("TOPIC_FOUNDRY_ENABLE_DYNAMIC"))
    topic_foundry_enable_guided: bool = Field(True, validation_alias=AliasChoices("TOPIC_FOUNDRY_ENABLE_GUIDED"))
    topic_foundry_enable_zeroshot: bool = Field(True, validation_alias=AliasChoices("TOPIC_FOUNDRY_ENABLE_ZEROSHOT"))
    topic_foundry_cosine_impl: str = Field(
        "normalize_euclidean",
        validation_alias=AliasChoices("TOPIC_FOUNDRY_COSINE_IMPL"),
    )
    topic_foundry_model_dir: str = Field(
        "/mnt/telemetry/models/topic-foundry",
        validation_alias=AliasChoices("TOPIC_FOUNDRY_MODEL_DIR"),
    )
    topic_foundry_llm_timeout_secs: int = Field(
        60,
        validation_alias=AliasChoices("TOPIC_FOUNDRY_LLM_TIMEOUT_SECS"),
    )
    topic_foundry_llm_max_concurrency: int = Field(
        4,
        validation_alias=AliasChoices("TOPIC_FOUNDRY_LLM_MAX_CONCURRENCY"),
    )
    topic_foundry_llm_use_bus: bool = Field(
        True,
        validation_alias=AliasChoices("TOPIC_FOUNDRY_LLM_USE_BUS"),
    )
    topic_foundry_llm_intake_channel: str = Field(
        "orion:exec:request:LLMGatewayService",
        validation_alias=AliasChoices("TOPIC_FOUNDRY_LLM_INTAKE_CHANNEL", "CHANNEL_LLM_INTAKE"),
    )
    topic_foundry_llm_reply_prefix: str = Field(
        "orion:llm:reply",
        validation_alias=AliasChoices("TOPIC_FOUNDRY_LLM_REPLY_PREFIX"),
    )
    topic_foundry_llm_route: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("TOPIC_FOUNDRY_LLM_BUS_ROUTE"),
    )
    topic_foundry_llm_enable: bool = Field(
        False,
        validation_alias=AliasChoices("TOPIC_FOUNDRY_LLM_ENABLE"),
    )

    orion_bus_enabled: bool = Field(
        True,
        validation_alias=AliasChoices("ORION_BUS_ENABLED"),
    )
    orion_bus_url: str = Field(
        "redis://100.92.216.81:6379/0",
        validation_alias=AliasChoices("ORION_BUS_URL"),
    )

    topic_foundry_drift_daemon: bool = Field(
        False,
        validation_alias=AliasChoices("TOPIC_FOUNDRY_DRIFT_DAEMON"),
    )
    topic_foundry_drift_poll_seconds: int = Field(
        900,
        validation_alias=AliasChoices("TOPIC_FOUNDRY_DRIFT_POLL_SECONDS"),
    )
    topic_foundry_drift_window_hours: int = Field(
        24,
        validation_alias=AliasChoices("TOPIC_FOUNDRY_DRIFT_WINDOW_HOURS"),
    )
    topic_foundry_introspect_schemas: str = Field(
        "public",
        validation_alias=AliasChoices("TOPIC_FOUNDRY_INTROSPECT_SCHEMAS"),
    )
    topic_foundry_introspect_cache_secs: int = Field(
        30,
        validation_alias=AliasChoices("TOPIC_FOUNDRY_INTROSPECT_CACHE_SECS"),
    )
    topic_foundry_introspect_max_tables: int = Field(
        5000,
        validation_alias=AliasChoices("TOPIC_FOUNDRY_INTROSPECT_MAX_TABLES"),
    )
    topic_foundry_introspect_max_columns: int = Field(
        5000,
        validation_alias=AliasChoices("TOPIC_FOUNDRY_INTROSPECT_MAX_COLUMNS"),
    )


settings = Settings()
