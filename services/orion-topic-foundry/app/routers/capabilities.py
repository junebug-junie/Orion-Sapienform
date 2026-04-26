from __future__ import annotations

from fastapi import APIRouter

from app.models import CapabilitiesResponse, ModelSpec, WindowingSpec
from app.services.metrics import SUPPORTED_METRICS
from app.topic_engine import _build_vectorizer
from app.settings import settings


router = APIRouter()


@router.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities() -> CapabilitiesResponse:
    segmentation_modes = list(WindowingSpec.model_fields["segmentation_mode"].annotation.__args__)
    enricher_modes = ["heuristic", "llm"]
    llm_transport = "bus" if settings.topic_foundry_llm_use_bus and settings.orion_bus_enabled else "http"
    vectorizer = _build_vectorizer({})
    vectorizer_params = vectorizer.get_params()
    stop_words_mode = str(settings.topic_foundry_vectorizer_stop_words).strip().lower() or "none"
    stop_words_extra = [x.strip().lower() for x in str(settings.topic_foundry_stop_words_extra or "").split(",") if x.strip()]
    defaults = {
        "embedding_source_url": settings.topic_foundry_embedding_url,
        "metric": ModelSpec.model_fields["metric"].default,
        "min_cluster_size": ModelSpec.model_fields["min_cluster_size"].default,
        "llm_bus_route": settings.topic_foundry_llm_bus_route,
        "vectorizer": {
            "stop_words": stop_words_mode,
            "max_df": float(vectorizer_params.get("max_df", settings.topic_foundry_vectorizer_max_df)),
            "min_df": int(vectorizer_params.get("min_df", settings.topic_foundry_vectorizer_min_df)),
            "max_features": int(vectorizer_params.get("max_features", settings.topic_foundry_vectorizer_max_features)),
            "ngram_range": [
                int((vectorizer_params.get("ngram_range") or (settings.topic_foundry_vectorizer_ngram_min, settings.topic_foundry_vectorizer_ngram_max))[0]),
                int((vectorizer_params.get("ngram_range") or (settings.topic_foundry_vectorizer_ngram_min, settings.topic_foundry_vectorizer_ngram_max))[1]),
            ],
            "token_pattern": str(vectorizer_params.get("token_pattern") or ""),
            "stop_words_extra": stop_words_extra,
        },
        "ctfidf": {"reduce_frequent_words": True},
    }
    capabilities_block = {
        "topic_modeling": {
            "class_based": bool(settings.topic_foundry_enable_class_based),
            "long_document": bool(settings.topic_foundry_enable_long_document),
            "hierarchical": bool(settings.topic_foundry_enable_hierarchical),
            "dynamic": bool(settings.topic_foundry_enable_dynamic),
            "guided": bool(settings.topic_foundry_enable_guided),
            "zeroshot": bool(settings.topic_foundry_enable_zeroshot),
        }
    }
    backends = {
        "embedding_backends": ["vector_host", "st"],
        "reducers": ["umap"],
        "clusterers": ["hdbscan"],
        "representations": ["ctfidf", "keybert", "mmr", "pos", "llm"],
    }
    return CapabilitiesResponse(
        service=settings.service_name,
        version=settings.service_version,
        node=settings.node_name,
        capabilities=capabilities_block,
        backends=backends,
        supported_metrics=sorted(str(metric) for metric in SUPPORTED_METRICS),
        default_metric=str(settings.topic_foundry_hdbscan_metric),
        llm_enabled=settings.topic_foundry_llm_enable,
        llm_transport=llm_transport,
        llm_bus_route=settings.topic_foundry_llm_bus_route,
        llm_intake_channel=settings.topic_foundry_llm_intake_channel if llm_transport == "bus" else None,
        llm_reply_prefix=settings.topic_foundry_llm_reply_prefix if llm_transport == "bus" else None,
        segmentation_modes_supported=segmentation_modes,
        enricher_modes_supported=enricher_modes,
        defaults=defaults,
    )
