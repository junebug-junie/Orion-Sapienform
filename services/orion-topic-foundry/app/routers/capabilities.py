from __future__ import annotations

from fastapi import APIRouter

from app.models import CapabilitiesResponse, ModelSpec
from app.services.metrics import SUPPORTED_METRICS
from app.settings import settings


router = APIRouter()


@router.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities() -> CapabilitiesResponse:
    segmentation_modes = ["document", "time_gap", "conversation_bound"]
    enricher_modes = ["heuristic", "llm"]
    llm_transport = "bus" if settings.topic_foundry_llm_use_bus and settings.orion_bus_enabled else "http"
    schemas = [s.strip() for s in settings.topic_foundry_introspect_schemas.split(",") if s.strip()]
    defaults = {
        "embedding_source_url": settings.topic_foundry_embedding_url,
        "metric": ModelSpec.model_fields["metric"].default,
        "min_cluster_size": ModelSpec.model_fields["min_cluster_size"].default,
        "llm_bus_route": settings.topic_foundry_llm_bus_route,
        "topic_engine": settings.topic_foundry_topic_engine,
        "embedding_backend": settings.topic_foundry_embedding_backend,
        "embedding_model": settings.topic_foundry_embedding_model,
        "reducer": settings.topic_foundry_reducer,
        "clusterer": settings.topic_foundry_clusterer,
        "representation": settings.topic_foundry_representation,
    }
    return CapabilitiesResponse(
        service=settings.service_name,
        version=settings.service_version,
        node=settings.node_name,
        llm_enabled=settings.topic_foundry_llm_enable,
        llm_transport=llm_transport,
        llm_bus_route=settings.topic_foundry_llm_bus_route,
        llm_intake_channel=settings.topic_foundry_llm_intake_channel if llm_transport == "bus" else None,
        llm_reply_prefix=settings.topic_foundry_llm_reply_prefix if llm_transport == "bus" else None,
        segmentation_modes_supported=segmentation_modes,
        enricher_modes_supported=enricher_modes,
        supported_metrics=sorted(SUPPORTED_METRICS),
        default_metric=ModelSpec.model_fields["metric"].default,
        cosine_impl_default=settings.topic_foundry_cosine_impl,
        defaults=defaults,
        introspection={"ok": bool(schemas), "schemas": schemas},
        default_embedding_url=settings.topic_foundry_embedding_url,
        capabilities={
            "topic_modeling": {
                "class_based": settings.topic_foundry_enable_class_based,
                "long_document": settings.topic_foundry_enable_long_document,
                "hierarchical": settings.topic_foundry_enable_hierarchical,
                "dynamic": settings.topic_foundry_enable_dynamic,
                "guided": settings.topic_foundry_enable_guided,
                "zeroshot": settings.topic_foundry_enable_zeroshot,
            }
        },
        backends={
            "embedding_backends": ["vector_host", "st"],
            "reducers": ["umap"],
            "clusterers": ["hdbscan"],
            "representations": ["ctfidf", "keybert", "mmr", "pos", "llm"],
        },
    )
