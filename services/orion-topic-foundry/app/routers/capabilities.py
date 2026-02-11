from __future__ import annotations

from typing import get_args

from fastapi import APIRouter

from app.models import CapabilitiesResponse, ModelSpec
from app.services.metrics import SUPPORTED_METRICS
from app.settings import settings


router = APIRouter()


@router.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities() -> CapabilitiesResponse:
    windowing_modes = ["document", "time_gap", "conversation_bound"]
    enricher_modes = ["heuristic", "llm"]
    llm_transport = "bus" if settings.topic_foundry_llm_use_bus and settings.orion_bus_enabled else "http"
    schemas = [s.strip() for s in settings.topic_foundry_introspect_schemas.split(",") if s.strip()]
    defaults = {
        "embedding_source_url": settings.topic_foundry_embedding_url,
        "metric": ModelSpec.model_fields["metric"].default,
        "min_cluster_size": ModelSpec.model_fields["min_cluster_size"].default,
    }
    return CapabilitiesResponse(
        service=settings.service_name,
        version=settings.service_version,
        node=settings.node_name,
        llm_enabled=settings.topic_foundry_llm_enable,
        llm_transport=llm_transport,
        llm_bus_route=None,
        llm_intake_channel=settings.topic_foundry_llm_intake_channel if llm_transport == "bus" else None,
        llm_reply_prefix=settings.topic_foundry_llm_reply_prefix if llm_transport == "bus" else None,
        windowing_modes_supported=windowing_modes,
        enricher_modes_supported=enricher_modes,
        supported_metrics=sorted(SUPPORTED_METRICS),
        default_metric=ModelSpec.model_fields["metric"].default,
        cosine_impl_default=settings.topic_foundry_cosine_impl,
        defaults=defaults,
        introspection={"ok": bool(schemas), "schemas": schemas},
        default_embedding_url=settings.topic_foundry_embedding_url,
    )
