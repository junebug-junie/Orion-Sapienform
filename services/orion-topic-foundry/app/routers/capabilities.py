from __future__ import annotations

from fastapi import APIRouter

from app.models import CapabilitiesResponse, ModelSpec, WindowingSpec
from app.settings import settings


router = APIRouter()


@router.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities() -> CapabilitiesResponse:
    segmentation_modes = list(WindowingSpec.model_fields["segmentation_mode"].annotation.__args__)
    enricher_modes = ["heuristic", "llm"]
    llm_transport = "bus" if settings.topic_foundry_llm_use_bus and settings.orion_bus_enabled else "http"
    defaults = {
        "embedding_source_url": settings.topic_foundry_embedding_url,
        "metric": ModelSpec.model_fields["metric"].default,
        "min_cluster_size": ModelSpec.model_fields["min_cluster_size"].default,
        "llm_bus_route": settings.topic_foundry_llm_bus_route,
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
        defaults=defaults,
    )
