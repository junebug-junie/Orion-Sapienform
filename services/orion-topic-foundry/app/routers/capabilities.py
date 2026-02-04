from __future__ import annotations

from fastapi import APIRouter

from app.models import CapabilitiesResponse, ModelSpec, WindowingSpec
from app.settings import settings


router = APIRouter()


@router.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities() -> CapabilitiesResponse:
    segmentation_modes = list(WindowingSpec.model_fields["segmentation_mode"].annotation.__args__)
    enricher_modes = ["heuristic", "llm"]
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
        segmentation_modes_supported=segmentation_modes,
        enricher_modes_supported=enricher_modes,
        defaults=defaults,
    )
