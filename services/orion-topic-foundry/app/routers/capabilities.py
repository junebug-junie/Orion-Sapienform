from __future__ import annotations

from fastapi import APIRouter

from app.models import CapabilitiesResponse
from app.settings import settings

router = APIRouter()


@router.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities() -> CapabilitiesResponse:
    return CapabilitiesResponse(
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
