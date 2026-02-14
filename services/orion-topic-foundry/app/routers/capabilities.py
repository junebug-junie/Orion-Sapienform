from __future__ import annotations

from fastapi import APIRouter
from hdbscan import dist_metrics

from app.models import CapabilitiesResponse
from app.settings import settings
from app.topic_engine import _build_vectorizer

router = APIRouter()


@router.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities() -> CapabilitiesResponse:
    vectorizer = _build_vectorizer({})
    vectorizer_params = vectorizer.get_params()
    stop_words_mode = str(settings.topic_foundry_vectorizer_stop_words).strip().lower()
    stop_words_extra = [x.strip().lower() for x in str(settings.topic_foundry_stop_words_extra or "").split(",") if x.strip()]
    supported_metrics = sorted(str(metric) for metric in dist_metrics.METRIC_MAPPING.keys())
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
        supported_metrics=supported_metrics,
        default_metric=str(settings.topic_foundry_hdbscan_metric),
        defaults={
            "vectorizer": {
                "stop_words": stop_words_mode or "none",
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
            "ctfidf": {
                "reduce_frequent_words": True,
            },
        },
    )

