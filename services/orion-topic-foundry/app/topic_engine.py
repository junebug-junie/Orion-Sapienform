from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

from app.services.embedding_client import VectorHostEmbeddingProvider
from app.settings import settings
import logging


logger = logging.getLogger("topic-foundry.topic_engine")


class VectorHostEmbeddingModel:
    def __init__(self, provider: VectorHostEmbeddingProvider, batch_size: int = 32) -> None:
        self.provider = provider
        self.batch_size = max(1, int(batch_size))

    def embed(self, documents: List[str], verbose: bool = False):
        vectors: List[List[float]] = []
        for start in range(0, len(documents), self.batch_size):
            chunk = documents[start : start + self.batch_size]
            vectors.extend(self.provider.embed_texts(chunk))
        return vectors


@dataclass
class TopicEngineParts:
    embedding_model: Any
    reducer: UMAP
    clusterer: HDBSCAN
    representation_model: Any
    ctfidf_model: Any
    vectorizer_model: Any
    bertopic_kwargs: Dict[str, Any]
    backend_names: Dict[str, str]


def _parse_topic_list(value: Any) -> Optional[List[List[str]]]:
    if not value:
        return None
    if isinstance(value, list):
        if not value:
            return None
        if all(isinstance(item, str) for item in value):
            return [[item.strip() for item in value if str(item).strip()]]
        parsed: List[List[str]] = []
        for item in value:
            if isinstance(item, list):
                parsed.append([str(x).strip() for x in item if str(x).strip()])
            elif isinstance(item, str) and item.strip():
                parsed.append([item.strip()])
        return [grp for grp in parsed if grp]
    if isinstance(value, str):
        chunks = [chunk.strip() for chunk in value.split("|") if chunk.strip()]
        if not chunks:
            return None
        return [[tok.strip() for tok in chunk.split(",") if tok.strip()] for chunk in chunks]
    return None


def _parse_zeroshot_list(value: Any) -> Optional[List[str]]:
    if not value:
        return None
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    return None




def _build_vectorizer(meta: Dict[str, Any]) -> CountVectorizer:
    stop_words_raw = str(meta.get("vectorizer_stop_words", settings.topic_foundry_vectorizer_stop_words)).strip().lower()
    if stop_words_raw in {"", "none", "null", "off", "false", "0"}:
        stop_words = None
        stop_words_label = "none"
    elif stop_words_raw == "english":
        extras_raw = str(meta.get("stop_words_extra", settings.topic_foundry_stop_words_extra) or "")
        extras = {x.strip().lower() for x in extras_raw.split(",") if x.strip()}
        stop_words = sorted(set(ENGLISH_STOP_WORDS).union(extras))
        stop_words_label = f"english+extra({len(extras)})"
    else:
        custom = [x.strip().lower() for x in stop_words_raw.split(",") if x.strip()]
        stop_words = custom or None
        stop_words_label = "custom" if custom else "none"

    ngram_min = int(meta.get("vectorizer_ngram_min", settings.topic_foundry_vectorizer_ngram_min))
    ngram_max = int(meta.get("vectorizer_ngram_max", settings.topic_foundry_vectorizer_ngram_max))
    if ngram_min < 1:
        ngram_min = 1
    if ngram_max < ngram_min:
        ngram_max = ngram_min

    vectorizer = CountVectorizer(
        stop_words=stop_words,
        max_df=float(meta.get("vectorizer_max_df", settings.topic_foundry_vectorizer_max_df)),
        min_df=int(meta.get("vectorizer_min_df", settings.topic_foundry_vectorizer_min_df)),
        ngram_range=(ngram_min, ngram_max),
        max_features=int(meta.get("vectorizer_max_features", settings.topic_foundry_vectorizer_max_features)),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    logger.info(
        "ctfidf vectorizer params=%s stop_words_mode=%s",
        vectorizer.get_params(),
        stop_words_label,
    )
    return vectorizer

def build_topic_engine(model_meta: Optional[Dict[str, Any]] = None) -> TopicEngineParts:
    meta = model_meta or {}
    embedding_backend = str(meta.get("embedding_backend") or settings.topic_foundry_embedding_backend).strip().lower()
    embedding_model_name = str(meta.get("embedding_model") or settings.topic_foundry_embedding_model).strip()
    reducer_name = str(meta.get("reducer") or settings.topic_foundry_reducer).strip().lower()
    clusterer_name = str(meta.get("clusterer") or settings.topic_foundry_clusterer).strip().lower()
    representation = str(meta.get("representation") or settings.topic_foundry_representation).strip().lower()

    if embedding_backend == "st":
        embedding_model = SentenceTransformer(embedding_model_name)
    else:
        provider = VectorHostEmbeddingProvider(
            settings.topic_foundry_vector_host_url,
            retries=settings.topic_foundry_embedding_max_retries,
            backoff_sec=settings.topic_foundry_embedding_retry_delay_secs,
        )
        embedding_model = VectorHostEmbeddingModel(provider, settings.topic_foundry_embedding_batch_size)
        embedding_backend = "vector_host"

    reducer = UMAP(
        n_neighbors=int(meta.get("umap_n_neighbors", settings.topic_foundry_umap_n_neighbors)),
        n_components=int(meta.get("umap_n_components", settings.topic_foundry_umap_n_components)),
        min_dist=float(meta.get("umap_min_dist", settings.topic_foundry_umap_min_dist)),
        metric=str(meta.get("umap_metric", settings.topic_foundry_umap_metric)),
        random_state=int(meta.get("random_state", settings.topic_foundry_random_state)),
    )

    clusterer = HDBSCAN(
        min_cluster_size=int(meta.get("hdbscan_min_cluster_size", settings.topic_foundry_hdbscan_min_cluster_size)),
        min_samples=int(meta.get("hdbscan_min_samples", settings.topic_foundry_hdbscan_min_samples)),
        metric=str(meta.get("hdbscan_metric", settings.topic_foundry_hdbscan_metric)),
        cluster_selection_method=str(
            meta.get(
                "hdbscan_cluster_selection_method",
                settings.topic_foundry_hdbscan_cluster_selection_method,
            )
        ),
        prediction_data=True,
    )

    if reducer_name != "umap":
        reducer_name = "umap"
    if clusterer_name != "hdbscan":
        clusterer_name = "hdbscan"

    ctfidf_model = None
    vectorizer_model = None
    if representation == "keybert":
        representation_model = KeyBERTInspired()
    elif representation == "mmr":
        representation_model = MaximalMarginalRelevance(diversity=0.3)
    elif representation == "pos":
        representation_model = PartOfSpeech("en_core_web_sm")
    elif representation == "llm":
        representation_model = None
    else:
        representation_model = None
        vectorizer_model = _build_vectorizer(meta)
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        logger.info("ctfidf_model reduce_frequent_words=%s", True)
        representation = "ctfidf"

    bertopic_kwargs: Dict[str, Any] = {
        "top_n_words": int(meta.get("top_n_words", settings.topic_foundry_top_n_words)),
        "min_topic_size": int(meta.get("hdbscan_min_cluster_size", settings.topic_foundry_hdbscan_min_cluster_size)),
    }
    seeds = _parse_topic_list(meta.get("seed_topic_list"))
    if seeds:
        bertopic_kwargs["seed_topic_list"] = seeds
    zeroshot = _parse_zeroshot_list(meta.get("zeroshot_topic_list"))
    if zeroshot:
        bertopic_kwargs["zeroshot_topic_list"] = zeroshot

    return TopicEngineParts(
        embedding_model=embedding_model,
        reducer=reducer,
        clusterer=clusterer,
        representation_model=representation_model,
        ctfidf_model=ctfidf_model,
        vectorizer_model=vectorizer_model,
        bertopic_kwargs=bertopic_kwargs,
        backend_names={
            "embedding_backend": embedding_backend,
            "embedding_model": embedding_model_name,
            "reducer": reducer_name,
            "clusterer": clusterer_name,
            "representation": representation,
        },
    )
