from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import requests
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from app.settings import Settings


class TopicEngineError(RuntimeError):
    pass


class VectorHostEmbedder:
    def __init__(self, *, url: str, batch_size: int, max_retries: int, retry_delay_secs: float) -> None:
        self.url = url
        self.batch_size = max(1, batch_size)
        self.max_retries = max(0, max_retries)
        self.retry_delay_secs = max(0.0, retry_delay_secs)

    def _post_batch(self, docs: Sequence[str]) -> List[List[float]]:
        attempt = 0
        while True:
            try:
                response = requests.post(self.url, json={"inputs": list(docs)}, timeout=60)
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    vectors = payload.get("embeddings") or payload.get("vectors") or payload.get("data")
                else:
                    vectors = payload
                if not isinstance(vectors, list):
                    raise TopicEngineError("Embedding backend response missing embeddings list")
                return vectors
            except Exception as exc:  # noqa: BLE001
                if attempt >= self.max_retries:
                    raise TopicEngineError(f"Vector host embedding request failed: {exc}") from exc
                attempt += 1
                time.sleep(self.retry_delay_secs)

    def embed(self, docs: Sequence[str]) -> np.ndarray:
        all_vectors: List[List[float]] = []
        for start in range(0, len(docs), self.batch_size):
            chunk = docs[start : start + self.batch_size]
            all_vectors.extend(self._post_batch(chunk))
        return np.asarray(all_vectors, dtype=np.float32)


class SentenceTransformerEmbedder:
    def __init__(self, *, model_name: str, batch_size: int) -> None:
        self.model = SentenceTransformer(model_name)
        self.batch_size = max(1, batch_size)

    def embed(self, docs: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(list(docs), batch_size=self.batch_size, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


def build_embedder(model_meta: Dict[str, Any], settings: Settings) -> Callable[[Sequence[str]], np.ndarray]:
    backend = (model_meta.get("embedding_backend") or settings.topic_foundry_embedding_backend or "vector_host").lower()
    if backend == "vector_host":
        url = model_meta.get("vector_host_url") or settings.topic_foundry_vector_host_url
        provider = VectorHostEmbedder(
            url=url,
            batch_size=int(model_meta.get("embedding_batch_size") or settings.topic_foundry_embedding_batch_size),
            max_retries=settings.topic_foundry_embedding_max_retries,
            retry_delay_secs=settings.topic_foundry_embedding_retry_delay_secs,
        )
        return provider.embed
    if backend == "st":
        model_name = model_meta.get("embedding_model") or settings.topic_foundry_embedding_model
        provider = SentenceTransformerEmbedder(
            model_name=model_name,
            batch_size=int(model_meta.get("embedding_batch_size") or settings.topic_foundry_embedding_batch_size),
        )
        return provider.embed
    raise TopicEngineError(f"Unsupported embedding_backend: {backend}")


def build_reducer(model_meta: Dict[str, Any], settings: Settings):
    reducer = (model_meta.get("reducer") or settings.topic_foundry_reducer or "umap").lower()
    if reducer != "umap":
        return None
    config = model_meta.get("reducer_config") or {}
    return UMAP(
        n_neighbors=int(config.get("n_neighbors", settings.topic_foundry_umap_n_neighbors)),
        n_components=int(config.get("n_components", settings.topic_foundry_umap_n_components)),
        min_dist=float(config.get("min_dist", settings.topic_foundry_umap_min_dist)),
        metric=str(config.get("metric", settings.topic_foundry_umap_metric)),
        random_state=int(model_meta.get("random_state", settings.topic_foundry_random_state)),
    )


def build_clusterer(model_meta: Dict[str, Any], settings: Settings):
    clusterer = (model_meta.get("clusterer") or settings.topic_foundry_clusterer or "hdbscan").lower()
    if clusterer != "hdbscan":
        return None
    config = model_meta.get("clusterer_config") or {}
    return HDBSCAN(
        min_cluster_size=int(config.get("min_cluster_size", settings.topic_foundry_hdbscan_min_cluster_size)),
        min_samples=int(config.get("min_samples", settings.topic_foundry_hdbscan_min_samples)),
        metric=str(config.get("metric", settings.topic_foundry_hdbscan_metric)),
        cluster_selection_method=str(
            config.get("cluster_selection_method", settings.topic_foundry_hdbscan_cluster_selection_method)
        ),
        prediction_data=True,
    )


def build_representation(model_meta: Dict[str, Any], settings: Settings):
    representation = (model_meta.get("representation") or settings.topic_foundry_representation or "ctfidf").lower()
    rep_config = model_meta.get("representation_config") or {}
    if representation == "ctfidf":
        return None
    if representation == "keybert":
        return KeyBERTInspired()
    if representation == "mmr":
        return MaximalMarginalRelevance(diversity=float(rep_config.get("diversity", 0.3)))
    if representation == "pos":
        return PartOfSpeech("en_core_web_sm")
    if representation == "llm":
        if not settings.topic_foundry_llm_enable:
            raise TopicEngineError("LLM representation selected but TOPIC_FOUNDRY_LLM_ENABLE=false")
        raise TopicEngineError("LLM representation backend is not configured for BERTopic representation in this environment")
    raise TopicEngineError(f"Unsupported representation: {representation}")


def _coerce_seed_topics(seed_topic_list: Any) -> List[List[str]]:
    if not isinstance(seed_topic_list, list):
        raise TopicEngineError("seed_topic_list must be a list")
    if not seed_topic_list:
        return []
    if all(isinstance(item, str) for item in seed_topic_list):
        return [list(seed_topic_list)]
    normalized: List[List[str]] = []
    for topic in seed_topic_list:
        if isinstance(topic, str):
            normalized.append([topic])
            continue
        if isinstance(topic, list) and all(isinstance(term, str) for term in topic):
            normalized.append(topic)
            continue
        raise TopicEngineError("seed_topic_list must be list[str] or list[list[str]]")
    return normalized


def build_topic_model(model_meta: Dict[str, Any], settings: Settings, topic_mode: str, topic_mode_params: Dict[str, Any]) -> BERTopic:
    vectorizer = CountVectorizer(
        ngram_range=(settings.topic_foundry_vectorizer_ngram_min, settings.topic_foundry_vectorizer_ngram_max),
        min_df=settings.topic_foundry_vectorizer_min_df,
        max_df=settings.topic_foundry_vectorizer_max_df,
        max_features=settings.topic_foundry_vectorizer_max_features,
        stop_words="english",
    )
    kwargs: Dict[str, Any] = {
        "umap_model": build_reducer(model_meta, settings),
        "hdbscan_model": build_clusterer(model_meta, settings),
        "vectorizer_model": vectorizer,
        "representation_model": build_representation(model_meta, settings),
        "top_n_words": int(model_meta.get("top_n_words") or settings.topic_foundry_top_n_words),
        "calculate_probabilities": True,
        "verbose": False,
    }
    if topic_mode == "guided":
        kwargs["seed_topic_list"] = _coerce_seed_topics(topic_mode_params.get("seed_topic_list"))
    if topic_mode == "zeroshot":
        zlist = topic_mode_params.get("zeroshot_topic_list")
        if not isinstance(zlist, list) or not all(isinstance(item, str) for item in zlist):
            raise TopicEngineError("zeroshot_topic_list must be list[str]")
        kwargs["zeroshot_topic_list"] = zlist
        if topic_mode_params.get("zeroshot_min_similarity") is not None:
            kwargs["zeroshot_min_similarity"] = float(topic_mode_params["zeroshot_min_similarity"])
    return BERTopic(**kwargs)
