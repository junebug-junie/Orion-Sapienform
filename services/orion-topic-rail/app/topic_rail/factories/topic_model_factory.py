from __future__ import annotations

from bertopic import BERTopic

from app.settings import Settings
from app.topic_rail.factories.cluster_factory import build_hdbscan
from app.topic_rail.factories.representation_factory import build_representation
from app.topic_rail.factories.umap_factory import build_umap
from app.topic_rail.factories.vectorizer_factory import build_vectorizer


def build_topic_model(settings: Settings) -> BERTopic:
    vectorizer = build_vectorizer(settings)
    umap_model = build_umap(settings)
    hdbscan_model = build_hdbscan(settings)
    representation_model = build_representation(settings)

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        representation_model=representation_model,
        calculate_probabilities=settings.topic_rail_calc_probs,
        verbose=False,
    )
    return topic_model
