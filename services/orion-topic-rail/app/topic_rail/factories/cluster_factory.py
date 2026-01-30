from __future__ import annotations

from hdbscan import HDBSCAN

from app.settings import Settings


def build_hdbscan(settings: Settings) -> HDBSCAN:
    return HDBSCAN(
        min_cluster_size=max(2, int(settings.topic_rail_min_topic_size)),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
