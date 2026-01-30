from __future__ import annotations

from typing import Optional

from umap import UMAP

from app.settings import Settings


def build_umap(settings: Settings) -> Optional[UMAP]:
    return UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
