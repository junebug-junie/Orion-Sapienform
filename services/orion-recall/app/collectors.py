# app/collectors.py
from __future__ import annotations

from typing import List

from .settings import settings
from .types import Fragment, RecallQuery
from .storage.sql_adapter import fetch_sql_fragments
from .storage.vector_adapter import fetch_vector_fragments
from .storage.rdf_adapater import fetch_rdf_fragments


def collect_fragments(q: RecallQuery) -> List[Fragment]:
    """
    Fan-out collector.

    This is the only place that knows which storage backends exist.
    """
    fragments: List[Fragment] = []

    # SQL (chat + collapse/enrichment)
    if settings.RECALL_ENABLE_SQL_CHAT or settings.RECALL_ENABLE_SQL_MIRRORS:
        fragments.extend(
            fetch_sql_fragments(
                time_window_days=q.time_window_days,
                include_chat=settings.RECALL_ENABLE_SQL_CHAT,
                include_mirrors=settings.RECALL_ENABLE_SQL_MIRRORS,
            )
        )

    # Vector neighbors (Chroma / orion-vector-db)
    if settings.RECALL_ENABLE_VECTOR and settings.RECALL_VECTOR_BASE_URL:
        fragments.extend(
            fetch_vector_fragments(
                query_text=q.text,
                time_window_days=q.time_window_days,
                max_items=settings.RECALL_VECTOR_MAX_ITEMS,
            )
        )

    # RDF (GraphDB)
    if settings.RECALL_ENABLE_RDF and settings.RECALL_RDF_ENDPOINT_URL:
        fragments.extend(
            fetch_rdf_fragments(
                query_text=q.text,
                max_items=8,
            )
        )

    return fragments
