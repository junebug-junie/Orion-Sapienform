# app/pipeline.py
from __future__ import annotations

from .types import RecallQuery, RecallResult
from .collectors import collect_fragments
from .scoring import score_fragments


def run_recall_pipeline(query: RecallQuery) -> RecallResult:
    """
    End-to-end recall pipeline:

    1. Collect candidate fragments from SQL, vector, RDF.
    2. Score them with semantic + salience + recency.
    3. Return a pruned, ranked RecallResult.
    """
    fragments = collect_fragments(query)
    result = score_fragments(query, fragments)
    return result
