"""Real scoring helpers for the chat_stance organ signal.

Replaces the old ``_TASK_MODE_COHERENCE``/``_SALIENCE_VALENCE`` lookup tables
(enum label -> hardcoded constant, measuring nothing). Both functions here are
grounded in something that actually happened on the turn:

  - ``score_stance_confidence`` reads cortex-exec's own repair/enforcement
    telemetry (whether the LLM's raw stance brief needed a fallback, semantic
    rewrite, quality-guard modification, or field normalization to become
    usable). More correction needed -> lower confidence.
  - ``cosine_similarity_01`` compares two embedding vectors (this turn's stance
    text vs. the previous turn's, same session) and rescales cosine similarity
    from [-1, 1] to [0, 1].

Neither function invents a number when the real input is missing -- callers
are expected to omit the dimension rather than fabricate a default.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from orion.signals.normalization import clamp01

# Each penalty corresponds to a real correction cortex-exec's own
# `build_chat_stance_debug_payload` / `parse_chat_stance_brief_with_debug`
# already had to make to the raw LLM output. Weights are ordered by how much
# the correction changes the brief's content (fallback replaces it wholesale;
# normalization only compacts/dedupes fields).
_FALLBACK_PENALTY = 0.40
_SEMANTIC_FALLBACK_PENALTY = 0.25
_QUALITY_MODIFIED_PENALTY = 0.15
_NORMALIZED_PENALTY = 0.05
_PARSE_ERROR_PENALTY = 0.20


def score_stance_confidence(debug: Dict[str, object]) -> Tuple[float, List[str]]:
    """Derive a real confidence score from ``chat_stance_debug``'s enforcement telemetry."""
    enforcement = debug.get("enforcement") if isinstance(debug.get("enforcement"), dict) else {}
    raw = debug.get("raw") if isinstance(debug.get("raw"), dict) else {}
    raw_enforcement = raw.get("enforcement") if isinstance(raw.get("enforcement"), dict) else {}
    parse_error = raw_enforcement.get("parse_error")

    score = 1.0
    reasons: List[str] = []
    if enforcement.get("fallback_invoked"):
        score -= _FALLBACK_PENALTY
        reasons.append("confidence_penalty_fallback_invoked")
    if enforcement.get("semantic_fallback"):
        score -= _SEMANTIC_FALLBACK_PENALTY
        reasons.append("confidence_penalty_semantic_fallback")
    if enforcement.get("quality_modified"):
        score -= _QUALITY_MODIFIED_PENALTY
        reasons.append("confidence_penalty_quality_modified")
    if enforcement.get("normalized_applied"):
        score -= _NORMALIZED_PENALTY
        reasons.append("confidence_penalty_normalized_applied")
    if parse_error:
        score -= _PARSE_ERROR_PENALTY
        reasons.append("confidence_penalty_parse_error")
    return clamp01(score), reasons


def cosine_similarity_01(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[float]:
    """Cosine similarity between two vectors, rescaled from [-1, 1] to [0, 1]. None if not comparable."""
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    cosine = dot / (norm_a * norm_b)
    return clamp01((cosine + 1.0) / 2.0)
