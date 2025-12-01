# app/scoring.py
from __future__ import annotations

import math
import time
from typing import List, Dict, Any

from .types import Fragment, RecallQuery, RecallResult


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _compute_recency_factor(ts: float | None, mode: str) -> float:
    """
    Recency weight in [0, 1].

    - short_term: very sharp decay (days-scale)
    - hybrid: medium decay (weeks-scale)
    - deep: slow decay (months-scale)
    """
    if not ts:
        return 0.5  # unknown timestamps get neutral-ish score

    now = time.time()
    age_days = max(0.0, (now - float(ts)) / 86400.0)

    if mode == "short_term":
        tau = 3.0   # ~3 day horizon
    elif mode == "deep":
        tau = 60.0  # ~2 month horizon
    else:
        tau = 14.0  # default / hybrid

    # exp(-age / tau) ∈ (0, 1]
    return math.exp(-age_days / tau)


def _compute_semantic_score(f: Fragment) -> float:
    """
    Look for semantic similarity coming from vector backends.

    We expect vector_adapter to stash a similarity / score field into meta.
    If none is present, we fall back to 0.0 and let salience/recency carry.
    """
    meta = f.meta or {}
    # common patterns from vector DBs: similarity, score, distance, etc.
    sim = meta.get("similarity")
    if sim is None:
        sim = meta.get("score")

    val = _safe_float(sim, 0.0)
    # clamp to [0, 1]
    return max(0.0, min(1.0, val))


def _compute_final_score(
    semantic: float,
    salience: float,
    recency: float,
) -> float:
    """
    Compound score: semantic dominates; salience + recency temper it.

    We can tweak these weights later, but this already moves us well
    beyond "grab everything from the last N days".
    """
    return (
        0.6 * semantic +
        0.25 * salience +
        0.15 * recency
    )


def score_fragments(query: RecallQuery, fragments: List[Fragment]) -> RecallResult:
    """
    Central scoring & pruning logic.

    - Build a working set (already done in collectors).
    - Compute semantic + salience + recency per fragment.
    - Filter out low-signal noise.
    - Return top-K, plus debug metadata.
    """
    scored: List[Dict[str, Any]] = []

    for f in fragments:
        semantic = _compute_semantic_score(f)
        salience = _safe_float(getattr(f, "salience", 0.0), 0.0)
        recency = _compute_recency_factor(getattr(f, "ts", None), query.mode)

        final_score = _compute_final_score(semantic, salience, recency)

        scored.append(
            {
                "fragment": f,
                "semantic": semantic,
                "salience": salience,
                "recency": recency,
                "score": final_score,
            }
        )

    # Sort best → worst
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Hard floor: anything below this is basically noise
    MIN_SCORE = 0.20

    final_fragments: List[Fragment] = []
    for row in scored:
        if row["score"] < MIN_SCORE:
            continue
        final_fragments.append(row["fragment"])
        if len(final_fragments) >= query.max_items:
            break

    debug = {
        "total_raw": len(fragments),
        "total_scored": len(scored),
        "total_final": len(final_fragments),
        "mode": query.mode,
        "time_window_days": query.time_window_days,
        "max_items": query.max_items,
        "note": "semantic + salience + recency scoring; MIN_SCORE=0.20",
    }

    return RecallResult(
        fragments=final_fragments,
        debug=debug,
    )
