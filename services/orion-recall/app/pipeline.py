# app/pipeline.py
from __future__ import annotations

from typing import Any, Dict, List

from .types import Fragment, RecallQuery, RecallResult
from .collectors import collect_fragments
from .scoring import score_fragments
from .postprocessing import postprocess_fragments


def run_recall_pipeline(q: RecallQuery) -> RecallResult:
    raw_frags: List[Fragment] = collect_fragments(q)
    scored_frags: List[Fragment] = score_fragments(raw_frags, q)
    final_frags: List[Fragment] = postprocess_fragments(scored_frags, q)

    debug: Dict[str, Any] = {
        "total_raw": len(raw_frags),
        "total_scored": len(scored_frags),
        "total_final": len(final_frags),
        "mode": q.mode,
        "time_window_days": q.time_window_days,
        "max_items": q.max_items,
    }

    return RecallResult(fragments=final_frags, debug=debug)
