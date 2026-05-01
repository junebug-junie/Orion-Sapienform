from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_RECALL_ROOT = _REPO / "services" / "orion-recall"
if str(_RECALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECALL_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app.fusion import fuse_candidates


def _base_profile() -> dict:
    return {
        "profile": "journal.daily.grounded.v1",
        "max_per_source": 10,
        "max_total_items": 20,
        "render_budget_tokens": 256,
        "time_decay_half_life_hours": 72,
        "relevance": {
            "backend_weights": {
                "vector": 1.0,
                "sql_chat": 0.8,
                "sql_timeline": 0.8,
            }
        },
    }


def test_fusion_filters_exclude_source() -> None:
    cands = [
        {"id": "a", "source": "pageindex_lexical", "text": "prior journal block", "score": 0.99},
        {"id": "b", "source": "vector", "text": "useful non-junk context", "score": 0.5},
    ]
    p = _base_profile()
    p["filters"] = {"exclude_sources": ["pageindex_lexical"]}
    bundle, _ = fuse_candidates(candidates=cands, profile=p, query_text="context", diagnostic=True)
    assert len(bundle.items) == 1
    assert bundle.items[0].id == "b"


def test_fusion_filters_exclude_templated_daily_prefix() -> None:
    cands = [
        {
            "id": "a",
            "source": "vector",
            "text": "## Orion — Daily Journal\n\n**Title:** X",
            "score": 0.99,
        },
        {"id": "b", "source": "vector", "text": "Notes from collapse triage work", "score": 0.4},
    ]
    p = _base_profile()
    p["filters"] = {"exclude_snippet_prefixes": ["## Orion — Daily Journal"]}
    bundle, _ = fuse_candidates(candidates=cands, profile=p, query_text="work", diagnostic=True)
    assert len(bundle.items) == 1
    assert bundle.items[0].id == "b"


def test_fusion_filters_exclude_snippet_substrings() -> None:
    cands = [
        {
            "id": "a",
            "source": "vector",
            "text": "The day unfolded with the usual rhythm of troubleshooting and incremental progress.",
            "score": 0.99,
        },
        {"id": "b", "source": "vector", "text": "Different discussion about deployment topology.", "score": 0.5},
    ]
    p = _base_profile()
    p["filters"] = {"exclude_snippet_substrings": ["the day unfolded with the usual rhythm"]}
    bundle, _ = fuse_candidates(candidates=cands, profile=p, query_text="hello", diagnostic=True)
    assert len(bundle.items) == 1
    assert bundle.items[0].id == "b"
