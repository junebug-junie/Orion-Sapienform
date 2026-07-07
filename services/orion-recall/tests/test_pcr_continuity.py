from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_RECALL_ROOT = _REPO / "services" / "orion-recall"
if str(_RECALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECALL_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app.fusion import render_continuity_bundle


def test_render_continuity_bundle_user_prioritized():
    candidates = [
        {"id": "t1", "source": "sql_chat", "snippet": "User: hey\nAssistant: hi", "score": 0.5},
        {"id": "t2", "source": "sql_chat", "snippet": "User: move stress\nAssistant: I hear you", "score": 0.8},
    ]
    profile = {
        "render_budget_tokens": 96,
        "max_total_items": 6,
        "render_lane": "continuity",
        "profile": "chat.continuity.v1",
    }
    bundle, _ = render_continuity_bundle(
        candidates=candidates,
        profile=profile,
        query_text="move stress",
        latency_ms=1,
    )
    assert "move stress" in bundle.rendered
