"""Vector source policy — removed from orion-recall; diagnostics only."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

_REPO = Path(__file__).resolve().parents[3]
_RECALL_ROOT = _REPO / "services" / "orion-recall"
if str(_RECALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECALL_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app.source_policy import build_vector_policy


def test_build_vector_policy_all_paths_removed() -> None:
    profile = {"profile": "assist.light.v1", "vector_top_k": 0}
    policy = build_vector_policy(profile, MagicMock(RECALL_ENABLE_VECTOR=True))
    for path in ("main", "anchor", "graphtri", "v2_shadow_exact", "v2_shadow_semantic"):
        assert policy[path]["allowed"] is False
        assert policy[path]["reason"] == "removed_from_orion_recall"
