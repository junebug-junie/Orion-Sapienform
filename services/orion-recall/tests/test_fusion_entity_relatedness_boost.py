"""Phase 2 of entity-graph-reasoning (docs/superpowers/specs/
2026-07-19-recall-entity-graph-reasoning-arc.md): fuse_candidates additively
boosts a candidate's composite score when its source turn is in the caller-
supplied entity_boost_map. Scoped to falkor_chat candidates only (the only
source whose id/uri reliably IS the real Falkor turn_id)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_RECALL_ROOT = _REPO / "services" / "orion-recall"
if str(_RECALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECALL_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app.fusion import _entity_relatedness_boost, fuse_candidates


def _base_profile(**overrides) -> dict:
    base = {
        "profile": "chat.general.v1",
        "max_per_source": 10,
        "max_total_items": 20,
        "render_budget_tokens": 256,
        "time_decay_half_life_hours": 72,
        "relevance": {"backend_weights": {"falkor_chat": 0.5, "sql_chat": 0.5}},
    }
    base.update(overrides)
    return base


def test_entity_relatedness_boost_zero_without_map() -> None:
    cand = {"source": "falkor_chat", "uri": "turn-1"}
    assert _entity_relatedness_boost(cand, _base_profile(), None) == 0.0
    assert _entity_relatedness_boost(cand, _base_profile(), {}) == 0.0


def test_entity_relatedness_boost_scoped_to_falkor_chat_only() -> None:
    """sql_chat's id is a correlation_id/synthetic fallback, not confirmed
    to match Falkor's real turn_id -- the boost must not apply to it, even
    if a turn_id happens to collide in the map."""
    boost_map = {"turn-1": 0.8}
    sql_cand = {"source": "sql_chat", "uri": "turn-1", "id": "turn-1"}
    assert _entity_relatedness_boost(sql_cand, _base_profile(), boost_map) == 0.0


def test_entity_relatedness_boost_scales_by_profile_weight() -> None:
    cand = {"source": "falkor_chat", "uri": "turn-1"}
    boost_map = {"turn-1": 0.8}
    profile = _base_profile(entity_relatedness_boost_weight=0.5)
    assert _entity_relatedness_boost(cand, profile, boost_map) == 0.5 * 0.8


def test_entity_relatedness_boost_clamps_raw_score_to_one() -> None:
    cand = {"source": "falkor_chat", "uri": "turn-1"}
    boost_map = {"turn-1": 5.0}  # malformed/out-of-range input must not blow the weight up
    profile = _base_profile(entity_relatedness_boost_weight=0.5)
    assert _entity_relatedness_boost(cand, profile, boost_map) == 0.5


def test_entity_relatedness_boost_falls_back_to_id_when_no_uri() -> None:
    cand = {"source": "falkor_chat", "id": "turn-1"}
    boost_map = {"turn-1": 1.0}
    assert _entity_relatedness_boost(cand, _base_profile(), boost_map) > 0.0


def test_fuse_candidates_reranks_boosted_falkor_chat_candidate_above_higher_base_score() -> None:
    """End-to-end: a falkor_chat candidate with a lower base score but a real
    entity-relatedness boost should be able to outrank a higher-base-score
    candidate with no boost -- the whole point of Phase 2."""
    cands = [
        {"id": "a", "source": "falkor_chat", "uri": "turn-boosted", "text": "unrelated text", "score": 0.5},
        {"id": "b", "source": "falkor_chat", "uri": "turn-plain", "text": "unrelated text", "score": 0.55},
    ]
    profile = _base_profile(entity_relatedness_boost_weight=0.3)
    boost_map = {"turn-boosted": 1.0}

    bundle, ranking_debug = fuse_candidates(
        candidates=cands,
        profile=profile,
        query_text="unrelated text",
        diagnostic=True,
        entity_boost_map=boost_map,
    )

    assert bundle.items[0].id == "a"
    boosted_debug = next(r for r in ranking_debug if r["id"] == "a")
    plain_debug = next(r for r in ranking_debug if r["id"] == "b")
    assert boosted_debug["composite_score"] > plain_debug["composite_score"]


def test_fuse_candidates_entity_boost_map_none_is_a_safe_noop() -> None:
    """Callers that don't pass entity_boost_map at all (every existing call
    site before this Phase 2 patch) must see identical behavior."""
    cands = [{"id": "a", "source": "falkor_chat", "uri": "turn-1", "text": "hello", "score": 0.5}]
    profile = _base_profile()
    bundle, _ = fuse_candidates(candidates=cands, profile=profile, query_text="hello")
    assert len(bundle.items) == 1
