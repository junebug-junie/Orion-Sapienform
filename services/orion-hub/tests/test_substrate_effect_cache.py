from __future__ import annotations

from scripts.substrate_effect_cache import (
    SubstrateEffectCache,
    SubstrateEffectSnapshot,
)


def _make_snapshot(turn_id: str) -> SubstrateEffectSnapshot:
    return SubstrateEffectSnapshot(
        turn_id=turn_id,
        message_id=None,
        user_text="hi",
        appraisal=None,
        signal=None,
        evidence=[],
        contract_before={"mode": "default"},
        contract_after={"mode": "default"},
        causal_molecule_ids=[],
    )


def test_store_then_get_returns_same_snapshot():
    cache = SubstrateEffectCache(max_entries=4)
    snap = _make_snapshot("t1")
    cache.store(snap)
    assert cache.get("t1") is snap


def test_unknown_turn_returns_none():
    cache = SubstrateEffectCache(max_entries=4)
    assert cache.get("missing") is None


def test_lru_eviction_drops_oldest():
    cache = SubstrateEffectCache(max_entries=2)
    cache.store(_make_snapshot("a"))
    cache.store(_make_snapshot("b"))
    cache.store(_make_snapshot("c"))
    assert cache.get("a") is None
    assert cache.get("b") is not None
    assert cache.get("c") is not None


def test_recent_returns_newest_first():
    cache = SubstrateEffectCache(max_entries=4)
    cache.store(_make_snapshot("a"))
    cache.store(_make_snapshot("b"))
    cache.store(_make_snapshot("c"))
    recent_ids = [s.turn_id for s in cache.recent(limit=10)]
    assert recent_ids == ["c", "b", "a"]
