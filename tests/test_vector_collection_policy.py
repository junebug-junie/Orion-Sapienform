"""No-silent-memory-loss gate for orion/memory/vector_collection_policy.py.

LIVE_2026_10_13 is a snapshot, so the coverage test only proves the table matched that day.
The live guard against a new unreviewed writer is the biometrics heartbeat's
`details.vector_db.unpoliced_collections` (+ WARNING log).
"""
from __future__ import annotations

from orion.memory.vector_collection_policy import POLICIES, retention_allowed, unpoliced

# Live collections in orion-athena-vector-db on 2026-10-13 (GET /api/v1/collections).
LIVE_2026_10_13 = [
    "orion_atlas_concepts", "orion_dreams", "orion_chat", "orion_curiosity",
    "orion_reading_results", "doc_semantic_drift", "orion_main_store",
    "orion_day_letter_parts", "orion_memory_crystallizations",
]

# Juniper 2026-10-13: these back recall and must never be auto-pruned.
PROTECTED = {"orion_memory_crystallizations", "orion_main_store", "orion_dreams", "orion_curiosity"}


def test_every_live_collection_has_a_policy():
    assert unpoliced(LIVE_2026_10_13) == []


def test_retention_only_where_rule_allows():
    for p in POLICIES.values():
        if p.retention is not None:
            assert retention_allowed(p), f"{p.name}: retention on a non-qualifying collection"


def test_protected_collections_never_qualify():
    for name in PROTECTED:
        p = POLICIES[name]
        assert p.recall_backing and p.retention is None and not retention_allowed(p)


def test_every_policy_states_a_reason_and_writer():
    for p in POLICIES.values():
        assert p.reason.strip() and p.writer.strip()


def test_unpoliced_names_new_collections():
    assert unpoliced(["orion_main_store", "zzz_new"]) == ["zzz_new"]
