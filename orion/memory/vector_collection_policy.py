"""Per-collection growth policy for Orion's vector DB (Chroma, ``orion-athena-vector-db``).

One place that says, for every Chroma collection, who writes it, whether it can be
rebuilt from somewhere else, and whether anything is ever allowed to delete from it.

Rule (Juniper, 2026-10-13): no silent memory loss. A collection may only get a
bounded retention if it is purely derived AND its source keeps the data. As of
2026-10-13 **no collection qualifies**, so ``retention`` is ``None`` everywhere and
nothing in the repo deletes from Chroma. Why each one does not qualify is in its
``reason``.

Consumers:
- ``services/orion-biometrics/app/vector_db_footprint.py`` labels the live per-collection
  counts it publishes on the heartbeat and names any live collection missing from this
  table (``unpoliced_collections``) so a new writer cannot appear unnoticed.
- ``tests/test_vector_collection_policy.py`` refuses a retention on anything that is not
  derived-with-a-rebuild-path, or on any recall-backing collection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

SourceKind = Literal["primary", "derived"]


@dataclass(frozen=True)
class Retention:
    """A bounded retention. Unused today -- kept so a future policy is typed, not ad hoc."""

    keep_newest: Optional[int] = None
    max_age_days: Optional[float] = None


@dataclass(frozen=True)
class CollectionPolicy:
    name: str
    source: SourceKind
    writer: str
    # How to re-project the collection if it is lost or pruned. None = no rebuild path.
    rebuild: Optional[str]
    # Read by recall / chat grounding. Never pruned automatically.
    recall_backing: bool
    retention: Optional[Retention]
    reason: str


POLICIES: dict[str, CollectionPolicy] = {
    p.name: p
    for p in (
        CollectionPolicy(
            name="orion_main_store",
            source="primary",
            writer="orion-vector-host (every orion:embedding:generate request, ~99% from llm-gateway)",
            rebuild=None,
            recall_backing=True,
            retention=None,
            reason=(
                "Fastest grower (~3k rows/day). The embedded text is not persisted anywhere "
                "else, so it cannot be re-projected; recall's default VECTOR_DB_COLLECTION. "
                "Monitor only."
            ),
        ),
        CollectionPolicy(
            name="orion_memory_crystallizations",
            source="derived",
            writer="orion-memory-consolidation (orion/memory/crystallization/projection_chroma.py)",
            rebuild="scripts/backfill_concept_relation_chroma.py (from Postgres)",
            recall_backing=True,
            retention=None,
            reason=(
                "Rebuildable from Postgres, but recall reads it for crystallized memory; "
                "pruning would make old memories unrecallable. Flat since 10-11."
            ),
        ),
        CollectionPolicy(
            name="orion_dreams",
            source="primary",
            writer="orion-dream (DREAM_SEARCH_COLLECTION)",
            rebuild=None,
            recall_backing=True,
            retention=None,
            reason="Dream search index; backs dream recall. A few rows/day.",
        ),
        CollectionPolicy(
            name="orion_curiosity",
            source="primary",
            writer="orion-hub curiosity investigations (HUB_CURIOSITY_SEARCH_COLLECTION)",
            rebuild=None,
            recall_backing=True,
            retention=None,
            reason="Orion's own investigation findings; backs curiosity recall. ~15 rows/day.",
        ),
        CollectionPolicy(
            name="orion_chat",
            source="primary",
            writer="orion-chat-memory (CHAT_MEMORY_COLLECTION)",
            rebuild=None,
            recall_backing=True,
            retention=None,
            reason="Chat memory recall index. A few rows/day.",
        ),
        CollectionPolicy(
            name="orion_day_letter_parts",
            source="derived",
            writer="orion-hub orion_day_introspect_listener (SEARCH_COLLECTION)",
            rebuild=None,
            recall_backing=True,
            retention=None,
            reason=(
                "Search index over Orion's Day letters, keyed by letter/part hash; Orion "
                "rereads its own letters through it. No backfill script exists, and a "
                "time cap would hide old letters from that reread. ~20 rows/day."
            ),
        ),
        CollectionPolicy(
            name="orion_reading_results",
            source="derived",
            writer="orion-hub reading search (HUB_READING_SEARCH_COLLECTION)",
            rebuild=None,
            recall_backing=True,
            retention=None,
            reason="Reading-result search index; no backfill script. A few rows/day.",
        ),
        CollectionPolicy(
            name="doc_semantic_drift",
            source="derived",
            writer="orion-cocreation-signals doc_semantic_drift producer",
            rebuild="re-run the producer over the repo docs",
            recall_backing=False,
            retention=None,
            reason=(
                "Derived from repo docs, but bounded by the doc count and the drift metric "
                "compares against older embeddings -- pruning would break the metric, "
                "not bound growth."
            ),
        ),
        CollectionPolicy(
            name="orion_atlas_concepts",
            source="derived",
            writer="orion/world_pulse_read/concept_search.py index_atlas_concepts (upsert per concept node)",
            rebuild="re-run index_atlas_concepts over the atlas graph",
            recall_backing=False,
            retention=None,
            reason=(
                "Upserted one row per atlas concept node, so size tracks the atlas, not "
                "time. A newest-N cap would only drop concepts from search."
            ),
        ),
    )
}


def policy_for(name: str) -> Optional[CollectionPolicy]:
    return POLICIES.get(name)


def unpoliced(collection_names: list[str]) -> list[str]:
    """Live collections nobody has written a policy for."""
    return sorted(n for n in collection_names if n not in POLICIES)


def retention_allowed(policy: CollectionPolicy) -> bool:
    """The no-silent-memory-loss rule as code: derived, rebuildable, not recall-backing."""
    return policy.source == "derived" and bool(policy.rebuild) and not policy.recall_backing
