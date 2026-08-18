"""Concept-profile repository adapter backed by the live substrate concept region.

The `local` (spaCy-extracted JSON store) and `graph` (Fuseki/SPARQL) backends in
`profile_repository.py` are both dead in practice: `local` is only ever written
by `bus_worker.py`'s autonomous trigger, which has been
`CONCEPT_AUTONOMOUS_TRIGGER_ENABLED=false` since 2026-07-11 (spaCy noun-chunk
extraction has no stopword/POS filtering -- see
`docs/superpowers/specs/2026-07-11-drive-engine-concept-induction-deactivation-design.md`),
and `graph` points at Apache Fuseki, which no longer exists as of the
2026-07-23 decommission campaign (`orion-athena-fuseki` resolves nowhere).

This backend reads the same live FalkorDB substrate concept region that
`orion/substrate/relational/adapters/concept_induction_ctx.py` already reads
for `chat_stance`'s `concept_induced` tier, and that Hub's Concept Atlas
(`services/orion-hub/scripts/concept_atlas_routes.py`) already renders --
populated by golden seed concepts (`orion/substrate/seed.py`) and
`orion-topic-foundry`-derived concepts (`orion/substrate/adapters/topic_foundry.py`).
It does not run its own extraction; it projects the live substrate concept
nodes into the same `ConceptProfile` shape the other backends produce, one
synthetic profile per subject, so downstream consumers (the
`concept_induction_pass` chat workflow / Hub's Concept Induction Details
Modal) don't need to change.

Concepts are grouped by `anchor_scope`, which is already exactly the same
three-value vocabulary as `ConceptProfile.subject` ("orion", "juniper",
"relationship") -- see `concept_induction_ctx.py`'s `_SUBJECTS` -- so no
subject-mapping logic is invented here, it mirrors that adapter's precedent.

No clustering and no `state_estimate`: the substrate store doesn't compute
either today, so both are left empty/`None` rather than fabricated (CLAUDE.md
"no empty-shell cognition" -- honest absence, not a placeholder pretending to
be a real reading).
"""

from __future__ import annotations

import logging
from typing import Sequence

from orion.core.schemas.concept_induction import ConceptItem, ConceptProfile, make_concept_id
from orion.substrate import build_substrate_store_from_env
from orion.substrate.store import SubstrateGraphStore

from .profile_repository import ConceptProfileLookupV1, ConceptProfileRepositoryStatus

logger = logging.getLogger("orion.spark.concept.substrate_repository")

_SUBJECTS = ("orion", "relationship", "juniper")


class SubstrateConceptProfileRepository:
    """Repository seam adapter backed by the live FalkorDB substrate concept region."""

    def __init__(
        self,
        *,
        store: SubstrateGraphStore | None = None,
        limit_nodes: int = 256,
    ) -> None:
        self._store = store
        self._limit_nodes = limit_nodes

    def _get_store(self) -> SubstrateGraphStore | None:
        if self._store is not None:
            return self._store
        try:
            self._store = build_substrate_store_from_env()
        except Exception as exc:
            logger.debug("substrate_concept_repository_store_init_failed error=%s", exc)
            return None
        return self._store

    def status(self) -> ConceptProfileRepositoryStatus:
        store = self._get_store()
        return ConceptProfileRepositoryStatus(
            backend="substrate",
            source_path="falkor:concept_region",
            placeholder_default_in_use=False,
            source_available=store is not None,
        )

    def _query_by_subject(self, subjects: Sequence[str]) -> dict[str, list] | None:
        """Return concept nodes grouped by anchor_scope, or None on failure."""
        store = self._get_store()
        if store is None:
            return None
        try:
            result = store.query_concept_region(limit_nodes=self._limit_nodes, limit_edges=0)
        except Exception as exc:
            logger.debug("substrate_concept_repository_query_failed error=%s", exc)
            return None
        if result is None or result.degraded:
            return None

        wanted = set(subjects)
        by_subject: dict[str, list] = {subject: [] for subject in subjects}
        for node in list(getattr(result.slice, "nodes", None) or []):
            if getattr(node, "node_kind", None) != "concept":
                continue
            anchor = getattr(node, "anchor_scope", None)
            if anchor not in wanted:
                continue
            by_subject[anchor].append(node)
        return by_subject

    def _build_profile(self, subject: str, nodes: list) -> ConceptProfile:
        observed_ats = [node.temporal.observed_at for node in nodes if node.temporal is not None]
        window_start = min(observed_ats) if observed_ats else _utc_now()
        window_end = max(observed_ats) if observed_ats else window_start

        concepts: list[ConceptItem] = []
        for node in sorted(nodes, key=lambda n: n.signals.salience, reverse=True):
            label = node.label
            metadata = dict(node.metadata or {})
            concept_type = str(metadata.get("concept_type") or "topic")
            concepts.append(
                ConceptItem(
                    concept_id=make_concept_id(label),
                    label=label,
                    aliases=[],
                    type=concept_type,
                    salience=node.signals.salience,
                    confidence=node.signals.confidence,
                    metadata={"node_id": node.node_id, "anchor_scope": node.anchor_scope, "source": "substrate"},
                )
            )

        return ConceptProfile(
            profile_id=f"substrate-{subject}",
            subject=subject,
            revision=1,
            window_start=window_start,
            window_end=window_end,
            concepts=concepts,
            clusters=[],
            state_estimate=None,
            metadata={"source": "substrate_concept_region", "node_count": len(nodes)},
        )

    def get_latest(self, subject: str, *, observer: dict[str, str] | None = None) -> ConceptProfileLookupV1:
        return self.list_latest([subject], observer=observer)[0]

    def list_latest(
        self, subjects: Sequence[str], *, observer: dict[str, str] | None = None
    ) -> list[ConceptProfileLookupV1]:
        by_subject = self._query_by_subject(subjects)
        if by_subject is None:
            results = [
                ConceptProfileLookupV1(
                    subject=subject,
                    profile=None,
                    availability="unavailable",
                    unavailable_reason="substrate_store_unavailable",
                )
                for subject in subjects
            ]
        else:
            results = []
            for subject in subjects:
                nodes = by_subject.get(subject) or []
                if not nodes:
                    results.append(ConceptProfileLookupV1(subject=subject, profile=None, availability="empty"))
                    continue
                results.append(
                    ConceptProfileLookupV1(
                        subject=subject,
                        profile=self._build_profile(subject, nodes),
                        availability="available",
                    )
                )
        logger.info(
            "concept_profile_repository_status %s",
            str(
                {
                    "backend": "substrate",
                    "consumer": (observer or {}).get("consumer"),
                    "correlation_id": (observer or {}).get("correlation_id"),
                    "session_id": (observer or {}).get("session_id"),
                    "subjects_requested": list(subjects),
                    "profiles_returned": sum(1 for item in results if item.availability == "available"),
                    "unavailable_reason": next(
                        (item.unavailable_reason for item in results if item.availability == "unavailable"), None
                    ),
                }
            ),
        )
        return results


def _utc_now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)
