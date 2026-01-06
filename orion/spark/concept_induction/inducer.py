from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.schemas.concept_induction import (
    ConceptCluster,
    ConceptEvidenceRef,
    ConceptItem,
    ConceptProfile,
    ConceptProfileDelta,
    StateEstimate,
    make_concept_id,
)

from .clusterer import ConceptClusterer
from .embedder import EmbeddingClient
from .extractor import SpacyConceptExtractor
from .settings import ConceptSettings
from .summarizer import Summarizer

logger = logging.getLogger("orion.spark.concept.inducer")


@dataclass
class WindowEvent:
    text: str
    timestamp: datetime
    envelope: BaseEnvelope


@dataclass
class InductionResult:
    profile: ConceptProfile
    delta: Optional[ConceptProfileDelta]


class ConceptInducer:
    """Extraction → embedding → clustering → profile assembly."""

    def __init__(
        self,
        settings: ConceptSettings,
        *,
        extractor: Optional[SpacyConceptExtractor] = None,
        embedder: Optional[EmbeddingClient] = None,
        clusterer: Optional[ConceptClusterer] = None,
        summarizer: Optional[Summarizer] = None,
        store_loader=None,
        store_saver=None,
        service_ref: Optional[ServiceRef] = None,
    ) -> None:
        self.settings = settings
        self.extractor = extractor or SpacyConceptExtractor(settings.spacy_model)
        self.embedder = embedder or EmbeddingClient(
            settings.embedding_base_url, settings.embedding_timeout_sec
        )
        self.clusterer = clusterer or ConceptClusterer(settings.cluster_cosine_threshold)
        self.summarizer = summarizer
        self.store_loader = store_loader
        self.store_saver = store_saver
        self.service_ref = service_ref or ServiceRef(
            name=settings.service_name,
            version=settings.service_version,
            node=settings.node_name,
        )

    def _evidence_from_env(self, env: BaseEnvelope) -> ConceptEvidenceRef:
        ts = env.created_at if env.created_at.tzinfo else env.created_at.replace(tzinfo=timezone.utc)
        corr = env.correlation_id if hasattr(env, "correlation_id") else None
        return ConceptEvidenceRef(
            message_id=env.id,
            correlation_id=corr,
            timestamp=ts,
            channel="unknown",
        )

    def _build_state_estimate(
        self, window: List[WindowEvent], window_start: datetime, window_end: datetime
    ) -> Optional[StateEstimate]:
        if not window:
            return None
        # Simple heuristics: novelty ~ unique texts, focus ~ avg length
        unique_text = len({w.text for w in window})
        novelty = min(1.0, unique_text / max(1, len(window)))
        avg_len = sum(len(w.text) for w in window) / max(1, len(window))
        focus = max(0.0, min(1.0, avg_len / 500.0))
        return StateEstimate(
            dimensions={"novelty": novelty, "focus": focus},
            trend={"novelty": 0.0, "focus": 0.0},
            confidence=0.6,
            window_start=window_start,
            window_end=window_end,
        )

    def _hash_profile(self, profile: ConceptProfile) -> str:
        h = hashlib.sha256()
        h.update(profile.model_dump_json().encode("utf-8"))
        return h.hexdigest()

    def _compute_delta(
        self, new: ConceptProfile, previous: Optional[ConceptProfile]
    ) -> Optional[ConceptProfileDelta]:
        if not previous:
            return None
        prev_ids = {c.concept_id for c in previous.concepts}
        new_ids = {c.concept_id for c in new.concepts}
        added = sorted(new_ids - prev_ids)
        removed = sorted(prev_ids - new_ids)
        updated = sorted(
            cid for cid in (new_ids & prev_ids) if cid != "ignored"
        )
        return ConceptProfileDelta(
            profile_id=new.profile_id,
            from_rev=previous.revision,
            to_rev=new.revision,
            added=added,
            removed=removed,
            updated=updated,
            rationale="auto-diff",
            evidence=[e for c in new.concepts for e in c.evidence][:20],
        )

    async def run(
        self,
        *,
        subject: str,
        window: List[WindowEvent],
    ) -> InductionResult:
        window_start = min((w.timestamp for w in window), default=datetime.now(timezone.utc))
        window_end = max((w.timestamp for w in window), default=datetime.now(timezone.utc))
        texts = [w.text for w in window]
        evidence = [self._evidence_from_env(w.envelope) for w in window]

        extraction = self.extractor.extract(texts, top_k=self.settings.max_candidates)
        candidates = extraction.candidates
        embeddings_resp = self.embedder.embed(candidates)
        clusters = self.clusterer.cluster(candidates, embeddings_resp.embeddings)

        concept_items: List[ConceptItem] = []
        for cand in candidates:
            concept_id = make_concept_id(cand)
            concept_items.append(
                ConceptItem(
                    concept_id=concept_id,
                    label=cand,
                    aliases=[],
                    type="identity" if subject == "orion" else "relationship",
                    salience=1.0,
                    confidence=0.6 if not embeddings_resp.embeddings else 0.75,
                    embedding_ref=f"embedding:{concept_id}"
                    if cand in embeddings_resp.embeddings
                    else None,
                    evidence=evidence[:5],
                    metadata={"source": "spacy", "extraction": "ner+chunks"},
                )
            )

        cluster_models: List[ConceptCluster] = []
        for idx, group in enumerate(clusters.clusters):
            cluster_id = f"cluster-{idx}"
            cluster_models.append(
                ConceptCluster(
                    cluster_id=cluster_id,
                    label=clusters.labels.get(idx, f"cluster-{idx}"),
                    summary=", ".join(group[:3]),
                    concept_ids=[make_concept_id(c) for c in group],
                    cohesion_score=0.8 if embeddings_resp.embeddings else 0.5,
                    metadata={"size": str(len(group))},
                )
            )

        state_estimate = self._build_state_estimate(window, window_start, window_end)

        metadata = {
            "extraction_debug": extraction.debug,
            "embedding_error": embeddings_resp.error,
            "algorithm": "concept_induction.v1",
        }

        profile = ConceptProfile(
            subject=subject,
            revision=1,
            window_start=window_start,
            window_end=window_end,
            concepts=concept_items[: self.settings.max_candidates],
            clusters=cluster_models,
            state_estimate=state_estimate,
            metadata=metadata,
        )

        # Optional refinement
        if self.summarizer:
            summary = await self.summarizer.summarize(
                subject=subject,
                candidates=[c.label for c in concept_items],
                clusters=[c.concept_ids for c in cluster_models],
                evidence=[str(e.message_id) for e in evidence],
            )
            profile.metadata["summary"] = summary

        previous_profile = None
        if self.store_loader:
            try:
                previous_profile = self.store_loader(subject)
            except Exception:  # noqa: BLE001
                previous_profile = None
        if previous_profile:
            profile.revision = previous_profile.revision + 1

        profile_hash = self._hash_profile(profile)
        delta = self._compute_delta(profile, previous_profile)

        if self.store_saver:
            try:
                self.store_saver(subject, profile, profile_hash)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to persist profile: %s", exc)

        return InductionResult(profile=profile, delta=delta)
