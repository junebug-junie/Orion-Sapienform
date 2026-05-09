"""Concept induction + Spark context adapter — concept_induced tier.

Wraps ``build_concept_profile_repository`` + ``map_concept_profile_to_substrate``
so that the concept induction and Spark producer lanes can be registered in
ProducerRegistryV1.

Queries orion, relationship, and juniper profiles and returns a combined
``SubstrateGraphRecordV1`` with concept_induced tier nodes.
"""

from __future__ import annotations

import logging
from typing import Any

from datetime import datetime, timezone

from orion.core.schemas.cognitive_substrate import SubstrateGraphRecordV1
from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
)
from orion.substrate.adapters._common import make_temporal

logger = logging.getLogger("orion.substrate.relational.adapters.concept_induction_ctx")

_TIER_RANK = 3  # concept_induced
_SUBJECTS = ("orion", "relationship", "juniper")


def map_concept_induction_ctx_to_substrate(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:  # noqa: ARG001
    """Fetch concept profiles for all subjects and map to substrate nodes (concept_induced)."""
    try:
        from orion.spark.concept_induction.profile_repository import build_concept_profile_repository  # noqa: PLC0415 — lazy to avoid spacy at import time
        from orion.substrate.adapters.concept_induction import map_concept_profile_to_substrate  # noqa: PLC0415
    except ImportError as exc:
        logger.debug("concept_induction_ctx_adapter_import_failed error=%s", exc)
        return None

    try:
        repository = build_concept_profile_repository()
    except Exception as exc:
        logger.debug("concept_induction_ctx_adapter_init_failed error=%s", exc)
        return None

    correlation_id = str(ctx.get("correlation_id") or ctx.get("trace_id") or "")
    session_id = str(ctx.get("session_id") or "")
    observer = {
        "consumer": "concept_induction_ctx_adapter",
        "correlation_id": correlation_id,
        "session_id": session_id,
    }

    try:
        lookups = repository.list_latest(_SUBJECTS, observer=observer)
    except Exception as exc:
        logger.debug("concept_induction_ctx_adapter_fetch_failed error=%s", exc)
        return None

    all_nodes: list[Any] = []
    for lookup in lookups:
        profile = lookup.profile
        if profile is None:
            continue

        anchor = lookup.subject
        if anchor not in ("orion", "relationship", "juniper"):
            continue

        try:
            record = map_concept_profile_to_substrate(profile=profile, anchor_scope=anchor)
        except Exception as exc:
            logger.debug("concept_induction_ctx_map_failed subject=%s error=%s", anchor, exc)
            continue

        # Stamp tier_rank on provenance of each mapped node (adapter is unchanged, so we
        # post-process to apply the tier_rank field introduced by the unified layer).
        for node in record.nodes:
            patched_prov = node.provenance.model_copy(update={"tier_rank": _TIER_RANK})
            all_nodes.append(node.model_copy(update={"provenance": patched_prov}))

    return SubstrateGraphRecordV1(anchor_scope="orion", nodes=all_nodes) if all_nodes else None
