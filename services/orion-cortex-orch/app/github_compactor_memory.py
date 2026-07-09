from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from orion.cognition.github_compactor.constants import REPO_DEV_SNAPSHOT_SLOT, REPO_DEV_SNAPSHOT_TAG
from orion.core.contracts.memory_cards import EvidenceItemV1, MemoryCardCreateV1
from orion.core.storage import memory_cards as mc_dal
from orion.schemas.actions.github_compactor import GithubCompactorDigestV1

from .memory_extractor import _get_memory_pool

logger = logging.getLogger("orion.cortex.github_compactor_memory")


async def persist_github_compactor_memory_card(
    *,
    digest: GithubCompactorDigestV1,
    repo: str,
    window_label: str,
    merged_pr_count: int,
    actor: str = "github_compactor_pass",
) -> Optional[UUID]:
    pool = await _get_memory_pool()
    if pool is None:
        raise RuntimeError("recall_pg_dsn_unavailable")

    evidence = [
        EvidenceItemV1(source=ref, ts=window_label)
        for ref in (digest.pr_refs or [])[:12]
    ]
    card = MemoryCardCreateV1(
        types=["fact"],
        anchor_class="project",
        status="active",
        priority="high_recall",
        provenance="repo_compactor",
        project=repo,
        title=f"Recent repo development ({window_label})",
        summary=digest.card_summary,
        tags=[REPO_DEV_SNAPSHOT_TAG],
        evidence=evidence,
        subschema={
            "compactor_slot": REPO_DEV_SNAPSHOT_SLOT,
            "window_label": window_label,
            "merged_pr_count": merged_pr_count,
            "source_repo": repo,
        },
    )
    return await mc_dal.supersede_and_insert_compactor_card(
        pool,
        slot=REPO_DEV_SNAPSHOT_SLOT,
        card=card,
        actor=actor,
    )
