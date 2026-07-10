from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from orion.cognition.chat_history_compactor.constants import CHAT_DEV_DIGEST_TAG, COMPACTOR_KIND
from orion.cognition.chat_history_compactor.window import ResolvedChatCompactorWindow
from orion.core.contracts.memory_cards import EvidenceItemV1, MemoryCardCreateV1, TimeHorizonV1
from orion.core.storage import memory_cards as mc_dal
from orion.schemas.actions.chat_history_compactor import ChatHistoryCompactorDigestV1

from .memory_extractor import _get_memory_pool

logger = logging.getLogger("orion.cortex.chat_history_compactor_memory")


async def persist_chat_history_compactor_memory_card(
    *,
    digest: ChatHistoryCompactorDigestV1,
    window: ResolvedChatCompactorWindow,
    turn_count: int,
    actor: str = "chat_history_compactor_pass",
) -> Optional[UUID]:
    pool = await _get_memory_pool()
    if pool is None:
        raise RuntimeError("recall_pg_dsn_unavailable")

    window_label = window.calendar_date or (
        f"{window.lookback_hours}h" if window.lookback_hours else "window"
    )
    evidence = [
        EvidenceItemV1(source=ref, ts=window_label)
        for ref in (digest.turn_refs or [])[:12]
    ]
    card = MemoryCardCreateV1(
        types=["fact"],
        anchor_class="event",
        status="active",
        priority="high_recall",
        provenance="chat_compactor",
        title=f"Chat digest ({window_label})",
        summary=digest.card_summary,
        tags=[CHAT_DEV_DIGEST_TAG],
        time_horizon=TimeHorizonV1(
            kind="era_bound",
            start=window.window_start.isoformat(),
            end=window.window_end.isoformat(),
        ),
        evidence=evidence,
        subschema={
            "compactor_index": window.compactor_index,
            "compactor_kind": COMPACTOR_KIND,
            "window_mode": window.mode,
            "turn_count": int(turn_count),
            "window_start": window.window_start.isoformat(),
            "window_end": window.window_end.isoformat(),
        },
    )
    return await mc_dal.upsert_indexed_compactor_card(
        pool,
        index=window.compactor_index,
        card=card,
        actor=actor,
    )
