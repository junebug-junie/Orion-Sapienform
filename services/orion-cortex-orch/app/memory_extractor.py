from __future__ import annotations

import logging
from typing import Any, Optional

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.core.contracts.memory_cards import MemoryCardCreateV1
from orion.core.storage import memory_cards as mc_dal
from orion.core.storage.memory_extraction import extract_candidates, fingerprint_from_candidate
from orion.schemas.chat_history import ChatHistoryTurnV1

from .settings import get_settings

logger = logging.getLogger("orion.cortex.memory_extractor")

_memory_pool: Optional[Any] = None
_memory_pool_failed: bool = False


async def _get_memory_pool() -> Optional[Any]:
    """Lazy asyncpg pool for memory_cards DAL (same RECALL_PG_DSN as Hub / recall)."""
    global _memory_pool, _memory_pool_failed
    if _memory_pool is not None:
        return _memory_pool
    if _memory_pool_failed:
        return None
    dsn = (get_settings().recall_pg_dsn or "").strip()
    if not dsn:
        return None
    try:
        import asyncpg  # type: ignore[import-untyped]

        _memory_pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=3)
        logger.info("memory_extractor_pg_pool_ready")
        return _memory_pool
    except Exception as exc:
        logger.warning("memory_extractor_pool_failed error=%s", exc)
        _memory_pool_failed = True
        return None


def _coerce_turn(env: BaseEnvelope) -> Optional[ChatHistoryTurnV1]:
    payload = env.payload
    if not isinstance(payload, dict):
        return None
    try:
        return ChatHistoryTurnV1.model_validate(payload)
    except Exception:
        logger.debug("memory_extractor_turn_parse_failed kind=%s", getattr(env, "kind", ""))
        return None


async def handle_memory_history_turn(env: BaseEnvelope) -> None:
    settings = get_settings()
    if not settings.orion_auto_extractor_enabled:
        return
    if settings.orion_auto_extractor_stage2_enabled:
        raise NotImplementedError("ORION_AUTO_EXTRACTOR_STAGE2_ENABLED is v1.5-only")

    turn = _coerce_turn(env)
    if turn is None or not (turn.prompt or "").strip():
        return

    pool = await _get_memory_pool()
    if pool is None:
        return

    candidates = extract_candidates(turn.prompt, speaker="user")
    for cand in candidates:
        fp = fingerprint_from_candidate(cand)
        if await mc_dal.card_exists_by_fingerprint(pool, fp):
            logger.debug("memory_extractor_skip_duplicate fp=%s", fp[:16])
            continue
        subschema = {"auto_extractor_fingerprint": fp}
        create = MemoryCardCreateV1(
            types=list(cand.types),
            anchor_class=cand.anchor_class,
            title=cand.summary,
            summary=cand.summary,
            provenance="auto_extractor",
            status="pending_review",
            subschema=subschema,
        )
        try:
            await mc_dal.insert_card(pool, create, actor="auto_extractor", op="create")
            logger.info(
                "memory_extractor_card_created fp=%s title=%r",
                fp[:16],
                cand.summary[:80] if cand.summary else "",
            )
        except Exception as exc:
            logger.warning("auto_extractor_insert_failed fp=%s err=%s", fp[:16], exc)
