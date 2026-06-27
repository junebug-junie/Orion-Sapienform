"""Re-classify turns that degraded or never received turn_change_appraisal."""

from __future__ import annotations

import asyncio
import json
import logging

from orion.core.bus.async_service import OrionBusAsync
from orion.schemas.memory_consolidation import MemoryTurnPersistedV1

from app.classify import classify_turn
from app.settings import settings
from app.window_state import WindowStore
from app.worker import ConsolidationSuggestRunner, publish_spark_meta_patch

logger = logging.getLogger(__name__)

_MAX_CLASSIFY_RETRIES = 5
_retry_counts: dict[str, int] = {}


def _spark_meta_dict(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


async def _prior_turns_for(pool, *, session_id: str | None, created_at) -> list[dict]:
    if not session_id or created_at is None:
        return []
    rows = await pool.fetch(
        """
        SELECT correlation_id, prompt, response, spark_meta
        FROM chat_history_log
        WHERE session_id = $1
          AND created_at < $2
          AND coalesce(trim(prompt), '') <> ''
          AND coalesce(trim(response), '') <> ''
        ORDER BY created_at ASC
        LIMIT 20
        """,
        session_id,
        created_at,
    )
    out: list[dict] = []
    for row in rows:
        out.append(
            {
                "correlation_id": str(row["correlation_id"]),
                "prompt": row["prompt"],
                "response": row["response"],
                "spark_meta": _spark_meta_dict(row["spark_meta"]),
            }
        )
    return out


async def retry_degraded_classifies(
    *,
    pool,
    bus: OrionBusAsync,
    window_store: WindowStore,
    suggest_runner: ConsolidationSuggestRunner,
) -> None:
    rows = await pool.fetch(
        """
        SELECT correlation_id, prompt, response, spark_meta, session_id, created_at
        FROM chat_history_log
        WHERE coalesce(trim(prompt), '') <> ''
          AND coalesce(trim(response), '') <> ''
          AND created_at > now() - interval '7 days'
          AND (
            spark_meta->'turn_change_appraisal' IS NULL
            OR spark_meta->'turn_change_appraisal'->>'turn_change_status' = 'degraded'
          )
          AND coalesce(spark_meta->'turn_change_appraisal'->>'turn_change_status', '') <> 'skipped'
        ORDER BY created_at DESC
        LIMIT 10
        """
    )
    for row in rows:
        corr = str(row["correlation_id"])
        prior_failures = _retry_counts.get(corr, 0)
        if prior_failures >= _MAX_CLASSIFY_RETRIES:
            continue
        prior_turns = await _prior_turns_for(
            pool,
            session_id=row["session_id"],
            created_at=row["created_at"],
        )
        turn = MemoryTurnPersistedV1(
            correlation_id=corr,
            prompt=str(row["prompt"] or ""),
            response=str(row["response"] or ""),
            spark_meta=_spark_meta_dict(row["spark_meta"]),
            session_id=row["session_id"],
        )
        try:
            patch_fields = await classify_turn(
                bus, turn=turn, prior_turns=prior_turns, settings=settings
            )
        except Exception:
            logger.exception("memory_classify_retry_failed corr=%s", corr)
            _retry_counts[corr] = prior_failures + 1
            continue
        status = patch_fields.get("memory_classify_status")
        await publish_spark_meta_patch(bus, corr, patch_fields)
        if status == "ok":
            _retry_counts.pop(corr, None)
            logger.info("memory_classify_retry_ok corr=%s", corr)
        else:
            _retry_counts[corr] = prior_failures + 1
            logger.warning(
                "memory_classify_retry_still_degraded corr=%s attempt=%s",
                corr,
                _retry_counts[corr],
            )


async def run_classify_retry_loop(
    *,
    pool,
    bus: OrionBusAsync,
    window_store: WindowStore,
    suggest_runner: ConsolidationSuggestRunner,
) -> None:
    interval = max(60, int(settings.MEMORY_CLASSIFY_RETRY_INTERVAL_SEC))
    while True:
        try:
            await retry_degraded_classifies(
                pool=pool,
                bus=bus,
                window_store=window_store,
                suggest_runner=suggest_runner,
            )
        except Exception:
            logger.exception("memory_classify_retry_loop_error")
        await asyncio.sleep(interval)
