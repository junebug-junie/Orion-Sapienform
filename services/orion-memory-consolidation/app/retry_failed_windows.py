from __future__ import annotations

import asyncio
import logging

from app.settings import settings
from app.window_state import WindowStore
from app.worker import ConsolidationSuggestRunner

logger = logging.getLogger(__name__)

_MAX_FAILED_WINDOW_RETRIES = 3
_failure_counts: dict[str, int] = {}


async def retry_failed_windows(*, pool, bus, window_store: WindowStore, suggest_runner: ConsolidationSuggestRunner) -> None:
    rows = await window_store.list_failed_windows(limit=20)
    for row in rows:
        window_id = str(row["memory_window_id"])
        prior_failures = _failure_counts.get(window_id, 0)
        if prior_failures >= _MAX_FAILED_WINDOW_RETRIES:
            logger.info(
                "memory_consolidation_retry_skip window_id=%s prior_failures=%s",
                window_id,
                prior_failures,
            )
            continue
        turns = await window_store.get_window_turns(window_id)
        closed = {
            "memory_window_id": window_id,
            "turn_correlation_ids": [t.get("correlation_id") for t in turns if isinstance(t, dict)],
            "turns": [t for t in turns if isinstance(t, dict)],
        }
        await suggest_runner.consolidate_window(closed, bus=bus)
        status = await pool.fetchval(
            "SELECT consolidation_status FROM memory_consolidation_windows WHERE memory_window_id = $1",
            window_id,
        )
        if str(status or "") == "ok":
            _failure_counts.pop(window_id, None)
            continue
        if str(status or "") == "failed":
            _failure_counts[window_id] = prior_failures + 1
            logger.warning(
                "memory_consolidation_retry_failed window_id=%s attempt=%s",
                window_id,
                _failure_counts[window_id],
            )


async def run_retry_loop(*, pool, bus, window_store: WindowStore, suggest_runner: ConsolidationSuggestRunner) -> None:
    interval = max(60, int(settings.MEMORY_FAILED_RETRY_INTERVAL_SEC))
    while True:
        try:
            await retry_failed_windows(
                pool=pool,
                bus=bus,
                window_store=window_store,
                suggest_runner=suggest_runner,
            )
        except Exception:
            logger.exception("memory_failed_window_retry_loop_error")
        await asyncio.sleep(interval)
