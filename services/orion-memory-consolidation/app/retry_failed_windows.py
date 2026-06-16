from __future__ import annotations

import asyncio
import logging

from app.settings import settings
from app.window_state import WindowStore
from app.worker import ConsolidationSuggestRunner

logger = logging.getLogger(__name__)

async def retry_failed_windows(*, pool, bus, window_store: WindowStore, suggest_runner: ConsolidationSuggestRunner) -> None:
    rows = await window_store.list_failed_windows(limit=20)
    for row in rows:
        window_id = row["memory_window_id"]
        turns = await window_store.get_window_turns(window_id)
        closed = {
            "memory_window_id": window_id,
            "turn_correlation_ids": [t.get("correlation_id") for t in turns if isinstance(t, dict)],
            "turns": [t for t in turns if isinstance(t, dict)],
        }
        await suggest_runner.consolidate_window(closed, bus=bus)


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
