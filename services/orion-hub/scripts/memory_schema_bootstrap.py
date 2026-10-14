"""Boot-time memory schema apply that never fails boot and retries later.

Before 2026-10-14 the hub ran memory_cards.sql then memory_crystallizations.sql
inline in ``startup_event``. A ``lock_timeout`` on the first (a hub restart
during the nightly pg_dump, see orion/core/storage/schema_fingerprint.py)
logged a traceback and silently skipped the second; nothing tried again until
the next restart.

Now each file is attempted independently. A current schema is skipped by its
fingerprint without touching table locks. A failure is retried off the event
loop on a fixed delay, for longer than the ~55 min backup window, and each
outcome is logged with the schema name.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger("orion-hub.memory_schema_bootstrap")

# 75 x 60s = 75 min: outlasts the nightly backup (03:45 UTC, 48-57 min on
# 10-13/10-14) a hub restart can land inside.
DEFAULT_RETRY_ATTEMPTS = 75
DEFAULT_RETRY_DELAY_SEC = 60.0

ApplyFn = Callable[[str], Any]
SleepFn = Callable[[float], Awaitable[None]]


async def _attempt(name: str, apply: ApplyFn, dsn: str, attempt: int) -> bool:
    try:
        ran = await asyncio.to_thread(apply, dsn)
    except Exception as exc:  # noqa: BLE001 - boot must never fail on schema apply
        logger.warning("memory_schema_apply_failed name=%s attempt=%d error=%s", name, attempt, exc)
        return False
    logger.info("memory_schema_ready name=%s ddl_ran=%s attempt=%d", name, bool(ran), attempt)
    return True


async def ensure_memory_schemas(
    dsn: str,
    schemas: list[tuple[str, ApplyFn]],
    *,
    attempts: int = DEFAULT_RETRY_ATTEMPTS,
    delay_sec: float = DEFAULT_RETRY_DELAY_SEC,
    sleep: SleepFn = asyncio.sleep,
) -> tuple[dict[str, bool], "asyncio.Task[dict[str, bool]] | None"]:
    """Apply each schema once, in order; one failing never skips the others.

    The inline attempt is the fast path (normally a fingerprint skip in
    milliseconds). Schemas that fail are retried in one background task,
    ``delay_sec`` apart, up to ``attempts`` total tries each. Returns which
    schemas are ready now, and the retry task (None when nothing failed) --
    keep a reference to it so it is not garbage-collected mid-retry.
    """
    ready: dict[str, bool] = {}
    for name, apply in schemas:
        ready[name] = await _attempt(name, apply, dsn, 1)
    pending = [(name, apply) for name, apply in schemas if not ready[name]]
    if not pending or attempts <= 1:
        for name, _ in pending:
            logger.error("memory_schema_apply_gave_up name=%s attempts=%d", name, 1)
        return ready, None

    async def _retry_pending() -> dict[str, bool]:
        outcome = dict(ready)
        remaining = list(pending)
        for attempt in range(2, attempts + 1):
            await sleep(delay_sec)
            still = []
            for name, apply in remaining:
                if await _attempt(name, apply, dsn, attempt):
                    outcome[name] = True
                else:
                    still.append((name, apply))
            remaining = still
            if not remaining:
                break
        for name, _ in remaining:
            logger.error("memory_schema_apply_gave_up name=%s attempts=%d", name, attempts)
        return outcome

    return ready, asyncio.create_task(_retry_pending())
