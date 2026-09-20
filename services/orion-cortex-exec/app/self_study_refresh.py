"""Periodic Layer-1 self-fact refresh (self-model rebuild arc follow-up,
2026-09-19).

Why this exists: ``self_repo_inspect`` is the only writer of
``self_knowledge_items``, the table Self Atlas clusters over. It is a
``@verb()`` action, and the design assumed Orion's own action-selection would
choose it. Live check on 2026-09-19: the table held exactly 1,248 rows, all
from one manual trigger on 2026-09-05 -- in two weeks no autonomous chooser
ever picked it, because none of them can (the three self-study verbs appear in
no priority/policy table: ``config/execution_dispatch/execution_dispatch_policy.v1.yaml``,
``config/proposals/proposal_policy.v1.yaml``, ``orion/substrate/endogenous_curiosity.py``
all score other things). Meanwhile the Hub's Self Atlas tick kept training a
fresh HDBSCAN run every UTC day over the identical frozen snapshot and writing
391 relabelled ``self_concept_history`` rows from it -- stale input read back
into Orion's own self-description as if it were new.

So the refresh is a plain timer inside this process, calling
``run_self_repo_inspect`` directly -- no bus hop, no orch, no verb-request
client (``scripts/platform_audits/audit_spine.py`` forbids any
``orion:verb:request`` emitter outside cortex-orch). The scan is pure
filesystem/YAML work plus one Postgres SELECT, no LLM -- seconds, not minutes.

Restart guard: every deploy restarts this container, and a naive "run at
startup" would append another 1,248 rows per deploy. The loop reads the newest
``self_knowledge_items.created_at`` first and only runs once a full interval
has passed since it, so a restart inside the interval sleeps out the remainder.

``self_study_reflect_refresh_loop`` (2026-09-20) is the same fix applied to
Layer 3 (``run_self_concept_reflect``), which had the identical disease and a
worse outcome: not "stale input read as new" but zero rows, ever, in
``self_concept_history`` under ``produced_by='layer3_reflect'``. Live check
that day found no autonomous or scheduled caller of it at all -- the two prior
failures that drove the ``SELF_STUDY_REFLECT_TIMEOUT_SEC`` 240->480 bump were
manual test invocations, and nothing has called it since. Reflect's own
snapshot/concept induction is independent of the ``self_knowledge_items``
table (``build_self_snapshot()`` is a fresh filesystem/YAML scan each call,
same as inspect), so this loop does not need to wait on the inspect loop --
it runs on its own interval, measured the same restart-safe way against the
newest ``self_concept_history`` row it has actually written. A run whose LLM
call fails deliberately writes no ``self_concept_history`` row (see
``run_self_concept_reflect``'s no-empty-shell-cognition guard), so a run of
failures is retried once per interval rather than spun in a tight loop.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("orion.cortex_exec.self_study_refresh")


def newest_self_knowledge_item_at() -> Optional[datetime]:
    """Newest ``self_knowledge_items.created_at`` (tz-aware UTC), or ``None``
    if the table is empty or the DB is unreachable. Never raises -- an
    unreadable table means "no evidence of a recent run", so the loop runs."""
    from app.self_study_analysis import _get_engine
    from sqlalchemy import text

    from orion.self_knowledge_freshness import NEWEST_SELF_KNOWLEDGE_ITEM_SQL, normalize_newest

    engine = _get_engine()
    if engine is None:
        return None
    try:
        with engine.connect() as conn:
            row = conn.execute(text(NEWEST_SELF_KNOWLEDGE_ITEM_SQL)).mappings().first()
    except Exception as exc:
        logger.debug("self_study_refresh_newest_unavailable error=%s", exc)
        return None
    return normalize_newest(row["newest"] if row else None)


def newest_self_concept_reflection_at() -> Optional[datetime]:
    """Newest ``self_concept_history.created_at`` where
    ``produced_by='layer3_reflect'`` (tz-aware UTC), or ``None`` if no
    reflection has ever been written or the DB is unreachable. Never raises --
    same "unreadable means run" contract as ``newest_self_knowledge_item_at``."""
    from app.self_study_analysis import _get_engine
    from sqlalchemy import text

    from orion.self_knowledge_freshness import NEWEST_SELF_CONCEPT_REFLECTION_SQL, normalize_newest

    engine = _get_engine()
    if engine is None:
        return None
    try:
        with engine.connect() as conn:
            row = conn.execute(text(NEWEST_SELF_CONCEPT_REFLECTION_SQL)).mappings().first()
    except Exception as exc:
        logger.debug("self_study_reflect_refresh_newest_unavailable error=%s", exc)
        return None
    return normalize_newest(row["newest"] if row else None)


def seconds_until_due(newest: Optional[datetime], *, interval_sec: float, now: datetime) -> float:
    """How long to wait before the next run is due. Zero when there is no
    prior run on record, or when a full interval has already elapsed."""
    if newest is None:
        return 0.0
    elapsed = (now - newest).total_seconds()
    return max(0.0, float(interval_sec) - elapsed)


async def self_study_refresh_loop(
    *,
    bus_getter: Callable[[], Any],
    source: Any,
    interval_sec: float,
    run_inspect: Callable[..., Awaitable[Any]],
    newest_at: Callable[[], Optional[datetime]] = newest_self_knowledge_item_at,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> None:
    """Run ``run_inspect`` once per ``interval_sec``, measured from the newest
    stored item rather than from process start. Fail-open: a failed run is
    logged and retried after a full interval, never re-raised."""
    interval = max(float(interval_sec), 60.0)
    while True:
        newest = await asyncio.to_thread(newest_at)
        wait = seconds_until_due(newest, interval_sec=interval, now=clock())
        if wait > 0:
            logger.info(
                "self_study_refresh_wait newest_item_at=%s wait_sec=%d interval_sec=%d",
                newest.isoformat() if newest else None,
                int(wait),
                int(interval),
            )
            await sleep(wait)
            continue
        correlation_id = str(uuid.uuid4())
        try:
            result = await run_inspect(bus=bus_getter(), source=source, correlation_id=correlation_id)
            items_write = getattr(result, "self_knowledge_items_write", None)
            logger.info(
                "self_study_refresh_ran corr=%s run_id=%s items_status=%s items_detail=%s",
                correlation_id,
                getattr(getattr(result, "snapshot", None), "run_id", None),
                getattr(items_write, "status", None),
                getattr(items_write, "detail", None),
            )
        except Exception:  # noqa: BLE001
            logger.warning("self_study_refresh_failed corr=%s", correlation_id, exc_info=True)
        await sleep(interval)


async def self_study_reflect_refresh_loop(
    *,
    bus_getter: Callable[[], Any],
    source: Any,
    interval_sec: float,
    run_reflect: Callable[..., Awaitable[Any]],
    newest_at: Callable[[], Optional[datetime]] = newest_self_concept_reflection_at,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> None:
    """Run ``run_reflect`` (``run_self_concept_reflect``) once per
    ``interval_sec``, measured from the newest stored reflection rather than
    from process start. Fail-open: a failed run is logged and retried after a
    full interval, never re-raised. See module docstring for why this exists."""
    interval = max(float(interval_sec), 60.0)
    while True:
        newest = await asyncio.to_thread(newest_at)
        wait = seconds_until_due(newest, interval_sec=interval, now=clock())
        if wait > 0:
            logger.info(
                "self_study_reflect_refresh_wait newest_reflection_at=%s wait_sec=%d interval_sec=%d",
                newest.isoformat() if newest else None,
                int(wait),
                int(interval),
            )
            await sleep(wait)
            continue
        correlation_id = str(uuid.uuid4())
        try:
            result = await run_reflect(bus=bus_getter(), source=source, correlation_id=correlation_id)
            history_write = getattr(result, "self_concept_history_write", None)
            logger.info(
                "self_study_reflect_refresh_ran corr=%s run_id=%s findings=%d history_status=%s history_detail=%s",
                correlation_id,
                getattr(result, "run_id", None),
                len(getattr(result, "findings", None) or []),
                getattr(history_write, "status", None),
                getattr(history_write, "detail", None),
            )
        except Exception:  # noqa: BLE001
            logger.warning("self_study_reflect_refresh_failed corr=%s", correlation_id, exc_info=True)
        await sleep(interval)
