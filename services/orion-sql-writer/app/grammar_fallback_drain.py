"""Replay grammar events shed to bus_fallback_log back into the grammar ledger.

When a grammar lane's queue is full the writer parks the event in ``bus_fallback_log`` with
error ``grammar queue full`` instead of blocking the bus. 2026-09-29: four ``harness_motor``
traces (~900 events in ~10s each) shed 722 events this way and nothing ever put them back.

This loop is the retry. It only runs while every lane is idle, so replay never competes with
live traffic, and it deletes a fallback row only after its trace batch was applied. Replay is
safe to repeat: the ledger dedupes on ``event_id`` (orion/grammar/ledger.py).
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from sqlalchemy import delete, select, update

from app.db import get_session, remove_session
from app.models.fallback_log import BusFallbackLog
from orion.schemas.grammar import GrammarEventV1

logger = logging.getLogger("sql-writer.grammar_fallback_drain")

GRAMMAR_KIND = "grammar.event.v1"
INVALID_ERROR = "grammar drain: invalid payload"


def _fetch_rows(limit: int) -> list[tuple[int, Any]]:
    from app.worker import GRAMMAR_QUEUE_FULL_ERROR

    sess = get_session()
    try:
        stmt = (
            select(BusFallbackLog.id, BusFallbackLog.payload)
            .where(BusFallbackLog.kind == GRAMMAR_KIND, BusFallbackLog.error == GRAMMAR_QUEUE_FULL_ERROR)
            .order_by(BusFallbackLog.id)
            .limit(limit)
        )
        return [(r[0], r[1]) for r in sess.execute(stmt).all()]
    finally:
        sess.close()
        remove_session()


def _finish_rows(delete_ids: list[int], invalid_ids: list[int]) -> None:
    sess = get_session()
    try:
        if delete_ids:
            sess.execute(delete(BusFallbackLog).where(BusFallbackLog.id.in_(delete_ids)))
        if invalid_ids:
            # Re-labelled, not deleted: an unparseable payload stays inspectable but can no
            # longer match the drain filter and be retried forever.
            sess.execute(update(BusFallbackLog).where(BusFallbackLog.id.in_(invalid_ids)).values(error=INVALID_ERROR))
        sess.commit()
    finally:
        sess.close()
        remove_session()


async def drain_once(batch: int) -> dict[str, int]:
    """One bounded replay pass. Returns counts; never raises for a single bad trace."""
    from app import worker as worker_mod
    from app.grammar_ledger_handler import cancel_active_grammar_persist, persist_grammar_trace_batch

    if worker_mod.grammar_queue_snapshot()["total_depth"] > 0:
        return {"skipped_busy": 1, "replayed": 0, "invalid": 0, "failed": 0}

    rows = await asyncio.to_thread(_fetch_rows, batch)
    by_trace: dict[str, list[tuple[int, GrammarEventV1]]] = defaultdict(list)
    invalid_ids: list[int] = []
    for row_id, payload in rows:
        try:
            event = GrammarEventV1.model_validate(payload)
        except Exception:
            invalid_ids.append(row_id)
            continue
        by_trace[event.trace_id].append((row_id, event))

    delete_ids: list[int] = []
    failed = 0
    timeout = max(
        float(worker_mod.settings.sql_writer_grammar_persist_timeout_sec),
        float(worker_mod.settings.sql_writer_grammar_trace_batch_timeout_sec),
    )
    loop = asyncio.get_running_loop()
    for trace_id, pairs in by_trace.items():
        if worker_mod.grammar_queue_snapshot()["total_depth"] > 0:
            break  # live traffic arrived; leave the rest for the next pass
        shard = worker_mod._grammar_shard_index(trace_id)
        events = [ev for _, ev in pairs]
        # Same single-thread executor the lane worker uses, so replay serialises with it.
        fut = loop.run_in_executor(worker_mod._get_grammar_executors()[shard], persist_grammar_trace_batch, events, shard)
        try:
            await asyncio.wait_for(fut, timeout=timeout)
            delete_ids.extend(rid for rid, _ in pairs)
        except asyncio.TimeoutError:
            cancel_active_grammar_persist(shard)
            failed += len(pairs)
            logger.error("grammar_drain_timeout trace_id=%s events=%s", trace_id, len(pairs))
        except Exception as exc:
            failed += len(pairs)
            logger.error("grammar_drain_failed trace_id=%s events=%s error=%s", trace_id, len(pairs), exc)

    if delete_ids or invalid_ids:
        await asyncio.to_thread(_finish_rows, delete_ids, invalid_ids)
    if delete_ids or invalid_ids or failed:
        logger.info(
            "grammar_drain replayed=%s invalid=%s failed=%s scanned=%s",
            len(delete_ids), len(invalid_ids), failed, len(rows),
        )
    return {"skipped_busy": 0, "replayed": len(delete_ids), "invalid": len(invalid_ids), "failed": failed}


async def grammar_fallback_drain_loop(settings: Any) -> None:
    interval = float(settings.sql_writer_grammar_drain_interval_sec)
    batch = max(1, int(settings.sql_writer_grammar_drain_batch))
    while True:
        await asyncio.sleep(interval)
        try:
            await drain_once(batch)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("grammar_drain_cycle_failed")
