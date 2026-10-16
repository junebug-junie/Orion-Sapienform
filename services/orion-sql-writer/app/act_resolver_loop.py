"""Clocked horizon resolver for the act ledger (orion/grammar/act_ledger.py).

Absence is a non-event: "Juniper did not reply within the horizon" and "the
painting was never reported as produced" can only be observed by a clock, so
they cannot ride the grammar stream. Every SQL_WRITER_ACT_RESOLVER_INTERVAL_SEC
this loop closes due acts and logs one summary line ([ACT_RESOLVER]).
"""

from __future__ import annotations

import asyncio
import logging

from app.settings import Settings

logger = logging.getLogger("sql-writer.act_resolver_loop")


def run_one_resolve() -> dict:
    from app.db import session_factory
    from orion.grammar.act_ledger import resolve_due_acts

    sess = session_factory()
    try:
        summary = resolve_due_acts(sess)
        sess.commit()
        return summary.__dict__
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


async def act_resolver_loop(settings: Settings) -> None:
    interval = float(getattr(settings, "sql_writer_act_resolver_interval_sec", 0.0) or 0.0)
    if interval <= 0:
        logger.info("act_resolver_loop DISABLED (SQL_WRITER_ACT_RESOLVER_INTERVAL_SEC=%s)", interval)
        return
    logger.info("act_resolver_loop starting interval_sec=%.0f", interval)
    while True:
        await asyncio.sleep(interval)
        try:
            summary = await asyncio.to_thread(run_one_resolve)
            if summary.get("examined"):
                logger.info("[ACT_RESOLVER] tick_complete %s", summary)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("act_resolver_cycle_failed error=%s", exc)
