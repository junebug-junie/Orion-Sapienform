"""Clocked horizon resolver for the act ledger (orion/grammar/act_ledger.py).

Absence is a non-event: "Juniper did not reply within the horizon" and "the
painting was never reported as produced" can only be observed by a clock, so
they cannot ride the grammar stream. Every SQL_WRITER_ACT_RESOLVER_INTERVAL_SEC
this loop closes due acts and logs one summary line ([ACT_RESOLVER]).

The same tick fills the GPU cost of acts that close before their work ends (a read's act closes
at stage 1; its stage 2 runs after): orion/grammar/act_ledger.py ``refill_late_costs``, one
[ACT_COST_LATE] line when it examined anything.
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
        summary = resolve_due_acts(sess, commit=sess.commit)
        sess.commit()
        return summary.__dict__
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


def run_one_late_cost_fill() -> dict:
    from app.db import session_factory
    from orion.grammar.act_ledger import refill_late_costs

    sess = session_factory()
    try:
        summary = refill_late_costs(sess, commit=sess.commit)
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
        try:
            late = await asyncio.to_thread(run_one_late_cost_fill)
            if late.get("examined"):
                logger.info("[ACT_COST_LATE] tick_complete %s", late)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("act_cost_late_cycle_failed error=%s", exc)
