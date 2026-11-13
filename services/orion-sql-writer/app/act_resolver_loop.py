"""Clocked horizon resolver for the act ledger (orion/grammar/act_ledger.py).

Absence is a non-event: "Juniper did not reply within the horizon" and "the
painting was never reported as produced" can only be observed by a clock, so
they cannot ride the grammar stream. Every SQL_WRITER_ACT_RESOLVER_INTERVAL_SEC
this loop closes due acts and logs one summary line ([ACT_RESOLVER]).

First, each tick opens a ``vision.ask`` act for every camera ask Orion made that has none yet
(app/ask_acts.py, attention patch 4a; SQL_WRITER_ACT_CAMERA_ASKS_ENABLED), one [ASK_ACTS] line when it opened any.

The same tick fills the GPU and energy cost of acts that close before their work ends (a read's act closes
at stage 1; its stage 2 runs after): orion/grammar/act_ledger.py ``refill_late_costs``, one
[ACT_COST_LATE] line when it examined anything.
"""

from __future__ import annotations

import asyncio
import logging

from app.settings import Settings

logger = logging.getLogger("sql-writer.act_resolver_loop")


def run_one_open_ask_acts() -> int:
    """Open a vision.ask act for each camera ask Orion made that has none (app/ask_acts.py, patch 4a)."""
    from app.ask_acts import open_ask_acts
    from app.db import session_factory

    sess = session_factory()
    try:
        opened = open_ask_acts(sess)
        sess.commit()
        return len(opened)
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


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
    ask_acts = bool(getattr(settings, "sql_writer_act_camera_asks_enabled", True))
    logger.info("act_resolver_loop starting interval_sec=%.0f ask_acts=%s", interval, ask_acts)
    while True:
        await asyncio.sleep(interval)
        if ask_acts:
            # Before the resolve, so an ask answered within the minute opens and closes in one tick.
            try:
                opened = await asyncio.to_thread(run_one_open_ask_acts)
                if opened:
                    logger.info("[ASK_ACTS] opened=%d", opened)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("ask_acts_open_failed error=%s", exc)
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
