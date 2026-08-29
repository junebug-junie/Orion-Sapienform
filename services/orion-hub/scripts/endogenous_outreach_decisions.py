"""Durable decision log for endogenous outreach.

Mirrors `hub_presence.py`'s write shape: best-effort, fire-and-forget onto a
daemon thread, never raises, never blocks the caller. Apply
`services/orion-sql-db/manual_migration_endogenous_outreach_decisions_v1.sql`
before expecting rows; without it (or without `POSTGRES_URI`) this module is
a silent no-op.

WHY A SEPARATE MODULE, NOT INLINE IN `endogenous_outreach.py`: the write is a
single INSERT with no read-modify-write, no upsert, and no dependency on any
other outreach state -- a plain function is the whole contract, same
narrow-module shape `hub_presence.py` and `curiosity_hint.py` already use for
this file's other cross-cutting reads/writes. Keeping it here also lets
tests patch `record_decision` as one seam instead of reaching into
`EndogenousOutreach`'s internals.

NOT rate-limited (unlike `hub_presence.py`'s writer), and this is a real,
disclosed gap, not a settled tradeoff: the periodic `_run()` loop alone
would throttle writes to roughly once per `EndogenousOutreach.
tick_interval_sec` (floor 5s), but `maybe_outreach`'s `already_sending`
early-return also calls this module, and that branch is reachable at
UNBOUNDED rate via the unauthenticated `POST /api/debug/endogenous-outreach/
trigger` while a slow `_generate()` call is in flight (up to
`HUB_ENDOGENOUS_OUTREACH_TIMEOUT_SEC`, which can be minutes) -- each such
call spawns a new thread and a new engine connection. Review finding,
2026-08-22: accepted as-is for now (this endpoint's unauthenticated nature
is an existing, separately-documented risk this module did not introduce,
and every write here is still best-effort/fire-and-forget with no unbounded
in-process accumulation), but a real rate limit on this path is a fair
follow-up if it turns out to matter in practice.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Optional
from uuid import uuid4

logger = logging.getLogger("orion-hub.endogenous_outreach_decisions")


def _write_decision_to_postgres(
    *,
    decision_id: str,
    result: Dict[str, Any],
    target_id: Optional[str],
    run_length: Optional[int],
    peak_deviation_pressure: Optional[float],
    sustained_load_pressure: Optional[float],
    forced: bool,
) -> None:
    try:
        import json

        from scripts.pg_engine import get_engine
        from sqlalchemy import text

        # Shared, process-lifetime cached engine (scripts/pg_engine.py) --
        # NOT disposed here, same contract that module's own docstring
        # states and `tension_outreach_trigger.py`/`_fetch_recent_turns`
        # already follow, rather than this module building and tearing
        # down a private pool on every single write (review finding,
        # 2026-08-22: an earlier version of this function did exactly
        # that, inconsistent with the sibling debug-read route added in
        # the same commit, which already used this shared engine).
        engine = get_engine()
        if engine is None:
            return
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO endogenous_outreach_decisions (
                        decision_id, outreach, reason, forced, target_id,
                        run_length, peak_deviation_pressure,
                        sustained_load_pressure, correlation_id, session_id,
                        result_json
                    ) VALUES (
                        :decision_id, :outreach, :reason, :forced, :target_id,
                        :run_length, :peak_deviation_pressure,
                        :sustained_load_pressure, :correlation_id, :session_id,
                        CAST(:result_json AS jsonb)
                    )
                    """
                ),
                {
                    "decision_id": decision_id,
                    "outreach": bool(result.get("outreach", False)),
                    "reason": str(result.get("reason") or "unknown"),
                    "forced": bool(forced),
                    "target_id": target_id,
                    "run_length": run_length,
                    "peak_deviation_pressure": peak_deviation_pressure,
                    "sustained_load_pressure": sustained_load_pressure,
                    "correlation_id": result.get("correlation_id"),
                    "session_id": result.get("session_id"),
                    "result_json": json.dumps(result),
                },
            )
    except Exception as exc:
        logger.warning("endogenous_outreach_decision_write_failed error=%s", exc)


def decision_log_enabled() -> bool:
    """Whether the decision log is switched on. Shared by the writer and the
    reader so the two can never disagree about whether rows exist."""
    flag = os.getenv("HUB_ENDOGENOUS_OUTREACH_DECISION_LOG_ENABLED", "true").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def record_decision(
    result: Dict[str, Any],
    *,
    tension_reason: Optional[Any] = None,
    forced: bool = False,
) -> None:
    """Persist one decision cycle's outcome. Best-effort, never raises,
    never blocks the caller -- the write runs on a daemon thread exactly
    like `hub_presence.py::record_turn`'s Postgres mirror.

    ``tension_reason`` is `scripts.tension_outreach_trigger.TensionTriggerReason`
    or `None` (typed loosely to match this file's existing lazy-import
    convention for cross-Hub-module dependencies) -- whatever
    `EndogenousOutreach._last_tension_reason` held at the moment this
    decision cycle's outcome was recorded. `None` on a blocked tick, a
    forced debug trigger, or an organic tick that never fired.
    """
    try:
        if not decision_log_enabled():
            return
        decision_id = str(uuid4())
        target_id = getattr(tension_reason, "target_id", None)
        run_length = getattr(tension_reason, "run_length", None)
        peak_deviation_pressure = getattr(tension_reason, "peak_deviation_pressure", None)
        sustained_load_pressure = getattr(tension_reason, "sustained_load_pressure", None)
        threading.Thread(
            target=_write_decision_to_postgres,
            kwargs={
                "decision_id": decision_id,
                "result": dict(result),
                "target_id": target_id,
                "run_length": run_length,
                "peak_deviation_pressure": peak_deviation_pressure,
                "sustained_load_pressure": sustained_load_pressure,
                "forced": forced,
            },
            name="hub-endogenous-outreach-decision-writer",
            daemon=True,
        ).start()
    except Exception as exc:
        logger.warning("endogenous_outreach_decision_record_failed error=%s", exc)


def count_sent_on(local_date: str, tz_name: str) -> Optional[int]:
    """How many outreaches were already delivered on ``local_date``.

    Returns ``None`` — meaning UNKNOWN, never zero — when the table, the
    engine, or `POSTGRES_URI` is unavailable. The caller must not read a
    failure as "nothing sent yet": that is precisely the bug this exists to
    close, and reading absence as zero would reintroduce it silently.

    WHY THIS READ EXISTS: `EndogenousOutreach._sent_today` is an in-process
    counter initialised to 0 in `__init__` and reset only on a day rollover.
    Nothing rehydrated it, so **every container restart granted Orion a fresh
    daily cap.** Confirmed live 2026-08-28: 4 sends by 12:29 MDT, then
    `daily_cap` blocking at 20:12, then a 5th send at 20:54 -- four minutes
    after a deploy restart, with no day rollover in between. Deploys happen
    several times a day in this repo, so the cap that is supposed to bound
    interruptions at `daily_cap` per day was bounded by nothing.

    A LOWER BOUND, not an exact reconstruction: `record_decision` is
    fire-and-forget on a daemon thread and swallows its INSERT failure, so a
    delivered message whose row never lands is a send with no row. That errs
    toward under-counting, i.e. toward allowing an extra send -- the same
    direction as the bug being fixed, so it narrows the gap without closing
    it completely.

    Counts `reason='sent'` rows, which is otherwise the same set the counter
    increments:
    both the organic tick and `offer_message` (the curiosity loop) bump it,
    because the cap is deliberately SHARED -- from the receiving end they are
    the same interruption.
    """
    if not decision_log_enabled():
        # UNKNOWN, not zero. With the log switched off the table stops
        # receiving rows while this read still succeeds, so returning 0 would
        # mark the count "recovered" at zero and leave the cap unenforced for
        # the rest of the day -- the exact bug this function exists to close,
        # reachable by flipping one env key.
        return None
    try:
        from scripts.pg_engine import get_engine
        from sqlalchemy import text

        engine = get_engine()
        if engine is None:
            return None
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT count(*) FROM endogenous_outreach_decisions "
                    "WHERE reason = 'sent' "
                    "AND (decided_at AT TIME ZONE :tz)::date = CAST(:d AS date)"
                ),
                {"tz": tz_name, "d": local_date},
            ).scalar()
        return None if row is None else int(row)
    except Exception as exc:  # noqa: BLE001
        logger.warning("endogenous_outreach_count_sent_failed date=%s: %s", local_date, exc)
        return None
