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

Not rate-limited (unlike `hub_presence.py`'s writer): every decision cycle
tick is already throttled by `EndogenousOutreach.tick_interval_sec` (floor
5s), so this cannot write faster than roughly once every 5 seconds even at
the tightest configured cadence -- nowhere near needing its own limiter.
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
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        return
    try:
        import json

        from sqlalchemy import create_engine, text

        engine = create_engine(uri, pool_pre_ping=True)
        try:
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
        finally:
            engine.dispose()
    except Exception as exc:
        logger.warning("endogenous_outreach_decision_write_failed error=%s", exc)


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
        flag = os.getenv("HUB_ENDOGENOUS_OUTREACH_DECISION_LOG_ENABLED", "true").strip().lower()
        if flag in {"0", "false", "no", "off"}:
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
