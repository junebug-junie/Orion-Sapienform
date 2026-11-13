"""The self-monitor E1 act gate for a loop's own acts (orion/self_monitor/gate.py), built from settings.

Moved from services/orion-mind/app/prime_loop.py (2026-11-04). Every instance in orion-mind shares the same
SELF_MONITOR_* settings; the gate itself keys faculties by actor and verb (``act.<actor>.<verb>[.<name>]``).
"""

from __future__ import annotations

from typing import Any

from orion.self_monitor.gate import ActGate, GateDecision, last_attempt_sql, read_view_sync

# A refused act is asked again at most this often (the tick is 15 s; the answer changes at most hourly,
# when the self-monitor publishes).
GATE_RECHECK_SEC = 60.0
_LEDGER_CONNECT_TIMEOUT_SEC = 3
_LEDGER_STATEMENT_TIMEOUT_MS = 5000


def _ledger_rows(dsn: str, actor: str, verb: str, secs: float) -> list[dict[str, Any]]:
    """grammar_acts rows for the gate's probe and same-job bookkeeping. Raises on failure; the gate
    then lets its in-process memo decide (never "no attempt")."""
    import psycopg2  # inside: a missing driver degrades to an unreadable ledger
    import psycopg2.extras

    conn = psycopg2.connect(dsn, connect_timeout=_LEDGER_CONNECT_TIMEOUT_SEC,
                            options=f"-c statement_timeout={_LEDGER_STATEMENT_TIMEOUT_MS}")
    try:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(last_attempt_sql("pyformat"), {"actor": actor, "verb": verb, "secs": float(secs)})
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def build_act_gate(settings: Any) -> ActGate:
    """The E1 gate. No DSN: the ledger is unreadable and the memo alone decides. An act's job is its
    ``object_ref`` (for Prime's concern act ``prime_concern:<id>``): a retry of a started open is the same job."""
    dsn = str(getattr(settings, "SELF_MONITOR_LEDGER_PG_DSN", "") or "").strip()
    return ActGate(
        enabled=bool(getattr(settings, "SELF_MONITOR_ACT_GATE_ENABLED", False)),
        probe_interval_sec=float(getattr(settings, "SELF_MONITOR_PROBE_INTERVAL_SEC", 21600.0)),
        read_view=lambda now: read_view_sync(str(getattr(settings, "ORION_BUS_URL", "") or ""), now=now),
        ledger_rows=lambda actor, verb, secs: _ledger_rows(dsn, actor, verb, secs) if dsn else None,
        job_key_of=lambda row: row.get("object_ref"),
    )


def gate_why(d: GateDecision, interval_sec: float) -> str:
    return (f"my own record says these have been {d.status} lately (self-monitor: track_record_failing); "
            f"one is re-tested every {interval_sec / 3600:g} h")


__all__ = ["GATE_RECHECK_SEC", "build_act_gate", "gate_why"]
