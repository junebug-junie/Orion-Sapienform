"""Canonical read-only Postgres connection helper.

Moved here 2026-08-19 (review finding on `orion/metrics/liveness.py`, phase 5
of the metric semantic layer): `scripts/analysis/_pg_readonly.py` already
existed with this exact contract -- factored out 2026-08-11 specifically to
stop `open_readonly_connection` being duplicated byte-for-byte across
`measure_*.py` probes -- but `orion/metrics/liveness.py` couldn't import it
without inverting this repo's layering (`orion/` must not depend on
`scripts/`), so it grew its own near-identical copy instead. That is exactly
the duplication `_pg_readonly.py` exists to prevent, just moved one layer
down instead of stopped.

This module is the real canonical version, in a layer both `orion/` and
`scripts/` can depend on. `scripts/analysis/_pg_readonly.py` now re-exports
this rather than defining its own copy, so its one real caller
(`measure_goal_provenance_streak_distribution.py`) is unaffected.

Read-only-session enforcement: refuses to return a connection unless
`default_transaction_read_only` is actually `on` for the session -- a
defense against a probe/tool script accidentally being pointed at write
credentials and mutating real data.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("orion.db_readonly")


def open_readonly_connection(
    dsn: str,
    *,
    connect_timeout: float | None = None,
    statement_timeout_ms: int | None = None,
):
    """Returns a psycopg2 connection with a confirmed read-only session, or
    `None` on any failure (psycopg2 unavailable, connection refused/timed
    out, or a session that cannot be confirmed read-only). Callers must treat
    `None` as "unknown", never as evidence of anything about the target.

    `connect_timeout` is optional and defaults to psycopg2's own default
    (none -- an unreachable, not merely refused, host can hang indefinitely).
    `statement_timeout_ms` (session-level `SET statement_timeout`) is
    likewise optional and unset by default -- added 2026-08-20 (review
    finding): `connect_timeout` alone only bounds the initial TCP connect; a
    query that hangs AFTER connecting (lock contention, an unindexed scan)
    reproduces the exact same "call hangs indefinitely" failure one phase
    later. Existing callers that pass neither keep their exact prior
    behavior -- unbounded by design, e.g. `measure_goal_provenance_streak_
    distribution.py`'s deliberately large historical scans.
    """
    try:
        import psycopg2
    except Exception:  # pragma: no cover
        logger.error("psycopg2 unavailable; cannot open DB session")
        return None
    try:
        kwargs = {"connect_timeout": connect_timeout} if connect_timeout is not None else {}
        conn = psycopg2.connect(dsn, **kwargs)
    except Exception:
        logger.error("failed to connect to postgres", exc_info=True)
        return None
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SET default_transaction_read_only = on;")
            if statement_timeout_ms is not None:
                cur.execute("SET statement_timeout = %s;", (statement_timeout_ms,))
            cur.execute("SHOW default_transaction_read_only;")
            value = cur.fetchone()
        if not value or str(value[0]).lower() != "on":
            logger.error("refusing to run: session is not read-only (got %r)", value)
            conn.close()
            return None
    except Exception:
        logger.error("failed to enforce read-only session", exc_info=True)
        try:
            conn.close()
        except Exception:
            pass
        return None
    return conn
