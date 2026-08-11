"""Shared read-only Postgres connection helper for scripts/analysis/'s measure_*.py probes.

Factored out 2026-08-11 (review fix on measure_goal_provenance_streak_distribution.py):
`open_readonly_connection` was already duplicated byte-for-byte across
measure_emergent_clustering_probe.py, measure_ast_hot_reducer.py, and
measure_capability_salience_coupling.py before this module existed -- adding a 4th copy
for the new streak-distribution probe would have made that four independent copies to keep
in sync by hand. This module stops the count growing further; it does not migrate the
three pre-existing copies (a separate, larger cleanup, out of scope for this patch).

Read-only-session enforcement: refuses to return a connection unless
`default_transaction_read_only` is actually `on` for the session -- a defense against a
probe script accidentally being pointed at write credentials and mutating real data.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("orion.analysis.pg_readonly")


def open_readonly_connection(dsn: str):
    try:
        import psycopg2
    except Exception:  # pragma: no cover
        logger.error("psycopg2 unavailable; cannot open DB session")
        return None
    try:
        conn = psycopg2.connect(dsn)
    except Exception:
        logger.error("failed to connect to postgres", exc_info=True)
        return None
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SET default_transaction_read_only = on;")
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
