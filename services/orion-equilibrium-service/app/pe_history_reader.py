"""Read-only reader for ``substrate_node_prediction_error_history``.

Written by orion-substrate-runtime (one row per real observed_at advance of a
``node:substrate.*`` node, manual_migration_node_prediction_error_history_v1.sql);
already read the same way by orion-attention-runtime
(app/store.py ``fetch_node_prediction_error_history``). Feeds
``orion.attention.pe_history_cache.NodePeHistoryCache`` for the
self_signal_episode gate.

Reuses ``AttentionSelfModelReader``'s cached psycopg2 connection, its enforced
``SET default_transaction_read_only = on`` and its rate-limited failure log.
Unlike that reader, a failure RAISES: the cache's contract is "raise on a read
error, the caller decides" -- and the gate's decision is "every signal is
absent this poll", never "every signal is calm".
"""

from __future__ import annotations

from datetime import datetime, timezone

from .attention_self_model_reader import AttentionSelfModelReader


class NodePeHistoryReadError(RuntimeError):
    pass


class NodePeHistoryReader(AttentionSelfModelReader):
    # A hung Postgres must not block the poll thread forever (review 2026-10-14).
    CONNECT_TIMEOUT_SEC = 10
    STATEMENT_TIMEOUT_MS = 20000

    def _connect(self):
        dsn = self._dsn
        if "connect_timeout" not in dsn:
            dsn = f"{dsn}{'&' if '?' in dsn else '?'}connect_timeout={self.CONNECT_TIMEOUT_SEC}"
        saved, self._dsn = self._dsn, dsn
        try:
            conn = super()._connect()
        finally:
            self._dsn = saved
        if conn is not None:
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SET statement_timeout = {int(self.STATEMENT_TIMEOUT_MS)};")
            except Exception:
                self._log_failure("node_pe_history_reader_statement_timeout_failed")
        return conn

    def fetch_since(self, since: datetime) -> list[tuple[str, datetime, float]]:
        conn = self._get_conn()
        if conn is None:
            raise NodePeHistoryReadError("postgres unavailable")
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT node_id, observed_at, value
                    FROM substrate_node_prediction_error_history
                    WHERE observed_at >= %s
                    ORDER BY observed_at
                    """,
                    (since,),
                )
                rows = cur.fetchall()
        except Exception as exc:
            self._log_failure("node_pe_history_reader_query_failed")
            self.close()
            raise NodePeHistoryReadError(type(exc).__name__) from exc
        self._consecutive_failures = 0
        out: list[tuple[str, datetime, float]] = []
        for node_id, observed_at, value in rows:
            if observed_at is None or value is None:
                continue
            if observed_at.tzinfo is None:
                observed_at = observed_at.replace(tzinfo=timezone.utc)
            out.append((str(node_id), observed_at, float(value)))
        return out
