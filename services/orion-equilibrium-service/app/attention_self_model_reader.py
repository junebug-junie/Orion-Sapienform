"""Read-only reader for `substrate_attention_self_model` tick history.

First Postgres access anywhere in orion-equilibrium-service's `app/` -- this
service was bus/Redis/FalkorDB only before the insight/flow gates. The table is
written every ~30s by orion-substrate-runtime's `_attention_self_model_tick()`
(PR #1459, gated `SUBSTRATE_ATTENTION_SELF_MODEL_TICK_ENABLED`); this service
only ever reads it and must never write it.

Connection/query contract deliberately mirrors
`scripts/analysis/measure_attention_self_model_confidence_baseline.py` (same
table, same `self_model_json`/`generated_at` columns, same
`SET default_transaction_read_only = on` enforcement) so the offline
measurement pass and the live gate cannot silently diverge on what they read.

Fail-open by design: every failure path returns an empty list, so a Postgres
outage degrades the insight/flow gates to "never fires" rather than crash-
looping the service or blocking the bus consumer.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from typing import Any

from orion.substrate.metacog_trigger_signals import ConfidenceSample

logger = logging.getLogger("orion.equilibrium.attention_self_model_reader")

_CONFIDENCE_FIELD = "prediction_error_confidence"


def _finite_float_or_none(value: object) -> float | None:
    """`isinstance(value, (int, float))` alone is not enough: `bool` subclasses
    `int` (a stray True would become 1.0), and `json.loads` accepts non-standard
    `NaN`/`Infinity` tokens -- a NaN would pass an isinstance check and then
    compare False against every gate threshold, silently mis-evaluating the
    window instead of being skipped. Same guard as the analysis script's own.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def parse_confidence_samples(rows: list[tuple[Any, Any]]) -> list[ConfidenceSample]:
    """Convert raw `(self_model_json, generated_at)` rows into ordered samples.

    Accepts rows in *either* order and always returns oldest -> newest, which is
    the ordering both detectors in orion/substrate/metacog_trigger_signals.py
    require. Rows with a missing/non-finite confidence or a null timestamp are
    dropped -- a tick that never recorded the field is not evidence of anything.
    """
    samples: list[ConfidenceSample] = []
    for payload, generated_at in rows:
        if generated_at is None:
            continue
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                continue
        if not isinstance(payload, dict):
            continue
        value = _finite_float_or_none(payload.get(_CONFIDENCE_FIELD))
        if value is None:
            continue
        ts = generated_at
        if isinstance(ts, datetime) and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        samples.append(ConfidenceSample(generated_at=ts, value=value))

    samples.sort(key=lambda s: s.generated_at)
    return samples


class AttentionSelfModelReader:
    """Caches one psycopg2 connection across polls, reconnecting on failure."""

    def __init__(self, *, dsn: str) -> None:
        self._dsn = dsn
        self._conn: Any = None

    def _connect(self) -> Any:
        try:
            import psycopg2
        except Exception:
            logger.exception("attention_self_model_reader_psycopg2_unavailable")
            return None
        try:
            conn = psycopg2.connect(self._dsn)
            conn.autocommit = True
            with conn.cursor() as cur:
                # Enforced, not assumed: this reader must never be able to write
                # a table another service exclusively owns.
                cur.execute("SET default_transaction_read_only = on;")
            return conn
        except Exception:
            logger.exception("attention_self_model_reader_connect_failed")
            return None

    def _get_conn(self) -> Any:
        conn = self._conn
        # psycopg2 sets `.closed` non-zero once the socket is gone (e.g. Postgres
        # restarted between polls); reusing it would raise on every future poll.
        if conn is not None and getattr(conn, "closed", 0):
            self._conn = None
            conn = None
        if conn is None:
            conn = self._connect()
            self._conn = conn
        return conn

    def close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def fetch_recent_samples(self, *, limit: int) -> list[ConfidenceSample]:
        """Newest `limit` ticks, returned oldest -> newest. `[]` on any failure.

        Queries `ORDER BY generated_at DESC LIMIT n` (newest-first) rather than
        the analysis script's `ASC` because the live gate only ever cares about
        the trailing window, and an ASC+LIMIT would return the *oldest* n rows
        of an ever-growing table -- i.e. would freeze on 2026-07-29 forever.
        `parse_confidence_samples` restores ascending order for the detectors.
        """
        if limit <= 0:
            return []
        conn = self._get_conn()
        if conn is None:
            return []
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT self_model_json, generated_at
                    FROM substrate_attention_self_model
                    ORDER BY generated_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
        except Exception:
            logger.exception("attention_self_model_reader_query_failed")
            # Drop the connection so the next poll reconnects instead of
            # retrying forever against a broken session.
            self.close()
            return []
        return parse_confidence_samples(list(rows))
