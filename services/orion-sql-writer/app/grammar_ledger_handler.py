from __future__ import annotations

import logging
import threading
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, OperationalError

from app.db import get_grammar_session, grammar_engine, remove_grammar_session
from orion.grammar.ledger import apply_grammar_event, apply_grammar_trace_batch
from orion.schemas.grammar import GrammarEventV1

logger = logging.getLogger("sql-writer")

# Per-shard active connections: cancel() runs on asyncio loop, persist runs in executor.
_active_grammar_dbapi_conns: dict[int, Any] = {}
_active_grammar_conn_lock = threading.Lock()


def cancel_active_grammar_persist(shard_id: int = 0) -> bool:
    """Cancel the in-flight Postgres query for the active grammar persist on a shard."""
    with _active_grammar_conn_lock:
        conn = _active_grammar_dbapi_conns.get(shard_id)
    if conn is None:
        return False
    backend_pid: int | None = None
    try:
        backend_pid = conn.get_backend_pid()
    except Exception:
        pass
    canceled = False
    try:
        conn.cancel()
        canceled = True
    except Exception as exc:
        logger.warning("grammar_persist_cancel_failed shard=%s error=%s", shard_id, exc)
    try:
        conn.close()
    except Exception:
        pass
    if backend_pid is not None:
        try:
            with grammar_engine.connect() as admin:
                admin.execute(
                    text("SELECT pg_terminate_backend(:pid)"),
                    {"pid": backend_pid},
                )
                admin.commit()
        except Exception as exc:
            logger.warning(
                "grammar_persist_terminate_failed shard=%s pid=%s error=%s",
                shard_id,
                backend_pid,
                exc,
            )
    return canceled


def _dispose_grammar_session(sess: Any, shard_id: int) -> None:
    with _active_grammar_conn_lock:
        _active_grammar_dbapi_conns.pop(shard_id, None)
    try:
        raw = sess.connection().connection
        raw.close()
    except Exception:
        pass
    try:
        sess.rollback()
    except Exception:
        pass
    try:
        sess.close()
    finally:
        remove_grammar_session()


def persist_grammar_trace_batch(events: list[GrammarEventV1], shard_id: int = 0) -> int:
    if not events:
        return 0
    sess = get_grammar_session()
    trace_id = events[0].trace_id
    try:
        dbapi_conn = sess.connection().connection
        with _active_grammar_conn_lock:
            _active_grammar_dbapi_conns[shard_id] = dbapi_conn
        applied = apply_grammar_trace_batch(sess, events)
        if applied:
            try:
                sess.commit()
            except IntegrityError as exc:
                sess.rollback()
                logger.warning(
                    "grammar_trace_batch_integrity_conflict trace_id=%s events=%s error=%s",
                    trace_id,
                    len(events),
                    exc,
                )
                return 0
            logger.info(
                "grammar_trace_batch_persisted trace_id=%s events=%s applied=%s",
                trace_id,
                len(events),
                applied,
            )
        else:
            logger.info("grammar_trace_batch_deduped trace_id=%s events=%s", trace_id, len(events))
        return applied
    except IntegrityError as exc:
        sess.rollback()
        logger.warning(
            "grammar_trace_batch_integrity_conflict trace_id=%s error=%s",
            trace_id,
            exc,
        )
        return 0
    except OperationalError as exc:
        _dispose_grammar_session(sess, shard_id)
        if "canceling statement" in str(exc).lower() or "query canceled" in str(exc).lower():
            logger.warning(
                "grammar_trace_batch_canceled trace_id=%s error=%s",
                trace_id,
                exc,
            )
            return 0
        raise
    except Exception:
        sess.rollback()
        raise
    finally:
        with _active_grammar_conn_lock:
            _active_grammar_dbapi_conns.pop(shard_id, None)
        try:
            sess.close()
        finally:
            remove_grammar_session()


def persist_grammar_event(event: GrammarEventV1, shard_id: int = 0) -> bool:
    sess = get_grammar_session()
    try:
        dbapi_conn = sess.connection().connection
        with _active_grammar_conn_lock:
            _active_grammar_dbapi_conns[shard_id] = dbapi_conn
        applied = apply_grammar_event(sess, event)
        if applied:
            try:
                sess.commit()
            except IntegrityError as exc:
                sess.rollback()
                logger.warning(
                    "grammar_event_integrity_conflict event_id=%s trace_id=%s kind=%s error=%s",
                    event.event_id,
                    event.trace_id,
                    event.event_kind,
                    exc,
                )
                return False
            logger.info(
                "grammar_event_persisted event_id=%s trace_id=%s kind=%s",
                event.event_id,
                event.trace_id,
                event.event_kind,
            )
        else:
            logger.info("grammar_event_deduped event_id=%s", event.event_id)
        return applied
    except IntegrityError as exc:
        sess.rollback()
        logger.warning(
            "grammar_event_integrity_conflict event_id=%s trace_id=%s error=%s",
            event.event_id,
            event.trace_id,
            exc,
        )
        return False
    except OperationalError as exc:
        _dispose_grammar_session(sess, shard_id)
        if "canceling statement" in str(exc).lower() or "query canceled" in str(exc).lower():
            logger.warning(
                "grammar_event_canceled event_id=%s trace_id=%s error=%s",
                event.event_id,
                event.trace_id,
                exc,
            )
            return False
        raise
    except Exception:
        sess.rollback()
        raise
    finally:
        with _active_grammar_conn_lock:
            _active_grammar_dbapi_conns.pop(shard_id, None)
        try:
            sess.close()
        finally:
            remove_grammar_session()
