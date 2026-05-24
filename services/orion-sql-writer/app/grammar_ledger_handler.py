from __future__ import annotations

import logging

from app.db import get_session, remove_session
from orion.grammar.ledger import apply_grammar_event
from orion.schemas.grammar import GrammarEventV1

logger = logging.getLogger("sql-writer")


def persist_grammar_event(event: GrammarEventV1) -> bool:
    sess = get_session()
    try:
        applied = apply_grammar_event(sess, event)
        if applied:
            sess.commit()
            logger.info(
                "grammar_event_persisted event_id=%s trace_id=%s kind=%s",
                event.event_id,
                event.trace_id,
                event.event_kind,
            )
        else:
            logger.info("grammar_event_deduped event_id=%s", event.event_id)
        return applied
    except Exception:
        sess.rollback()
        raise
    finally:
        try:
            sess.close()
        finally:
            remove_session()
