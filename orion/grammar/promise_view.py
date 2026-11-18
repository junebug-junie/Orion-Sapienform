"""Read side of the intention ledger: which promises are open, and which offers wait on Juniper.

Shared by orion-hub (the 5b judge's view, orion/hub/promise_ledger.py) and orion-thought (the stance
cue, services/orion-thought/app/open_promises_cue.py, OFF until #3531). Read-only SQL over
``grammar_acts`` (the promise acts), ``harness_turn_trace`` (the quotes: the act holds none) and
``chat_history_log`` (the previous turn of a session). Sync SQLAlchemy; callers run it on a worker
thread with a timeout. ``view_from_rows`` / ``offers_from_made`` are pure.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Optional

from orion.grammar.promise_acts import (
    OFFER_ACCEPT_TTL_SEC,
    PROMISE_ACTOR,
    PROMISE_SIGNAL,
    is_firm_ledger_promise,
    parse_promise_ref,
    promise_ref,
)
from orion.regulation.juniper_turns import JUNIPER_TURN_PREDICATE
from orion.schemas.harness_finalize import (
    PROMISE_ITEMS_MAX,
    PROMISE_JUDGE_VIEW_MAX,
    OpenPromiseRefV1,
    PromiseV1,
)

logger = logging.getLogger("orion.grammar.promise_view")

OPEN_PROMISES_SQL = """
SELECT act_id, test_before, opened_at
  FROM grammar_acts
 WHERE actor = :actor AND signal_id = :signal
   AND closed_at IS NULL AND opened_at IS NOT NULL AND horizon_at > now()
 ORDER BY opened_at ASC
 LIMIT :n
"""

TRACE_PROMISES_SQL = """
SELECT correlation_id, run_artifact -> 'reflection' -> 'promises_made' AS made
  FROM harness_turn_trace
 WHERE correlation_id = ANY(:corrs)
"""

# The newest earlier row of this session, whether it answered a Juniper turn, and whether it was stored
# within the offer acceptance window (OFFER_ACCEPT_TTL_SEC, #3595: one clock with the promise horizon).
# created_at is a naive timestamp on the server's clock, compared in the server's TimeZone (the act
# resolver's rule, orion/grammar/act_ledger.py _JUNIPER_REPLY_SQL).
PREVIOUS_TURN_SQL = f"""
SELECT correlation_id, ({JUNIPER_TURN_PREDICATE}) AS is_juniper,
       created_at >= (now() - make_interval(secs => :ttl)) AT TIME ZONE current_setting('TimeZone') AS within_ttl
  FROM chat_history_log
 WHERE session_id = :sid AND correlation_id IS DISTINCT FROM :corr
 ORDER BY created_at DESC
 LIMIT 1
"""


@dataclass
class PromiseView:
    open_promises: list[OpenPromiseRefV1] = field(default_factory=list)
    # (ref, quote) of the previous reply's offers; the judge sees the quotes.
    offers: list[tuple[str, str]] = field(default_factory=list)
    # Open acts whose quote could not be read (no trace row, index out of range).
    unreadable: int = 0
    # act id -> when it opened (for the stance cue's "2 days ago").
    opened_at: dict[str, Any] = field(default_factory=dict)

    @property
    def offer_quotes(self) -> list[str]:
        return [q for _, q in self.offers]


def _made_list(raw: Any) -> list[dict]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    return [x for x in raw if isinstance(x, dict)] if isinstance(raw, list) else []


def _promise(item: dict) -> Optional[PromiseV1]:
    if not isinstance(item.get("quote"), str) or not item["quote"].strip():
        return None
    try:
        return PromiseV1.model_validate(item)
    except Exception:  # noqa: BLE001 -- an unexpected stored shape is skipped, never a guess
        return None


def view_from_rows(open_rows: Iterable[dict], trace_rows: Iterable[dict]) -> tuple[list[OpenPromiseRefV1], int]:
    """Pure: open act rows + their turns' stored promises -> what the judge is shown."""
    made_by_corr = {str(r["correlation_id"]): _made_list(r.get("made")) for r in trace_rows}
    shown: list[OpenPromiseRefV1] = []
    unreadable = 0
    for row in open_rows:
        parsed = parse_promise_ref(str(row.get("test_before") or ""))
        made = made_by_corr.get(parsed[0], []) if parsed else []
        item = made[parsed[1]] if parsed and parsed[1] < len(made) else None
        promise = _promise(item) if item else None
        if promise is None:
            unreadable += 1
            continue
        condition = " ".join(promise.condition.split()) if promise.condition and promise.condition.strip() else None
        shown.append(OpenPromiseRefV1(id=str(row["act_id"]), quote=" ".join(promise.quote.split()),
                                      condition=condition))
    return shown[:PROMISE_JUDGE_VIEW_MAX], unreadable


def offers_visible(prev: Optional[dict]) -> bool:
    """Pure: whether the previous row's offers are shown to the judge. The row must exist, have answered
    a Juniper turn, and have been stored within ``OFFER_ACCEPT_TTL_SEC`` (``within_ttl``; a row without
    the column, from an older query, reads as inside it)."""
    return bool(prev is not None and prev.get("is_juniper") and prev.get("correlation_id")
                and prev.get("within_ttl", True) is not False)


def offers_from_made(correlation_id: str, made_raw: Any) -> list[tuple[str, str]]:
    """Pure: the firm offers one stored reply made, as (ref, quote). Items without force/done_when
    (written before #3532) are skipped: they were never gated as commitments."""
    out: list[tuple[str, str]] = []
    for i, item in enumerate(_made_list(made_raw)):
        promise = _promise(item)
        if promise is None or not promise.offer or not is_firm_ledger_promise(promise):
            continue
        out.append((promise_ref(correlation_id, i), promise.quote))
    return out[:PROMISE_ITEMS_MAX]


STATEMENT_TIMEOUT_MS = 1500


@lru_cache(maxsize=4)
def _engine_for(uri: str) -> Any:
    """One small pool per process and URI (built on first use, then reused: a pool per call left
    connections open until garbage collection, services/orion-hub/tests/test_route_engine_reuse.py)."""
    from sqlalchemy import create_engine

    # Bounded at the database too: the caller's timeout stops waiting but cannot stop a worker thread, so
    # a slow or hung DB must fail fast here or every turn would strand one thread and one connection.
    return create_engine(uri, pool_pre_ping=True, pool_size=2, max_overflow=2, pool_timeout=1,
                         connect_args={"connect_timeout": 2,
                                       "options": f"-c statement_timeout={STATEMENT_TIMEOUT_MS}"})


def _engine() -> Any:
    uri = os.getenv("POSTGRES_URI", "").strip()
    return _engine_for(uri) if uri else None


def fetch_promise_view(session_id: Optional[str], correlation_id: str, *, open_too: bool = True) -> Optional[PromiseView]:
    """Sync. None when nothing could be read (no POSTGRES_URI, or a read error)."""
    from sqlalchemy import text

    try:
        engine = _engine()
        if engine is None:
            return None
        view = PromiseView()
        with engine.connect() as conn:
            if open_too:
                open_rows = [dict(r) for r in conn.execute(text(OPEN_PROMISES_SQL), {
                    "actor": PROMISE_ACTOR, "signal": PROMISE_SIGNAL, "n": PROMISE_JUDGE_VIEW_MAX}).mappings()]
                corrs = sorted({p[0] for p in (parse_promise_ref(str(r.get("test_before") or "")) for r in open_rows) if p})
                trace_rows = [dict(r) for r in conn.execute(text(TRACE_PROMISES_SQL), {"corrs": corrs}).mappings()] if corrs else []
                view.open_promises, view.unreadable = view_from_rows(open_rows, trace_rows)
                view.opened_at = {str(r["act_id"]): r.get("opened_at") for r in open_rows}
            sid = str(session_id or "").strip()
            if sid:
                prev = conn.execute(text(PREVIOUS_TURN_SQL), {"sid": sid, "corr": str(correlation_id),
                                                              "ttl": OFFER_ACCEPT_TTL_SEC}).mappings().first()
                if offers_visible(dict(prev) if prev is not None else None):
                    rows = conn.execute(text(TRACE_PROMISES_SQL), {"corrs": [str(prev["correlation_id"])]}).mappings().all()
                    if rows:
                        view.offers = offers_from_made(str(prev["correlation_id"]), rows[0]["made"])
        return view
    except Exception as exc:  # noqa: BLE001 -- fail-open: the judge sees no promises this turn
        logger.warning("promise_view_read_failed corr=%s err=%s: %s", correlation_id, type(exc).__name__, exc)
        return None


__all__ = [
    "OPEN_PROMISES_SQL", "PREVIOUS_TURN_SQL", "PromiseView", "TRACE_PROMISES_SQL", "fetch_promise_view",
    "offers_from_made", "offers_visible", "view_from_rows",
]
