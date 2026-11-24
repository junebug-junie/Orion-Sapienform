"""The promise adapter: a concern a chat turn opened because Orion promised something (``turn:<correlation_id>``).

Answers, from the concern's ``event_ref``, what the carry turn needs (orion/prime/intention.py):

* **horizon**: the promise's time + 72 h (the intention ledger's horizon, design 2026-11-11 D2; Juniper accepted
  72 h for intentions on 2026-11-13). The promise's time is the ``harness_turn_trace`` row's ``created_at``.
* **prompt**: the exact quotes from ``harness_turn_trace.run_artifact.reflection.promises_made`` (offers left
  out, as the hand-back counts them), plus the one originating exchange from ``chat_history_log`` (Juniper's
  prompt and Orion's reply). Nothing from any other thread.
* **route**: back into that chat thread (``chat_history_log.session_id``).
* **take-on refusal** (``take_on_refusal``, before any act opens): only a promise made in a chat with Juniper has a
  thread to answer in. A promise from a curiosity run, consult, letter or game sitting is refused up front with
  what to do instead (#3772: two take-ons in 11-20..23 burned their act on an order Hub had to refuse).

Loop guard (design "Dangerous failure"): a promise made in one of Orion's own sessions (``orion_*``, the carry
session ``orion_intention`` included) is refused here, on top of the hand-back opening no concern for it.

``fetch`` is the only I/O (one read-only query, asyncpg-style ``conn.fetchrow``); ``adapt`` is pure.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Optional, Union

from orion.prime.handback import WHO_WORDS, concern_turn_who
from orion.prime.state import PrimeConcernV1
from orion.schemas.intention import INTENTION_SESSION_ID, IntentionSourceV1, aware
from orion.schemas.promise_horizon import PROMISE_HORIZON_SEC

PREFIX = "turn:"
SOURCE = "turn_promise"
HORIZON_SEC = PROMISE_HORIZON_SEC  # the same clock as the promise act (#3595)
SYSTEM_SESSION_PREFIX = "orion_"
PROMPT_CAP = 3000
REPLY_CAP = 6000
MAX_QUOTES = 5

FETCH_SQL = """
SELECT t.created_at AS promised_at,
       t.run_artifact -> 'reflection' -> 'promises_made' AS promises_made,
       c.session_id, c.prompt, c.response
FROM harness_turn_trace t
LEFT JOIN chat_history_log c ON c.correlation_id = t.correlation_id
WHERE t.correlation_id = $1
ORDER BY c.created_at NULLS LAST
LIMIT 1
"""


def correlation_id(event_ref: str) -> str:
    return event_ref[len(PREFIX):]


# The one kind of turn whose promise has a chat thread to carry the reply back to (orion/prime/handback.py TurnWho).
THREAD_WHO = "chat_juniper"


def take_on_refusal(concern: PrimeConcernV1) -> Optional[str]:
    """Why Prime cannot take this promise on, or None. Pure: the concern's ``turn_who`` names the kind of turn that
    made the promise (#3785); a concern saved before that field is read from its text. Neither readable (older
    wording) passes; Hub's ``adapt`` still refuses it."""
    who = concern_turn_who(concern)
    if who is None or who == THREAD_WHO:
        return None
    return (f"that promise was made in a {WHO_WORDS.get(who, who)}, not in a chat with Juniper: there is no chat "
            "thread to carry a reply back to, so Hub would refuse the order. To act on it, send a curiosity_focus "
            "question for this concern; or drop it with your why")


def horizon_from(opened_at: datetime) -> datetime:
    """Prime's estimate before Hub answers: the concern opens seconds after the promise."""
    return opened_at + timedelta(seconds=HORIZON_SEC)


async def fetch(conn: Any, event_ref: str) -> Optional[dict[str, Any]]:
    row = await conn.fetchrow(FETCH_SQL, correlation_id(event_ref))
    return dict(row) if row is not None else None


def _promises(raw: Any) -> list[tuple[int, str]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    if not isinstance(raw, list):
        return []
    return [(i, str(p["quote"]).strip()) for i, p in enumerate(raw)
            if isinstance(p, dict) and str(p.get("quote") or "").strip() and p.get("offer") is not True]


def _cut(text: Any, cap: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= cap else text[:cap - 3].rstrip() + "..."


def adapt(event_ref: str, row: Optional[dict[str, Any]], *, why: str) -> Union[IntentionSourceV1, str]:
    """The carry turn's source, or why there is none (a plain sentence Prime is shown)."""
    corr = correlation_id(event_ref)
    if row is None:
        return f"no turn trace for {corr}: the promise cannot be read"
    promised_at = aware(row.get("promised_at"))
    if promised_at is None:
        return f"the turn trace for {corr} has no time"
    session = str(row.get("session_id") or "")
    if not session:
        return f"no chat history row for turn {corr}: there is no thread to answer in"
    if session == INTENTION_SESSION_ID or session.startswith(SYSTEM_SESSION_PREFIX):
        return (f"the promise was made in my own session {session}, not in a chat with Juniper "
                "(a carry turn's promise opens nothing: loop guard)")
    quotes = _promises(row.get("promises_made"))[:MAX_QUOTES]
    if not quotes:
        return f"the turn trace for {corr} records no promise (offers are not promises)"
    said = "\n".join(f'- "{q}"' for _, q in quotes)
    prompt = (
        "I am carrying out something I told Juniper I would do. Nobody asked me to now: I chose to take it on.\n\n"
        f"What I said, word for word, in our chat at {promised_at.strftime('%Y-%m-%d %H:%M UTC')}:\n{said}\n\n"
        f"Why I took it on now: {_cut(why, 600)}\n\n"
        "The exchange it came from (the only part of that chat I need):\n"
        f"Juniper: {_cut(row.get('prompt'), PROMPT_CAP) or '(no text)'}\n"
        f"Me: {_cut(row.get('response'), REPLY_CAP) or '(no text)'}\n\n"
        "Do what I said I would do, now, as well as I honestly can. Times in what I said are relative to when I "
        "said it. Then write the reply I will leave in that chat for Juniper: what I did and what I found, "
        "plainly, as myself. If I could not do it, I say so and why. I make no new promise in this reply."
    )
    return IntentionSourceV1(
        source=SOURCE, horizon_at=promised_at + timedelta(seconds=HORIZON_SEC), prompt=prompt, route="chat_thread",
        route_target=session[:200], quote_ref=f"harness_turn_trace:{corr}#promises_made[{quotes[0][0]}]",
        retrieval_query=_cut(quotes[0][1], 1000), promised_at=promised_at)


__all__ = ["FETCH_SQL", "HORIZON_SEC", "PREFIX", "SOURCE", "THREAD_WHO", "adapt", "correlation_id", "fetch", "horizon_from",
           "take_on_refusal"]
