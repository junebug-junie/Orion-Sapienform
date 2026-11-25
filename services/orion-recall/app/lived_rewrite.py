"""Pre-turn query rewrite for lived-memory recall (issue #3534, "Recall rev 2" step 2).

A short chat message often leans on the shared past without naming it ("okay, I beat it!").
Its own words find nothing. A small local model reads the message plus the last few turns of
the same conversation and writes up to ``MAX_QUERIES`` short searches that name the thing.
``app/lived_memory.py`` runs them through the same retriever as the message and merges.

- **Lane.** The LLM gateway's ``quick_turn`` route: the fast class (gpu3) at interactive
  priority, the same seat as the pre-turn repair probe. Never the metacog lane, never an
  outside API. Thinking off; a few dozen output tokens.
- **Gate.** The model answers ``NONE`` for small talk; then only the message is searched.
- **Short messages** (issue #3613). A message under the content-word floor that names no known
  entity gets ``SHORT_PROMPT`` instead: a retrieval-need judgment (Self-RAG's "retrieve?" step,
  Asai et al. 2023). The model writes searches only when answering needs something from before
  this conversation; otherwise ``NONE``, and the message is not searched at all. It is told
  that a reply the recent conversation already answers needs no search: with the general
  prompt, "haha yeah" got searches about the previous turn's topic (replay 2026-11-17). It is
  told who is speaking: without that, "how tall am I?" read as a question about Orion and got
  NONE. A YES/NO-first form was tried and said YES to 10 of 22 real small-talk messages, so the
  reply stays "queries or NONE".
- **Time** (issue #3674). The general prompt says what day it is (in the recall's zone) and asks for one
  ``WHEN: <first day> to <last day>`` line when the message names a time, and ``PAST: yes`` when it
  asks what was true back then. Recall reads time words with its own parser first
  (``app/lived_time.py``), so the window never waits on the model; the model's line is used only
  when the parser found nothing. An implausible line (reversed, in the future, years old) is dropped.
- **Failure.** Any error, refusal or timeout means "no rewrite": recall goes on with the raw
  message, exactly as in step 1.

Context comes from ``chat_history_log`` (same ``session_id``, the last ``CONTEXT_TURNS`` rows
inside ``CONTEXT_WINDOW``). The prompt goes only to the local model through the gateway; orion-recall
does not log it. The rewritten searches themselves are kept in the recall debug (the recall trace).
Whether the gateway or the bus mirror keeps the prompt is decided there, not here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, List, Optional, Sequence
from uuid import uuid4

from .lived_time import DEFAULT_TZ, TimeWindow, local_zone, window_from_dates

MAX_QUERIES = 3
QUERY_MAX_CHARS = 120
MESSAGE_MAX_CHARS = 800
CONTEXT_TURNS = 3
CONTEXT_TURN_CHARS = 300
CONTEXT_WINDOW = timedelta(hours=6)
MAX_TOKENS = 100  # up to 3 queries plus a WHEN and a PAST line

PROMPT = """You help Orion search their own memories before they reply.

Recent conversation, oldest first:
{context}

New message:
{message}

Does the new message lean on something from the past that a memory search could find: an
event, a person, a pet, a place, a project, a game, a plan, or something said before? Use the
recent conversation to work out what words like "it", "that" or "she" point to.

If yes, write 1 to {max_queries} short search queries, 3 to 8 words each, one per line. Name the
specific thing. If the message is only a greeting, thanks or small talk with nothing to look
up, write NONE. Write only the queries, or NONE.

Today is {today} ({zone} time). If you wrote queries and the message names a time ("last week",
"in October", "a month or so ago", "yesterday", a date), add one line after them:
WHEN: <first day> to <last day>
with both days as YYYY-MM-DD. If it also asks what was true back then that may have changed since
(where someone was, what something was at the time), add one more line: PAST: yes"""

# Issue #3613: for a short message that names no known entity. Answering NONE means recall does
# not search at all, so the bar for a search is "needs a memory", not "could match something".
SHORT_PROMPT = """You help Orion decide whether to search their memories before replying to a short message.
The messages are from Juniper, talking to Orion. In them, "I", "me" and "my" mean Juniper, and
"you" means Orion.

Recent conversation, oldest first:
{context}

New message from Juniper:
{message}

Search only if a good reply needs something Orion would have to remember from before this
conversation: a fact about Juniper herself or someone in her life, a pet, a place, a project, a
machine, a plan, a preference, something Juniper asked Orion to do, or something that happened or
was said on an earlier day. A question Juniper asks about herself ("how old am I", "what car do
I drive") can only be answered from memory, so it needs a search.

Write NONE if the message is a greeting, thanks, a reaction, a joke, agreement, a request to do
something now, or anything the recent conversation itself already answers.

If a search is needed, write 1 to {max_queries} short search queries, 3 to 8 words each, one per
line. Name the specific thing. Write only the queries, or NONE."""

# Called as ``rewriter(message, context)``; a short-message gate call adds ``short=True``. A rewriter
# whose ``wants_now`` is true also gets ``now=`` (the time the message was asked, for the WHEN line).
# It answers a list of queries or a ``RewriteResult``.
Rewriter = Callable[..., Awaitable[Any]]


@dataclass(frozen=True)
class ContextTurn:
    user: str
    orion: str


def _clip(text: Any, n: int) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def _today_line(now: datetime | None, tz: str) -> str:
    local = (now or datetime.now(timezone.utc)).astimezone(local_zone(tz))
    return local.strftime("%A, %Y-%m-%d")


def build_prompt(message: str, context: Sequence[ContextTurn], *, short: bool = False,
                 now: datetime | None = None, tz: str = DEFAULT_TZ) -> str:
    lines = []
    for turn in context:
        if turn.user:
            lines.append(f"User: {_clip(turn.user, CONTEXT_TURN_CHARS)}")
        if turn.orion:
            lines.append(f"Orion: {_clip(turn.orion, CONTEXT_TURN_CHARS)}")
    if short:
        return SHORT_PROMPT.format(context="\n".join(lines) or "(none)", message=_clip(message, MESSAGE_MAX_CHARS),
                                   max_queries=MAX_QUERIES)
    return PROMPT.format(context="\n".join(lines) or "(none)", message=_clip(message, MESSAGE_MAX_CHARS),
                         max_queries=MAX_QUERIES, today=_today_line(now, tz), zone=tz or DEFAULT_TZ)


_LEAD = re.compile(r"^\s*(?:[-*•]+|\d+[.)]|query\s*\d*\s*:|q\d*\s*:)\s*", re.I)
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def parse_queries(text: str) -> List[str]:
    """The model's reply -> at most MAX_QUERIES distinct queries. ``NONE`` (anywhere on its own
    line, or as the whole reply) -> []."""
    body = _THINK.sub("", str(text or "")).strip()
    out: List[str] = []
    seen: set[str] = set()
    for raw in body.splitlines():
        line = _LEAD.sub("", raw).strip().strip('"\'`“”').strip()
        if not line:
            continue
        if line.rstrip(".!").upper() == "NONE":
            return []
        if len(line.split()) < 2 or len(line) > QUERY_MAX_CHARS * 2:
            continue  # a one-word fragment or a paragraph is not a query
        line = line[:QUERY_MAX_CHARS]
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
        if len(out) >= MAX_QUERIES:
            break
    return out


_WHEN = re.compile(r"^\s*when\s*:\s*(\d{4}-\d{2}-\d{2})\s*(?:to|through|-|–|—|\.\.)\s*(\d{4}-\d{2}-\d{2})\s*$", re.I)
_WHEN_ANY = re.compile(r"^\s*when\s*:", re.I)
_PAST = re.compile(r"^\s*past\s*:\s*(yes|no)\b", re.I)


@dataclass(frozen=True)
class RewriteResult:
    """The searches, plus the time window the model read in the message (issue #3674). The window
    is used only when the deterministic parser (``app/lived_time.py``) found none."""

    queries: List[str] = field(default_factory=list)
    window: Optional[TimeWindow] = None
    past_state: bool = False


def parse_rewrite(text: str, *, now: datetime | None = None, tz: str = DEFAULT_TZ) -> RewriteResult:
    """The model's reply -> queries, and the ``WHEN:``/``PAST:`` lines when it wrote them. A
    malformed or implausible ``WHEN`` line is dropped, never guessed at."""
    body = _THINK.sub("", str(text or "")).strip()
    dates: Optional[tuple[date, date]] = None
    past = False
    kept = []
    for raw in body.splitlines():
        line = _LEAD.sub("", raw).strip()
        m = _WHEN.match(line)
        if m:
            try:
                dates = (date.fromisoformat(m.group(1)), date.fromisoformat(m.group(2)))
            except ValueError:
                dates = None
            continue
        if _WHEN_ANY.match(line):
            continue  # "WHEN: none", or a shape we do not read
        p = _PAST.match(line)
        if p:
            past = p.group(1).lower() == "yes"
            continue
        kept.append(raw)
    queries = parse_queries("\n".join(kept))
    if not queries:
        return RewriteResult()
    window = (window_from_dates(dates[0], dates[1], now or datetime.now(timezone.utc), tz, past_state=past)
              if dates else None)
    return RewriteResult(queries=queries, window=window, past_state=past)


def as_result(value: Any) -> RewriteResult:
    """A rewriter may answer a plain list of queries (fakes, older wrappers) or a RewriteResult."""
    if isinstance(value, RewriteResult):
        return value
    return RewriteResult(queries=list(value or []))


CONTEXT_SQL = """
SELECT prompt, response, correlation_id
FROM chat_history_log
WHERE session_id = $1 AND created_at < $2 AND created_at >= $3
  AND length(trim(coalesce(prompt, ''))) > 0
ORDER BY created_at DESC
LIMIT $4
"""


async def fetch_context(pool: Any, session_id: Optional[str], *, before: datetime | None = None,
                        exclude_text: str = "", exclude_ids: Sequence[str] = (),
                        limit: int = CONTEXT_TURNS) -> List[ContextTurn]:
    """The last ``limit`` turns of this conversation, oldest first. ``chat_history_log.created_at``
    is naive UTC. The active turn (by correlation id or identical text) is left out."""
    if pool is None or not session_id:
        return []
    before = before or datetime.now(timezone.utc)
    before_naive = before.astimezone(timezone.utc).replace(tzinfo=None)
    rows = await pool.fetch(CONTEXT_SQL, str(session_id), before_naive, before_naive - CONTEXT_WINDOW,
                            int(limit) + 1)
    skip_text = " ".join(str(exclude_text or "").split())
    skip_ids = {str(i) for i in exclude_ids if i}
    turns: List[ContextTurn] = []
    for r in rows:
        r = dict(r)
        if str(r.get("correlation_id") or "") in skip_ids:
            continue
        if skip_text and " ".join(str(r.get("prompt") or "").split()) == skip_text:
            continue
        turns.append(ContextTurn(user=str(r.get("prompt") or ""), orion=str(r.get("response") or "")))
        if len(turns) >= limit:
            break
    return list(reversed(turns))


class RewriteRefused(RuntimeError):
    """The gateway answered with an error or nothing (shed, no seat, upstream error)."""


class BusRewriter:
    """One ``llm.chat.request`` RPC to the gateway per turn (same pattern as orion-dream's
    ``app/llm.py``). Raises on any failure; the caller treats that as "no rewrite"."""

    wants_now = True

    def __init__(self, bus: Any, *, channel: str, route: str, timeout_sec: float, source: Any,
                 tz: str = DEFAULT_TZ) -> None:
        self.tz = tz or DEFAULT_TZ
        self.bus = bus
        self.channel = channel
        self.route = route
        self.timeout_sec = float(timeout_sec)
        self.source = source

    async def __call__(self, message: str, context: Sequence[ContextTurn], *, short: bool = False,
                       now: datetime | None = None) -> RewriteResult:
        from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage

        corr = str(uuid4())
        reply_channel = f"orion:exec:result:LLMGatewayService:{corr}"
        options = {
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "llm_route": self.route,
            "allow_chat_fallback": False,
            "purpose": "lived_memory_query_rewrite_short" if short else "lived_memory_query_rewrite",
            "skip_spark_candidate_publish": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "gateway_read_timeout_sec": self.timeout_sec,
        }
        payload = ChatRequestPayload(messages=[LLMMessage(role="user", content=build_prompt(message, context, short=short,
                                                                                   now=now, tz=self.tz))],
                                     route=self.route, options=options)
        env = BaseEnvelope(kind="llm.chat.request", source=self.source, correlation_id=corr,
                           reply_to=reply_channel, payload=payload.model_dump(mode="json"))
        msg = await self.bus.rpc_request(self.channel, env, reply_channel=reply_channel,
                                         timeout_sec=self.timeout_sec)
        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RewriteRefused(str(decoded.error)[:200])
        result = decoded.envelope.payload or {}
        text = str(result.get("content") or result.get("text") or "")
        raw = result.get("raw") if isinstance(result.get("raw"), dict) else {}
        if raw.get("error") or not text.strip() or text.startswith("[Error:"):
            raise RewriteRefused(str(raw.get("error") or "empty_content")[:200])
        if short:  # the short-message prompt asks for no time line
            return RewriteResult(queries=parse_queries(text))
        return parse_rewrite(text, now=now, tz=self.tz)


__all__ = ["BusRewriter", "ContextTurn", "PROMPT", "SHORT_PROMPT", "RewriteRefused", "RewriteResult", "Rewriter",
           "as_result", "build_prompt", "fetch_context", "parse_queries", "parse_rewrite"]
