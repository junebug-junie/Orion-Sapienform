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
- **Recall-seeking questions** (live miss 2026-11-25). "do you recall when I traveled to X?" got
  NONE: the general prompt read a message that names its own subject as needing no search, and
  the reranker scores a third-person memory ("Juniper headed to X") far below the raw first-person
  question. The prompt now says who is speaking and that a question about Juniper's past always
  needs a search. When the model still answers NONE, fails or times out on a message that
  ``asks_about_past``, recall builds one search itself (``lived_memory.fallback_queries``).
  The check is grammar, not topics: a memory verb in an asking form ("do you remember", "remind me
  what"), or a question about Juniper's own past ("when did I ...?", "was I ...?").
- **Failure.** Any error, refusal or timeout means "no rewrite": recall goes on with the raw
  message, exactly as in step 1 (except for the recall-seeking fallback above).

Context comes from ``chat_history_log`` (same ``session_id``, the last ``CONTEXT_TURNS`` rows
inside ``CONTEXT_WINDOW``). The prompt goes only to the local model through the gateway; orion-recall
does not log it. The rewritten searches themselves are kept in the recall debug (the recall trace).
Whether the gateway or the bus mirror keeps the prompt is decided there, not here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, List, Optional, Sequence
from uuid import uuid4

MAX_QUERIES = 3
QUERY_MAX_CHARS = 120
MESSAGE_MAX_CHARS = 800
CONTEXT_TURNS = 3
CONTEXT_TURN_CHARS = 300
CONTEXT_WINDOW = timedelta(hours=6)
MAX_TOKENS = 80

PROMPT = """You help Orion search their own memories before they reply.
The messages are from Juniper, talking to Orion. In them, "I", "me" and "my" mean Juniper, and
"you" means Orion.

Recent conversation, oldest first:
{context}

New message from Juniper:
{message}

Does the new message lean on something from the past that a memory search could find: an
event, a person, a pet, a place, a project, a game, a plan, or something said before? Use the
recent conversation to work out what words like "it", "that" or "she" point to.

A message that asks Orion to remember or recall something, or asks when, where or what happened
in Juniper's past, always needs a search, even when it names the thing itself. Search for that
thing.

If yes, write 1 to {max_queries} short search queries, 3 to 8 words each, one per line. Name the
specific thing. If the message is only a greeting, thanks or small talk with nothing to look
up, write NONE. Write only the queries, or NONE."""

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

# The person whose messages these are: first-person words in a recall-seeking question are
# rewritten to this name, because the memories are written about her in the third person.
SPEAKER = "Juniper"

# Closed-class cues only (memory verbs, auxiliaries, pronouns, wh-words): never topic words.
# A memory verb counts only in an asking form: "do you remember / recall", "remind me what /
# about", "remember when ...?". Plain statements ("I forgot to tell you"), reminders about the
# future ("remind me to call"), likenesses ("you remind me of") and store requests ("can you
# remember this") do not ask about the past.
RECALL_VERB = re.compile(r"\b(?:remember|remembers|remembered|recall|recalls|recalled|remind|reminds|reminded)\b", re.I)
_RECALL_ASK = re.compile(
    r"\b(?:do|did|does)\s+(?:you|u)\s+(?:still\s+|even\s+)?(?:remember|recall)\b"
    r"|\bremind\s+me\s+(?:what|when|where|who|how|why|which|about|of\s+(?:the|that|what|when|where|how))\b"
    r"|\b(?:remember|recall)\s+(?:when|where|what|who|how|which|the\s+time)\b[^.!?]*\?",
    re.I)
PAST_SELF = re.compile(r"\b(?:did|was|were|had)(?:n['\u2019]?t)?\s+(?:i|we)\b", re.I)
_QUESTION = re.compile(r"[^.!?]*\?")


def asks_about_past(text: str) -> bool:
    """True when the message asks Orion to remember something ("do you recall ...", "remind me
    what ...") or asks a question about Juniper's own past ("when did I ...?", "was I ...?"; the
    "?" must close the same sentence)."""
    t = str(text or "")
    if _RECALL_ASK.search(t):
        return True
    return any(PAST_SELF.search(q) for q in _QUESTION.findall(t))


# Called as ``rewriter(message, context)``; a short-message gate call adds ``short=True``.
Rewriter = Callable[..., Awaitable[List[str]]]


@dataclass(frozen=True)
class ContextTurn:
    user: str
    orion: str


def _clip(text: Any, n: int) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def build_prompt(message: str, context: Sequence[ContextTurn], *, short: bool = False) -> str:
    lines = []
    for turn in context:
        if turn.user:
            lines.append(f"User: {_clip(turn.user, CONTEXT_TURN_CHARS)}")
        if turn.orion:
            lines.append(f"Orion: {_clip(turn.orion, CONTEXT_TURN_CHARS)}")
    return (SHORT_PROMPT if short else PROMPT).format(context="\n".join(lines) or "(none)", message=_clip(message, MESSAGE_MAX_CHARS),
                         max_queries=MAX_QUERIES)


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

    def __init__(self, bus: Any, *, channel: str, route: str, timeout_sec: float, source: Any) -> None:
        self.bus = bus
        self.channel = channel
        self.route = route
        self.timeout_sec = float(timeout_sec)
        self.source = source

    async def __call__(self, message: str, context: Sequence[ContextTurn], *, short: bool = False) -> List[str]:
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
        payload = ChatRequestPayload(messages=[LLMMessage(role="user", content=build_prompt(message, context, short=short))],
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
        return parse_queries(text)


__all__ = ["BusRewriter", "ContextTurn", "PROMPT", "SHORT_PROMPT", "SPEAKER", "RewriteRefused", "Rewriter",
           "PAST_SELF", "RECALL_VERB", "asks_about_past", "build_prompt", "fetch_context", "parse_queries"]
