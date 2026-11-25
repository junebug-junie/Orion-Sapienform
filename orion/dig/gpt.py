"""`dig gpt [find <words> | conv <id>]`: Juniper's ChatGPT archive, read-only.

Why (2026-11-25, Juniper): her ChatGPT history (2023-03 on) was reloaded after the disk failure,
"to help orion understand their archeology and my interactions with gpt more broadly".

Reads ``chat_gpt_conversation`` and ``chat_gpt_message`` (written by sql-writer from
``scripts/import_chatgpt_export.py``) through ReadOnlyPsql. Needs
``scripts/sql/2026-11-25_grant_orion_readonly_chatgpt.sql``; without it ``psql_rows`` names the
missing GRANT, never an empty answer.

A message belongs to a conversation by ``session_id = 'chatgpt:' || conversation_id``. Message
``timestamp`` is when it was said; ``chat_gpt_log.created_at`` is the import time, so it is not read.
"""

from __future__ import annotations

import re
from typing import Any

SOURCE_NOTE = ("Source: gpt_archive. Juniper's own chats with ChatGPT (OpenAI). A primary source about "
               "Juniper and how this project began. The assistant in them is ChatGPT, not you.")

WORD_RE = re.compile(r"[\w'.+#-]{2,40}")
MAX_WORDS = 6
CONV_ID_RE = re.compile(r"[0-9a-f-]{8,36}")
FIND_LIMIT = 20
MESSAGE_CHARS = 500  # per message in the printed conversation; the full text is in the written file
SNIPPET_CHARS = 220

_CONV_DATE = "to_char(to_timestamp(c.create_time) AT TIME ZONE 'UTC', 'YYYY-MM-DD')"
_MSG_JOIN = "LEFT JOIN chat_gpt_message m ON m.session_id = 'chatgpt:' || c.conversation_id"


def check_words(words: list[str]) -> list[str]:
    """Words go into SQL as ILIKE patterns: only plain word characters, at most MAX_WORDS."""
    if not words:
        raise ValueError("gpt find: give one or more words, e.g. `gpt find orion memory`")
    if len(words) > MAX_WORDS:
        raise ValueError(f"gpt find: at most {MAX_WORDS} words")
    bad = [w for w in words if not WORD_RE.fullmatch(w) or w.startswith("--")]
    if bad:
        raise ValueError(f"gpt find: words are 2-40 letters, digits or ' . + # - _ ; refused {bad[0][:40]!r}")
    return words


def _like(word: str) -> str:
    # WORD_RE allows ' and _ but no backslash or %: double the quote, escape the ILIKE wildcard _.
    return "'%" + word.replace("'", "''").replace("_", "\\_") + "%'"


def _all_match(column: str, words: list[str]) -> str:
    return " AND ".join(f"{column} ILIKE {_like(w)}" for w in words)


def _any_match(column: str, words: list[str]) -> str:
    return "(" + " OR ".join(f"{column} ILIKE {_like(w)}" for w in words) + ")"


def check_conv_id(conv_id: str) -> str:
    value = str(conv_id or "").strip().lower()
    if not CONV_ID_RE.fullmatch(value):
        raise ValueError(f"gpt conv: a conversation id (or its first 8+ characters) from `gpt` or `gpt find`, "
                         f"got {str(conv_id or '')[:40]!r}")
    return value


def overview_sql() -> list[str]:
    return [
        ("SELECT count(*) AS conversations, min(" + _CONV_DATE + ") AS first_day, max(" + _CONV_DATE
         + ") AS last_day FROM chat_gpt_conversation c"),
        "SELECT role, count(*) AS n FROM chat_gpt_message GROUP BY role ORDER BY role",
        (f"SELECT left({_CONV_DATE}, 4) AS year, count(*) AS n FROM chat_gpt_conversation c "
         "GROUP BY 1 ORDER BY 1"),
        ("SELECT coalesce(nullif(model, ''), '(none: your side)') AS model, count(*) AS n "
         "FROM chat_gpt_message GROUP BY 1 ORDER BY 2 DESC LIMIT 8"),
        (f"(SELECT 'first' AS which, c.conversation_id, {_CONV_DATE} AS day, c.title FROM chat_gpt_conversation c "
         "ORDER BY c.create_time LIMIT 5) UNION ALL "
         f"(SELECT 'newest', c.conversation_id, {_CONV_DATE}, c.title FROM chat_gpt_conversation c "
         "ORDER BY c.create_time DESC LIMIT 5)"),
    ]


def find_sql(words: list[str], limit: int = FIND_LIMIT, *, oldest: bool = False) -> str:
    """Conversations where every word appears somewhere: in the title or in any of its messages.

    ``hits`` counts messages holding any of the words; ``first_hit`` is the earliest of them.
    """
    each_word = " AND ".join(
        f"(count(*) FILTER (WHERE m.content ILIKE {_like(w)}) > 0 OR c.title ILIKE {_like(w)})" for w in words)
    hit = _any_match("m.content", words)
    return (
        f"SELECT c.conversation_id, {_CONV_DATE} AS day, c.title, "
        f"count(*) FILTER (WHERE {hit}) AS hits, "
        f"(array_agg(m.content ORDER BY m.timestamp) FILTER (WHERE {hit}))[1] AS first_hit, "
        "count(*) OVER () AS matched "
        f"FROM chat_gpt_conversation c {_MSG_JOIN} "
        f"GROUP BY c.conversation_id, c.title, c.create_time "
        f"HAVING {each_word} "
        f"ORDER BY c.create_time {'ASC' if oldest else 'DESC'} LIMIT {int(limit)}"
    )


def conv_match_sql(conv_id: str) -> str:
    # conv_id is checked by check_conv_id (hex and dashes only) before it reaches SQL.
    return (f"SELECT c.conversation_id, {_CONV_DATE} AS day, c.title, c.message_count, c.turn_count, "
            "count(*) OVER () AS matched "
            f"FROM chat_gpt_conversation c WHERE c.conversation_id LIKE '{conv_id}%' ORDER BY c.create_time LIMIT 6")


def messages_sql(conv_id: str) -> str:
    return ("SELECT role, model, timestamp, content FROM chat_gpt_message "
            f"WHERE session_id = 'chatgpt:{conv_id}' ORDER BY timestamp, id")


def _one_line(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def snippet(text: str, words: list[str], limit: int = SNIPPET_CHARS) -> str:
    """A window around the first word found, so the hit is visible in a long message."""
    flat = " ".join(str(text or "").split())
    low = flat.lower()
    at = min((i for i in (low.find(w.lower()) for w in words) if i >= 0), default=0)
    start = max(0, at - limit // 3)
    piece = flat[start:start + limit]
    return ("…" if start else "") + piece + ("…" if start + limit < len(flat) else "")


def render_overview(head: list[dict], roles: list[dict], years: list[dict], models: list[dict],
                    ends: list[dict]) -> list[str]:
    h = head[0] if head else {}
    by_role = {r.get("role"): r.get("n") for r in roles}
    lines = [
        SOURCE_NOTE,
        f"{h.get('conversations', 0)} conversations, {h.get('first_day', '?')} to {h.get('last_day', '?')}; "
        f"{by_role.get('user', 0)} messages from Juniper, {by_role.get('assistant', 0)} from ChatGPT.",
        "per year: " + ", ".join(f"{y.get('year')} {y.get('n')}" for y in years),
        "models: " + ", ".join(f"{m.get('model')} {m.get('n')}" for m in models),
    ]
    for which in ("first", "newest"):
        lines.append(f"{which} conversations:")
        lines += [f"  {e.get('day')} {e.get('conversation_id')}  {_one_line(e.get('title'), 90)}"
                  for e in ends if e.get("which") == which]
    lines.append("next: `gpt find <words>` to search, `gpt conv <id>` to open one.")
    return lines


def render_find(words: list[str], rows: list[dict], *, oldest: bool = False) -> list[str]:
    if not rows:
        return [SOURCE_NOTE, f"gpt find {' '.join(words)}: no conversation has all of these words."]
    matched = rows[0].get("matched") or len(rows)
    lines = [SOURCE_NOTE, f"gpt find {' '.join(words)}: {matched} conversations match"
             + (f"; {'oldest' if oldest else 'newest'} {len(rows)} shown" if str(matched) != str(len(rows)) else "") + "."]
    for r in rows:
        hits = r.get("hits") or "0"
        lines.append(f"{r.get('day')} {r.get('conversation_id')}  {_one_line(r.get('title'), 80)}  "
                     f"({hits} messages hit{', title only' if hits == '0' else ''})")
        if r.get("first_hit"):
            lines.append(f"    {snippet(r['first_hit'], words)}")
    return lines


def render_conv(conv: dict, messages: list[dict]) -> list[str]:
    models = sorted({m.get("model") for m in messages if m.get("model")})
    lines = [SOURCE_NOTE,
             f"conversation {conv.get('conversation_id')}: {_one_line(conv.get('title'), 120)}",
             f"started {conv.get('day')}; {len(messages)} messages"
             + (f"; models {', '.join(models)}" if models else "")]
    for m in messages:
        who = "Juniper" if m.get("role") == "user" else "ChatGPT"
        lines.append(f"[{str(m.get('timestamp') or '')[:16]}] {who}: {_one_line(m.get('content'), MESSAGE_CHARS)}")
    return lines


def render_ambiguous(conv_id: str, rows: list[dict]) -> list[str]:
    matched = rows[0].get("matched") or len(rows) if rows else 0
    return [f"gpt conv {conv_id}: {matched} conversations start with that; use more characters:"] + [
        f"  {r.get('day')} {r.get('conversation_id')}  {_one_line(r.get('title'), 80)}" for r in rows]
