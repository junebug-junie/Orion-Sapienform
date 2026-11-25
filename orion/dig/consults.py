"""`dig consult [<id>]`: open a consult (Orion + Claude -> Juniper) by the id on its Hub card.

Why (2026-11-25, Juniper): the Hub card "Orion & Claude -> Juniper" had no id, so she could not
point Orion at it in chat. The card now shows ``ref consult <id>``; this command opens it.

Reads ``orion_consult`` and ``orion_consult_turn`` (orion-hub writes both) through ReadOnlyPsql.
The role needs ``scripts/sql/2026-11-25_grant_orion_readonly_consult.sql``; without it the
command says exactly which GRANT is missing (``psql_rows``), never an empty answer.
"""

from __future__ import annotations

import re
from typing import Any

CONSULT_ID_RE = re.compile(r"[0-9a-f]{12}")
RECENT_LIMIT = 10
TURN_CHARS = 600  # per turn in the printed summary; the full text is in the written file

DECISION_WORDS = {"yes": "Juniper said yes", "no": "Juniper said no", "no_because": "Juniper said no, because"}

_CONSULT_COLUMNS = ("consult_id, source_kind, source_ref, status, ending, turns, question, answer, brief_text, "
                    "decision, decision_reason, decided_at, brief_sent_at, created_at")


def check_id(consult_id: str) -> str:
    value = str(consult_id or "").strip().lower()
    if not CONSULT_ID_RE.fullmatch(value):
        raise ValueError(f"consult id must be 12 hex characters (the 'ref consult ...' on the Hub card), "
                         f"got {consult_id[:40]!r}")
    return value


def consult_sql(consult_id: str) -> str:
    # consult_id is checked by check_id (12 hex) before it reaches SQL.
    return f"SELECT {_CONSULT_COLUMNS} FROM orion_consult WHERE consult_id = '{consult_id}'"


def turns_sql(consult_id: str) -> str:
    return ("SELECT turn_no, speaker, ending, text, created_at FROM orion_consult_turn "
            f"WHERE consult_id = '{consult_id}' ORDER BY turn_no")


def recent_sql(limit: int = RECENT_LIMIT) -> str:
    return (f"SELECT consult_id, status, ending, decision, created_at, left(question, 120) AS question "
            f"FROM orion_consult ORDER BY created_at DESC LIMIT {int(limit)}")


def _one_line(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def decision_line(row: dict[str, Any]) -> str:
    decision = (row.get("decision") or "").strip()
    if not decision:
        return "Juniper has not answered yet." if row.get("brief_sent_at") else "No brief was sent to Juniper."
    words = DECISION_WORDS.get(decision, f"Juniper answered {decision}")
    reason = (row.get("decision_reason") or "").strip()
    return f"{words}{': ' + reason if reason and decision == 'no_because' else ''} ({row.get('decided_at') or '?'})."


def render(row: dict[str, Any], turns: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"consult {row.get('consult_id')}: {row.get('status')}"
        + (f", ending {row['ending']}" if row.get("ending") else "")
        + f"; started {row.get('created_at')}; {len(turns)} turns",
        f"came from {row.get('source_kind')} {row.get('source_ref')}",
        f"question: {_one_line(row.get('question'), 400)}",
        decision_line(row),
    ]
    if row.get("brief_text"):
        lines.append(f"brief Juniper saw: {_one_line(row['brief_text'], 900)}")
    elif row.get("answer"):
        lines.append(f"answer: {_one_line(row['answer'], 600)}")
    for t in turns:
        lines.append(f"turn {t.get('turn_no')} {t.get('speaker')}"
                     + (f" ({t['ending']})" if t.get("ending") else "")
                     + f": {_one_line(t.get('text'), TURN_CHARS)}")
    return lines


def render_recent(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["no consults yet"]
    lines = [f"newest {len(rows)} consults (open one: consult <id>):"]
    for r in rows:
        state = r.get("decision") or r.get("ending") or r.get("status")
        lines.append(f"{r.get('consult_id')}  {str(r.get('created_at') or '')[:16]}  {state}: "
                     f"{_one_line(r.get('question'), 120)}")
    return lines


__all__ = ["CONSULT_ID_RE", "check_id", "consult_sql", "decision_line", "recent_sql", "render",
           "render_recent", "turns_sql"]
