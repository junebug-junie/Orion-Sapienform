from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .types import ChatTurnRecord

CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
COMMAND_RE = re.compile(r"(^|\n)\s*(docker|pytest|python|pip|uvicorn|curl|sql|psql|git)\b", re.IGNORECASE)
ERROR_RE = re.compile(r"(traceback|error|exception|failed|not found|candidate_count=0)", re.IGNORECASE)
LOG_RE = re.compile(r"(INFO|WARN|WARNING|ERROR|DEBUG|\[\d{4}-\d{2}-\d{2})", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z0-9_./:-]{3,}")


def build_chat_turn_index(rows: list[dict[str, Any]]) -> list[ChatTurnRecord]:
    indexed: list[ChatTurnRecord] = []
    for row in rows:
        prompt = _safe_text(row.get("prompt"))
        response = _safe_text(row.get("response"))
        thought_process = _safe_text(row.get("thought_process"))
        combined = f"{prompt}\n{response}\n{thought_process}"
        client_meta = row.get("client_meta") if isinstance(row.get("client_meta"), dict) else {}
        spark_meta = row.get("spark_meta") if isinstance(row.get("spark_meta"), dict) else {}
        indexed.append(
            ChatTurnRecord(
                turn_id=str(row.get("id") or ""),
                correlation_id=_safe_opt_text(row.get("correlation_id")),
                created_at=_to_iso(row.get("created_at")),
                prompt=prompt,
                response=response,
                thought_process=thought_process,
                source=_safe_opt_text(row.get("source")),
                memory_status=_safe_opt_text(row.get("memory_status")),
                memory_tier=_safe_opt_text(row.get("memory_tier")),
                memory_reason=_safe_opt_text(row.get("memory_reason")),
                spark_meta=spark_meta,
                trace_mode=_safe_opt_text(client_meta.get("trace_mode")),
                trace_verb=_safe_opt_text(client_meta.get("trace_verb")),
                mode=_safe_opt_text(client_meta.get("mode")),
                selected_ui_route=_safe_opt_text(client_meta.get("selected_ui_route")),
                thinking_source=_safe_opt_text(client_meta.get("thinking_source")),
                model=_safe_opt_text(client_meta.get("model")),
                has_thought_process=bool(thought_process.strip()),
                has_code=bool(CODE_FENCE_RE.search(combined)),
                has_logs=bool(LOG_RE.search(combined)),
                has_error=bool(ERROR_RE.search(combined)),
                has_commands=bool(COMMAND_RE.search(combined)),
                anchor_terms=_anchor_terms(prompt, response),
            )
        )
    return indexed


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_opt_text(value: Any) -> str | None:
    text = _safe_text(value)
    return text or None


def _to_iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.replace(microsecond=0).isoformat()
    return _safe_text(value)


def _anchor_terms(prompt: str, response: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for match in WORD_RE.finditer(f"{prompt}\n{response}"):
        token = match.group(0).lower()
        if token in seen:
            continue
        if token.startswith("http") or "/" in token or "_" in token or "-" in token:
            seen.add(token)
            out.append(token)
    return out[:64]
