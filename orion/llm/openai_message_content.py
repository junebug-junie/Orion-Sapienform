"""Normalize OpenAI-compatible message.content (str or part list) to plain text."""

from __future__ import annotations

from typing import Any

REASONING_PART_TYPES = frozenset({"reasoning", "reasoning_text", "thinking", "analysis"})


def join_openai_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                part_type = str(part.get("type") or "").strip().lower()
                if part_type in REASONING_PART_TYPES:
                    continue
                text = part.get("text")
                if text is None:
                    text = part.get("content")
                if text is not None:
                    parts.append(str(text).strip())
            elif part is not None:
                parts.append(str(part).strip())
        return "\n".join(p for p in parts if p).strip()
    if isinstance(content, dict):
        import json

        return json.dumps(content)
    return str(content).strip()
