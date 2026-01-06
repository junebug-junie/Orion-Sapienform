from __future__ import annotations

from typing import Iterable, List

from orion.core.contracts.recall import MemoryItemV1


def render_items(items: Iterable[MemoryItemV1], budget_tokens: int) -> str:
    """
    Render memory items into a compact bullet list suitable for prompts.
    """
    lines: List[str] = []
    tokens_used = 0
    for item in items:
        snippet = (item.snippet or "").strip().replace("\n", " ")
        if not snippet:
            continue
        prefix = f"[{item.source}"
        if item.source_ref:
            prefix += f":{item.source_ref}"
        prefix += "]"
        line = f"- {prefix} {snippet}"
        est_tokens = max(len(line.split()), 1)
        if tokens_used + est_tokens > budget_tokens:
            break
        tokens_used += est_tokens
        lines.append(line)
    return "\n".join(lines)
